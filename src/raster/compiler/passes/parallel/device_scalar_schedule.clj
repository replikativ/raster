(ns raster.compiler.passes.parallel.device-scalar-schedule
  "Canonical physical schedule for one retained TypedSOAC scalar equation.

   This namespace owns no dispatch policy. It projects already-validated ScalarIR to the existing
   extent-one SegMap/KernelBody path and checks that projection exactly. The result must already
   carry the resident-singleton representation; initialized producer storage and observable
   source-result projection remain later activation obligations."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn- materialize-locals
  [locals body]
  (if (seq locals)
    (list 'let*
          (vec
           (mapcat (fn [{:keys [id dtype init]}]
                     (let [tag (dtype/scalar-tag-for-dtype dtype)]
                       [(with-meta id {:raster.type/tag tag})
                        (list tag init)]))
                   locals))
          body)
    body))

(defn- fresh-id
  [reserved stem]
  (first (remove reserved
                 (map #(symbol (str "rstr_device_scalar_" stem "_" %)) (range)))))

(defn- default-value-facets? [value]
  (and (nil? (:logical-layout value))
       (nil? (:placement value)) (nil? (:sharding value))
       (nil? (:ownership value)) (empty? (:effects value))
       (empty? (:attributes value))))

(defn- tensor-capture?
  [value]
  (and (= :tensor (:kind value)) (seq (:shape value))
       (dtype/known? (:dtype value))
       (= (dtype/canon (:dtype value)) (:dtype value))
       (= {:kind :plain} (:representation value))
       (contains? #{nil :device} (:memory-space value))
       (default-value-facets? value)))

(defn- scalar-capture?
  [value]
  (and (= :tensor (:kind value)) (= [] (:shape value))
       (dtype/known? (:dtype value))
       (= (dtype/canon (:dtype value)) (:dtype value))
       (= {:kind :plain} (:representation value))
       (default-value-facets? value)
       ;; A rank-zero resident scalar is physically a pointer, not a scalar ABI value. Supporting
       ;; that capture requires the existing explicit resident-load transport, never rank guessing.
       (nil? (:memory-space value))))

(defn projection
  "Project one validated ScalarIR equation to its exact extent-one SegMap fields."
  [program]
  (let [program (dialect/validate! program)
        equations (dialect/equations program)
        equation (first equations)
        [_ equation-id results] equation
        {:keys [kind attributes captures lambda]} (when equation
                                                    (dialect/operation-parts equation))
        {:keys [parameters locals body-results]}
        (if lambda (dialect/lambda-parts lambda)
            {:parameters [] :locals [] :body-results []})
        facts (dialect/facts program)
        values (:values facts)
        equation-facts (get-in facts [:equations equation-id])
        result (first results)
        result-value (get values result)
        physical-results (when equation (dialect/physical-results facts equation))
        capture-values (mapv values captures)
        result-representation (:representation result-value)
        resident-result? (and (= {:kind :resident-scalar-buffer :elements 1}
                                  result-representation)
                              (= :device (:memory-space result-value)))
        result-dtype (some-> result-value :dtype dtype/canon)]
    (when-not (and (= 1 (count equations)) (= 'scalar kind)
                   (= 1 (count results)) (= 1 (count body-results))
                   (= results (dialect/outputs program))
                   (= (count captures) (count parameters))
                   (= [result] physical-results)
                   (= [result-dtype] (:dtypes attributes))
                   (= :tensor (:kind result-value)) (= [] (:shape result-value))
                   (dtype/known? result-dtype) (= result-dtype (:dtype result-value))
                   resident-result? (default-value-facets? result-value)
                   (set/subset? (set (keys attributes)) #{:dtypes :attributes})
                   (empty? (:attributes attributes))
                   (empty? (:attributes equation-facts))
                   (empty? (dialect/result-storage facts equation-id))
                   (empty? (:effects equation-facts)) (empty? (:aliases equation-facts))
                   (every? #(or (tensor-capture? %) (scalar-capture? %)) capture-values))
      (throw (ex-info "device scalar lowering requires one closed typed scalar result"
                      {:reason :typed-soac-device-scalar-subset
                       :equation equation-id :kind kind :results results
                       :physical-results physical-results :captures captures})))
    (let [substitutions (zipmap parameters captures)
          locals (mapv #(update % :init (fn [init] (util/subst-syms substitutions init)))
                       locals)
          body (util/subst-syms substitutions (first body-results))
          arrays (into #{} (keep-indexed (fn [index capture]
                                           (when (tensor-capture? (nth capture-values index))
                                             capture))) captures)
          scalars (set/difference (set captures) arrays)
          reserved (into (set (keys values))
                         (filter symbol? (tree-seq coll? seq equation)))
          index (fresh-id reserved "index")
          flat-index (fresh-id (conj reserved index) "flat_index")]
      {:equation-id equation-id
       :result result
       :dtype result-dtype
       :index index
       :flat-index flat-index
       :arrays arrays
       :scalars scalars
       :locals locals
       :body body})))

(defn- semantic-fields
  [{:keys [equation-id result dtype index flat-index arrays scalars locals body]}]
  (let [space (segop/make-seg-space-nd [{:name index :bound 1}]
                                       :flat-idx flat-index
                                       :scope [arrays scalars locals body result])]
    {:id equation-id
     :space space
     :level (segop/->SegLevel :thread :virtual)
     :lambda (materialize-locals locals body)
     :scalar-region {:locals locals :result body}
     :inputs arrays
     :outputs #{result}
     :scalars scalars
     :dtype dtype
     :out-sym result
     :cast-fn nil
     :algorithm-dialect :typed-soac
     :algorithm-equation equation-id}))

(defn lower
  "Build the canonical extent-one SegMap for an already certified resident scalar result,
   without selecting it for production dispatch."
  [program device-id & {:keys [target-descriptor]}]
  (let [projected (projection program)
        fields (semantic-fields projected)
        grid (segop/compute-launch-params :map device-id 1 :dtype (:dtype projected)
                                          :target-descriptor target-descriptor)]
    (assoc (segop/map->SegMap
            (assoc (select-keys fields
                                [:id :space :level :lambda :scalar-region :inputs :outputs
                                 :scalars :dtype :out-sym :cast-fn])
                   :grid grid))
           :algorithm-dialect :typed-soac
           :algorithm-equation (:equation-id projected))))

(defn validate-correspondence!
  "Check every computation/capture/write field against retained ScalarIR.

   Grid sizing remains target-owned, but the schedule must retain the canonical positive map-grid
   shape for literal extent one. It is not semantic evidence and cannot authorize a different
   scalar projection."
  [program operation]
  (let [expected (semantic-fields (projection program))
        actual (select-keys operation (keys expected))
        grid (:grid operation)
        block-size (:block-size grid)
        num-blocks (:num-blocks grid)
        [minimum maximum-blocks launch-blocks]
        (when (and (seq? num-blocks) (= 3 (count num-blocks))) num-blocks)
        canonical-launch
        (when (and (integer? block-size) (pos? block-size))
          (list 'int (list 'Math/ceil
                           (list '/ (list 'double 1) (double block-size)))))
        canonical-grid?
        (and (= "raster.compiler.ir.segop.KernelGrid" (some-> grid class .getName))
             (integer? block-size) (pos? block-size) (= 0 (:shared-mem-bytes grid))
             (= 'min minimum) (integer? maximum-blocks) (pos? maximum-blocks)
             (= canonical-launch launch-blocks))]
    (when-not (and (segop/seg-map? operation) canonical-grid?
                   (semantic/equivalent? expected actual))
      (throw (ex-info "scheduled device scalar is not the exact ScalarIR projection"
                      {:reason :typed-soac-device-scalar-correspondence
                       :expected expected :actual actual :grid grid})))
    operation))
