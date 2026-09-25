(ns raster.compiler.passes.parallel.typed-ordered-effect-map-test
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.core]
            [raster.arrays]
            [raster.abm.firms.phases :as firms-phases]
            [raster.par]))

(def ^:private extent (av/tensor {:dtype :long :shape []}))
(def ^:private vector-value (av/tensor {:dtype :float :shape '[n]}))
(def ^:private scalar-slot (av/tensor {:dtype :float :shape '[1]}))

(defn- effect-program
  []
  (let [out-result [:effect 0 0]
        total-result [:effect 0 1]
        reduction (dialect/reducing-scatter-conflict '+ :float)
        equation
        (list '= 0 [out-result total-result]
              (list 'effect-map
                    {:index 'i :extent 'n :dtypes [:float :float]
                     :iteration-order :independent}
                    '[x] [] '[out total]
                    (dialect/effect-lambda-form
                     '[element out-destination total-destination]
                     [(dialect/local-value 'shifted :float '(+ element 1.0))]
                     [(list 'effect 'out-destination :unique 'i '(> element 0.0) 'shifted)
                      (list 'effect 'total-destination reduction 0 true 'element)])))
        storage [{:destination 'out :access :write :host-return :effect}
                 {:destination 'total :access :read-write :host-return :effect}]
        equation-facts
        (assoc (dialect/default-equation-facts)
               :effects #{:memory/write}
               :aliases {out-result 'out total-result 'total}
               :attributes {:result-storage storage})]
    (dialect/make
     (dialect/default-program-facts
      {:values {'n extent 'x vector-value 'out vector-value 'total scalar-slot
                out-result vector-value total-result scalar-slot}
       :inputs '[n x]
       :equations {0 equation-facts}
       :effects #{:memory/write}})
     [equation] [])))

(deftest element-array-name-does-not-capture-the-logical-map-index
  (let [program (dialect/remap-values (effect-program) {'x 'i})
        operation (first (soac-lower/lower-typed-effect-map program :ze:0))
        execute (eval (list 'fn '[i out total n]
                            ((requiring-resolve 'raster.compiler.backend.jvm.segop-simd/compile-effect-segmap)
                             operation)))
        out (float-array [-77 -77]) total (float-array 1)]
    (execute (float-array [-1 2]) out total 2)
    (is (= [-77.0 3.0] (vec out)))
    (is (= [1.0] (vec total)))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (= :kernel-body
             (get-in (segop-opencl/generate-scheduled-segmap-kernel
                      operation :dtype :float :target-dialect target
                      :array-types {'i :float 'out :float 'total :float} :scalar-types {'n :long})
                     [:attributes :emission-route]))))))

(deftest host-effect-realization-shares-hygienic-physical-binding
  (doseq [physical ['i 'shifted 'float 'count 'long]]
    (let [program (dialect/remap-values (effect-program) {'x physical})
          equation (first (dialect/equations program))
          realized ((ns-resolve 'raster.compiler.passes.parallel.typed-soac-route 'realize-equation)
                    program equation)
          execute (eval (list 'fn [physical 'out 'total 'n] (:source realized)))
          out (float-array [-77 -77]) total (float-array 1)]
      (is (nil? (execute (float-array [-1 2]) out total 2)))
      (is (= [-77.0 3.0] (vec out)))
      (is (= [1.0] (vec total))))))

(defn- nested-operations
  [operations]
  (mapcat (fn [operation]
            (cons operation
                  (concat (nested-operations (or (:operations operation) []))
                          (nested-operations (or (:then-operations operation) []))
                          (nested-operations (or (:else-operations operation) [])))))
          operations))

(defn- normalize-fresh-thread-index
  [operation]
  (assoc-in operation [:space :flat-idx] 'thread-index))

(def ^:private mixed-effect-source
  '(let* [effect
          (raster.par/map-void!
           i n
           (let* [^float previous
                  (clojure.core/aget out (clojure.core/aget slots i))]
             (do
               (if (> (clojure.core/aget x i) 0.0)
                 (clojure.core/aset
                  out (raster.par/unique-index (clojure.core/aget slots i))
                  (float (+ previous (clojure.core/aget x i)))))
               (raster.par/atomic-add! total 0 (float (clojure.core/aget x i))))))]
         effect))

(deftest map-owner-reserves-all-local-binders-before-emission
  (let [operation (first (soac-lower/lower-typed-effect-map
                          (effect-program) :ze:0 :dtype :float))
        operation (assoc-in operation [:scalar-region :locals]
                            [{:id 'shifted :dtype :float :init '(+ 1.0 2.0)}
                             {:id 'map-value-1 :dtype :float :init '(+ shifted 3.0)}])
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                   operation :dtype :float :target-dialect :opencl-portable
                   :array-types {'x :float 'out :float 'total :float}
                   :scalar-types {'n :long})
        operations (nested-operations (get-in artifact [:attributes :kernel-body :operations]))
        ids (keep #(get-in % [:result :id]) operations)]
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (not-any? #{'map-value-1} ids))
    (is (seq ids))))

(deftest ordered-effects-lower-to-portable-kernelbody-control-and-atomics
  (let [program (effect-program)
        operation (first (soac-lower/lower-typed-effect-map
                          program :ze:0 :dtype :float))
        scheduled (:form (segop-lower/segop-lower-pass
                          (route/program-envelope program)
                          {:target-device :ze:0 :dtype :float}))
        scheduled-operation (first (get-in scheduled [:equations 0 :operations]))]
    (is (= :ordered-effects (:write-conflict operation)))
    (is (= (normalize-fresh-thread-index operation)
           (normalize-fresh-thread-index scheduled-operation))
        "whole-program scheduling dispatches the same semantic effect-map lowering")
    (is (= #{'out 'total} (:outputs operation)))
    (is (= {:locals [{:id 'shifted :dtype :float :init '(+ (clojure.core/aget x i) 1.0)}]
            :effects
            [{:destination 'out :dtype :float :conflict :unique :destination-index 'i
              :predicate '(> (clojure.core/aget x i) 0.0) :value 'shifted}
             {:destination 'total
              :dtype :float
              :conflict (dialect/reducing-scatter-conflict '+ :float)
              :destination-index 0 :predicate true
              :value '(clojure.core/aget x i)}]
            :iteration-order :independent}
           (:scalar-region operation)))
    (doseq [[target atomic]
            [[:opencl-portable "atomic_add_float"]
             [:cuda "atomicAdd"]
             [:hip "atomicAdd"]]]
      (testing (name target)
        (let [artifact (segop-opencl/generate-scheduled-segmap-kernel
                        operation :dtype :float :target-dialect target
                        :array-types {'x :float 'out :float 'total :float}
                        :scalar-types {'n :long})
              kernel-body (get-in artifact [:attributes :kernel-body])
              operations (nested-operations (:operations kernel-body))
              pointer-slots (filterv #(not= :scalar (:kind %)) (:abi artifact))]
          (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
          (is (= ['x 'out 'total] (mapv :name pointer-slots)))
          (is (= [:input :output :inout] (mapv :kind pointer-slots)))
          (is (some #(= "IfRegion" (some-> % class .getSimpleName)) operations))
          (is (some #(= "ScalarStore" (some-> % class .getSimpleName)) operations))
          (is (some #(= "AtomicRMW" (some-> % class .getSimpleName)) operations))
          (is (= {:mode :reassociated :policy :certified-reducing-scatter
                  :accumulators [{:value 'total :dtype :float
                                  :rounding :implementation-defined
                                  :policy :proof-carrying-destination}]}
                 (get-in artifact [:attributes :numerics]))
              "a proof-carrying per-destination reduction is never certified as exact")
          (is (str/includes? (:source artifact) atomic)))))))

(deftest guarded-effect-locals-lower-under-portable-kernelbody-control
  (let [source '(let* [effect
                        (raster.par/map-void!
                         i n
                         (if (clojure.core/< i active)
                           (let* [^long destination (clojure.core/rem i width)]
                             (clojure.core/aset
                              out destination
                              (clojure.core/+ (clojure.core/aget out destination)
                                              (clojure.core/aget input i))))))]
                       effect)
        result (route/attempt source :float {'out :float 'input :float}
                              {:scalar-types {'n :long 'active :long 'width :long}})
        program (-> result :program :equations first :algorithm)
        operation (first (soac-lower/lower-typed-effect-map program :ze:0 :dtype :float))]
    (is (= :independent (get-in operation [:scalar-region :iteration-order])))
    (is (some? (get-in operation [:scalar-region :effects 0 :region :predicate])))
    (doseq [target [:opencl-portable :cuda :hip]]
      (let [artifact (segop-opencl/generate-scheduled-segmap-kernel
                      operation :dtype :float :target-dialect target
                      :array-types {'out :float 'input :float}
                      :scalar-types {'n :long 'active :long 'width :long})
            operations (nested-operations (get-in artifact [:attributes :kernel-body :operations]))]
        (is (some #(= "IfRegion" (some-> % class .getSimpleName)) operations))
        (is (some #(= "AtomicRMW" (some-> % class .getSimpleName)) operations))))))

(deftest guarded-effect-loops-lower-under-portable-kernelbody-control
  (let [source '(let* [effect
                        (raster.par/map-void!
                         i rows
                         (if (clojure.core/> enabled 0)
                           (dotimes [j width]
                             (clojure.core/aset
                              out (clojure.core/+ (clojure.core/* i width) j)
                              (clojure.core/aget
                               input (clojure.core/+ (clojure.core/* i width) j))))))]
                       effect)
        result (route/attempt source :float {'out :float 'input :float}
                              {:scalar-types {'rows :long 'width :long 'enabled :int}})
        program (-> result :program :equations first :algorithm)
        operation (first (soac-lower/lower-typed-effect-map program :ze:0 :dtype :float))]
    (is (= :independent (get-in operation [:scalar-region :iteration-order])))
    (is (some? (get-in operation [:scalar-region :effects 0 :region :predicate])))
    (is (some? (get-in operation [:scalar-region :effects 0 :region :effects 0 :loop])))
    (doseq [target [:opencl-portable :cuda :hip]]
      (let [artifact (segop-opencl/generate-scheduled-segmap-kernel
                      operation :dtype :float :target-dialect target
                      :array-types {'out :float 'input :float}
                      :scalar-types {'rows :long 'width :long 'enabled :int})
            operations (nested-operations (get-in artifact [:attributes :kernel-body :operations]))
            kinds (set (map #(some-> % class .getSimpleName) operations))]
        (is (contains? kinds "IfRegion"))
        (is (contains? kinds "ForLoop"))
        (is (contains? kinds "ScalarStore"))))))

(deftest analyzed-source-selects-the-same-ordered-effect-schedule
  (let [result (route/attempt mixed-effect-source :float
                              {'x :float 'slots :int 'out :float 'total :float})
        parallel-program (:program result)
        typed-operation (-> parallel-program :equations first :algorithm
                            dialect/equations first dialect/operation-kind)
        scheduled (:form (segop-lower/segop-lower-pass
                          parallel-program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                  operation :dtype :float :target-dialect :opencl-portable
                  :array-types {'x :float 'slots :int 'out :float 'total :float}
                  :scalar-types {'n :long})
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x slots out total n] (:form jvm)))
        x (float-array [-1.0 2.0 3.0])
        out (float-array [10.0 20.0 30.0])
        total (float-array 1)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= 'effect-map typed-operation))
    (is (= :ordered-effects (:write-conflict operation)))
    (is (= ['slots 'x 'out 'total]
           (mapv :name (filterv #(not= :scalar (:kind %)) (:abi artifact)))))
    (is (= [:input :input :inout :inout]
           (mapv :kind (filterv #(not= :scalar (:kind %)) (:abi artifact)))))
    (is (str/includes? (:source artifact) "atomic_add_float"))
    (is (nil? (execute x (int-array [2 0 1]) out total 3)))
    (is (= [12.0 23.0 30.0] (mapv double out)))
    (is (= [4.0] (mapv double total)))
    (is (= 1 (get-in jvm [:stats :segop-reused])))
    (is (zero? (get-in jvm [:stats :fallback])))))

(deftest potentially-conflicting-effects-use-one-ordered-device-loop
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (do
                   (clojure.core/aset out (clojure.core/aget slots i)
                                      (float (clojure.core/aget x i)))
                   (raster.par/atomic-add! total 0 (float 1.0))))]
               effect)
        result (route/attempt source :float
                              {'x :float 'slots :int 'out :float 'total :float})
        program (:program result)
        equation (-> program :equations first :algorithm dialect/equations first)
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        artifacts
        (into {}
              (map (fn [target]
                     [target
                      (segop-opencl/generate-scheduled-segmap-kernel
                       operation :dtype :float :target-dialect target
                       :array-types {'x :float 'slots :int 'out :float 'total :float}
                       :scalar-types {'n :long})]))
              [:opencl-portable :cuda :hip])
        artifact (get artifacts :opencl-portable)
        kernel-body (get-in artifact [:attributes :kernel-body])
        operations (nested-operations (:operations kernel-body))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x slots out total n] (:form jvm)))
        out (float-array [10.0 20.0])
        total (float-array 1)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= :sequential (get-in (dialect/operation-parts equation)
                                [:attributes :iteration-order])))
    (is (= [:ordered :reduce]
           (mapv (fn [effect]
                   (let [conflict (:conflict (dialect/effect-parts effect))]
                     (if (keyword? conflict) conflict (:kind conflict))))
                 (:body-results
                  (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))))))
    (is (= :sequential (:effect-iteration-order operation)))
    (is (= [1] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [1] (get-in kernel-body [:launch :group-count])))
    (is (= :one-work-item-ordered-loop (get-in kernel-body [:schedule :strategy])))
    (is (some #(= "ForLoop" (some-> % class .getSimpleName)) operations))
    (doseq [[target emitted] artifacts]
      (testing (name target)
        (is (= :kernel-body (get-in emitted [:attributes :emission-route])))
        (is (= :one-work-item-ordered-loop
               (get-in emitted [:attributes :kernel-body :schedule :strategy])))
        (is (str/includes? (:source emitted) "for ("))))
    (is (nil? (execute (float-array [1.0 2.0 3.0]) (int-array [1 1 1]) out total 3)))
    (is (= [10.0 3.0] (mapv double out)) "the last source-order overwrite wins")
    (is (= [3.0] (mapv double total)))
    (is (= 1 (get-in jvm [:stats :segop-reused])))
    (is (zero? (get-in jvm [:stats :fallback])))))

(deftest firms-decision-queue-uses-the-certified-ordered-kernel
  (let [descriptor (pipeline/compile-gpu-program
                    #'firms-phases/execute-stay-switch-par!
                    :ze:0 :dtype :float :on-non-resident :nil)
        step (first (:steps descriptor))
        kernel-body (get-in step [:artifact :attributes :kernel-body])]
    (is (some? descriptor))
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))
    (is (= :one-work-item-ordered-loop (get-in kernel-body [:schedule :strategy])))
    (is (= :ordered (get-in kernel-body [:schedule :association])))
    (is (= [1] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [1] (get-in kernel-body [:launch :group-count])))
    (is (some #(= "ForLoop" (some-> % class .getSimpleName))
              (:operations kernel-body)))
    (is (= :sequential (get-in kernel-body [:attributes :effect-iteration-order])))))

(def ^:private row-loop-source
  ;; rms-norm-shaped: per row `r`, an ordered fold over the row whose exit divides the carry, then
  ;; a counted loop storing every element of the row at `r*feat + i`.
  '(let* [effect
          (raster.par/map-void!
           r rows
           (let* [^long offset (clojure.core/* r feat)
                  ^double ms (loop* [i 0 s 0.0]
                                (if (clojure.core/< i feat)
                                  (let* [^double v (double (clojure.core/aget
                                                            x (clojure.core/+ offset i)))]
                                    (recur (clojure.core/inc i)
                                           (clojure.core/+ s (clojure.core/* v v))))
                                  (clojure.core// s (double feat))))]
             (loop* [i 0]
               (if (clojure.core/< i feat)
                 (let* [^float v (float (clojure.core/aget x (clojure.core/+ offset i)))]
                   (clojure.core/aset out (clojure.core/+ offset i)
                                      (float (clojure.core/* v ms)))
                   (recur (clojure.core/inc i)))))))]
         effect))

(deftest row-store-loops-are-independent-effect-loops
  (let [result (route/attempt row-loop-source :float {'x :float 'out :float}
                              {:scalar-types {'rows :long 'feat :long}})
        program (:program result)
        algorithm (-> program :equations first :algorithm)
        equation (first (dialect/equations algorithm))
        {:keys [attributes lambda]} (dialect/operation-parts equation)
        {:keys [body-results]} (dialect/lambda-parts lambda)
        parts (mapv dialect/effect-parts body-results)
        leaves (dialect/effect-part-leaves parts)
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                  ;; The scheduled device is :ze:0. `r * feat` is ordinary long arithmetic,
                  ;; so this concrete artifact uses Intel OpenCL's checked-trap contract.
                  operation :dtype :float :target-dialect :opencl-intel
                  :array-types {'x :float 'out :float}
                  :scalar-types {'rows :long 'feat :long})
        portable-reason (try
                          (segop-opencl/generate-scheduled-segmap-kernel
                           operation :dtype :float :target-dialect :opencl-portable
                           :array-types {'x :float 'out :float}
                           :scalar-types {'rows :long 'feat :long})
                          nil
                          (catch clojure.lang.ExceptionInfo exception
                            (:reason (ex-data exception))))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x out rows feat] (:form jvm)))
        x (float-array [1.0 2.0 3.0 4.0])
        out (float-array 4)]
    (testing "the store loop is one effect-loop item whose store is injective across rows"
      (is (= :typed-soac (get-in result [:stats :route])))
      (is (= 'effect-map (dialect/operation-kind equation)))
      (is (= :independent (:iteration-order attributes)))
      (is (= [true] (mapv :loop parts)))
      (is (= [:unique] (mapv :conflict leaves)))
      (is (= (dialect/validate! algorithm) algorithm)))
    (testing "the scheduled SegMap carries the loop as structured data, not source"
      (is (nil? (:lambda operation)))
      (is (some :loop (get-in operation [:scalar-region :effects]))))
    (testing "the device kernel is a verified KernelBody with a nested loop"
      (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
      (is (nil? (get-in artifact [:attributes :kernel-body-decline])))
      (is (= :kernel-body-c-trap-unsupported portable-reason)
          "portable OpenCL does not pretend it can preserve checked long arithmetic")
      (is (str/includes? (:source artifact) "for (")))
    (testing "the JVM schedule executes the loop"
      (is (nil? (execute x out 2 2)))
      (is (= [2.5 5.0 37.5 50.0] (mapv double out)))
      (is (zero? (get-in jvm [:stats :fallback]))))))

(def ^:private nested-row-update-source
  '(let* [effect
          (raster.par/map-void!
           row rows
           (loop* [j 0]
             (if (clojure.core/< j columns)
               (do
                 (loop* [d 0]
                   (if (clojure.core/< d width)
                     (let* [^long address (clojure.core/+ (clojure.core/* row width) d)
                            ^float previous (clojure.core/aget out address)
                            ^float increment (float (clojure.core/+ j d))]
                       (clojure.core/aset out address
                                          (float (clojure.core/+ previous increment)))
                       (recur (clojure.core/inc d)))))
                 (recur (clojure.core/inc j))))))]
         effect))

(deftest nested-ordinary-effect-loops-retain-one-recursive-typed-region
  (let [result (route/attempt nested-row-update-source :float {'out :float}
                              {:scalar-types {'rows :long 'columns :long 'width :long}})
        program (:program result)
        algorithm (-> program :equations first :algorithm)
        equation (first (dialect/equations algorithm))
        {:keys [attributes lambda]} (dialect/operation-parts equation)
        outer (-> lambda dialect/lambda-parts :body-results first dialect/effect-parts)
        inner (-> outer :lambda dialect/lambda-parts :body-results first dialect/effect-parts)
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[out rows columns width] (:form jvm)))
        out (float-array 4)]
    (testing "source admission and ownership preserve the recursive SOAC algebra"
      (is (= :typed-soac (get-in result [:stats :route])))
      (is (= :independent (:iteration-order attributes)))
      (is (:loop outer))
      (is (:loop inner))
      (is (= [:unique] (mapv :conflict (dialect/effect-part-leaves [inner])))
          "the leaf is sequentially revisited inside one lane but disjoint across map lanes")
      (is (= (dialect/validate! algorithm) algorithm)))
    (testing "one scheduled operation lowers the same loop tree for every supported C target"
      (doseq [target [:opencl-intel :cuda :hip]]
        (let [artifact (segop-opencl/generate-scheduled-segmap-kernel
                        operation :dtype :float :target-dialect target
                        :array-types {'out :float}
                        :scalar-types {'rows :long 'columns :long 'width :long})
              operations (nested-operations
                          (get-in artifact [:attributes :kernel-body :operations]))]
          (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
          (is (= 2 (count (filter #(= "ForLoop" (some-> % class .getSimpleName)) operations)))))))
    (testing "JVM realization consumes the identical scheduled tree"
      (is (nil? (execute out 2 3 2)))
      (is (= [3.0 6.0 3.0 6.0] (vec out)))
      (is (zero? (get-in jvm [:stats :fallback]))))))

(def ^:private branch-store-then-loop-source
  '(let* [effect
          (raster.par/map-void!
           i n
           (if (< i n)
             (do
               (clojure.core/aset diary i (int 1))
               (loop* [e (int 0)]
                 (if (< e (int 2))
                   (do
                     (raster.par/atomic-add! choice 0 (int 1))
                     (recur (inc e))))))
             (clojure.core/aset diary i (int -1))))]
     effect))

(deftest branch-store-before-effect-loop-retains-the-loop
  ;; A previous branch merge projected only :stores and silently dropped the branch's :loops.
  ;; This shape occurs in the city choice kernel: a diary store precedes the episode walk.
  (let [result (route/attempt branch-store-then-loop-source :double
                              {'diary :int 'choice :int}
                              {:scalar-types {'n :long}})
        algorithm (-> result :program :equations first :algorithm)
        scheduled (:form (segop-lower/segop-lower-pass
                          (:program result) {:target-device :ze:0 :dtype :double}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                  operation :dtype :double :target-dialect :opencl-intel
                  :array-types {'diary :int 'choice :int}
                  :scalar-types {'n :long})
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[diary choice n] (:form jvm)))
        diary (int-array 3)
        choice (int-array 1)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= 2 (count (get-in operation [:scalar-region :effects]))))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (str/includes? (:source artifact) "for ("))
    (is (str/includes? (:source artifact) "choice"))
    (is (nil? (execute diary choice 3)))
    (is (= [1 1 1] (vec diary)))
    (is (= [6] (vec choice)))
    (is (zero? (get-in jvm [:stats :fallback])))
    (is (= (dialect/validate! algorithm) algorithm))))

(deftest branch-local-plain-store-never-silently-disappears
  ;; This indirect episode store still needs a shared-ownership proof. Until it has one,
  ;; compilation must decline instead of publishing a kernel that only writes the diary.
  (let [source (walk/postwalk
                (fn [form]
                  (if (= form '(raster.par/atomic-add! choice 0 (int 1)))
                    '(clojure.core/aset choice i (int 7))
                    form))
                branch-store-then-loop-source)
        result (route/attempt source :double {'diary :int 'choice :int}
                              {:scalar-types {'n :long}})]
    (if-let [program (:program result)]
      (is (str/includes? (pr-str (first (dialect/equations
                                          (-> program :equations first :algorithm))))
                         "choice"))
      (is (= :sequential-effect-continuation
             (get-in result [:declined :reason]))))))

(raster.core/deftm city-nested-search-effect!
  [loc :- (Array int), cdf :- (Array double), visits :- (Array int)
   n :- Long, len :- Long, m :- Long] :- Void
  (raster.par/map-void! i n
    (loop [e (int 0)]
      (when (< e len)
        (when (== (raster.arrays/aget loc (+ (* i len) e)) (int 2))
          (let [d (int (loop [lo (int 0) hi (int (dec m))]
                         (if (>= lo hi) lo
                             (let [mid (int (quot (+ lo hi) 2))]
                               (if (< 0.37 (raster.arrays/aget cdf mid))
                                 (recur lo mid) (recur (int (inc mid)) hi))))))]
            (raster.par/atomic-add! visits d (int 1))))
        (recur (int (inc e)))))))

(deftest nested-effect-loop-guard-and-search-retain-read-captures
  ;; City day kernels search inside an episode loop, then atomically record the result.
  ;; The guard and search are nested ordered-region expressions, not loop bounds or stores;
  ;; both still contribute tensor/scalar captures and the equation's memory-read effect.
  (let [descriptor (pipeline/compile-gpu-program #'city-nested-search-effect!
                                                 :ze:0 :dtype :double)
        step (first (:steps descriptor))
        names (set (map :name (get-in step [:artifact :abi])))]
    (is (= :map-void (:convention step)))
    (is (contains? names 'loc))
    (is (contains? names 'cdf))
    (is (contains? names 'm))
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))))

(raster.core/deftm city-effect-carry-unused!
  [loc :- (Array int), visits :- (Array int), n :- Long, len :- Long] :- Void
  (raster.par/map-void! i n
    (loop [e (int 0) anchor (int 0)]
      (when (< e len)
        (when (== (raster.arrays/aget loc (+ (* i len) e)) (int 2))
          (raster.par/atomic-add! visits 0 (int 1)))
        (recur (int (inc e)) (int (+ anchor 1)))))))

(raster.core/deftm city-effect-carry-indexed!
  [loc :- (Array int), visits :- (Array int), n :- Long, len :- Long] :- Void
  (raster.par/map-void! i n
    (loop [e (int 0) anchor (int 0)]
      (when (< e len)
        (when (== (raster.arrays/aget loc (+ (* i len) e)) (int 2))
          (raster.par/atomic-add! visits anchor (int 1)))
        (recur (int (inc e)) (int (rem (+ anchor 1) 10)))))))

(raster.core/deftm city-effect-carry-branch!
  [loc :- (Array int), visits :- (Array int), n :- Long, len :- Long] :- Void
  (raster.par/map-void! i n
    (loop [e (int 0) anchor (int (rem i 7))]
      (when (< e len)
        (let [l (int (raster.arrays/aget loc (+ (* i len) e)))]
          (when (== l (int 2))
            (raster.par/atomic-add! visits anchor (int 1)))
          (recur (int (inc e))
                 (int (if (== l (int 0)) (rem i 7)
                          (if (== l (int 1)) 9 anchor)))))))))

(raster.core/deftm city-effect-carry-search!
  [loc :- (Array int), weights :- (Array double), visits :- (Array int)
   n :- Long, len :- Long, choices :- Long] :- Void
  (raster.par/map-void! i n
    (loop [e (int 0) anchor (int (rem i 7))]
      (when (< e len)
        (let [l (int (raster.arrays/aget loc (+ (* i len) e)))]
          (when (== l (int 2))
            (let [target (* 0.37 (double (inc anchor)))
                  choice (int (loop [q (int 0) sum 0.0 hit (int -1)]
                                (if (or (>= q (int choices)) (>= hit (int 0)))
                                  (if (>= hit (int 0)) hit (int (dec choices)))
                                  (let [next-sum (+ sum (raster.arrays/aget weights q))]
                                    (recur (int (inc q)) next-sum
                                           (int (if (< target next-sum) q -1)))))))]
              (raster.par/atomic-add! visits choice (int 1))))
          (recur (int (inc e))
                 (int (if (== l (int 0)) (rem i 7)
                          (if (== l (int 1)) 9 anchor)))))))))

(raster.core/deftm city-two-effect-carry-loops!
  [loc :- (Array int), visits :- (Array int), n :- Long, len :- Long] :- Void
  (raster.par/map-void! i n
    (do
      (loop [e (int 0) anchor (int 0)]
        (when (< e len)
          (when (== (raster.arrays/aget loc (+ (* i len) e)) (int 2))
            (raster.par/atomic-add! visits anchor (int 1)))
          (recur (int (inc e)) (int (rem (+ anchor 1) 10)))))
      (loop [e (int 0) anchor (int 0)]
        (when (< e len)
          (when (== (raster.arrays/aget loc (+ (* i len) e)) (int 2))
            (raster.par/atomic-add! visits anchor (int 1)))
          (recur (int (inc e)) (int (rem (+ anchor 1) 10))))))))

(deftest effectful-counted-loop-retains-independent-carried-scalar
  (let [loc (int-array [2 0 2 1, 0 2 2 2, 1 1 2 0])]
    (doseq [[source expected]
            [[#'city-effect-carry-unused! [6 0 0 0 0 0 0 0 0 0]]
             [#'city-effect-carry-indexed! [1 1 3 1 0 0 0 0 0 0]]
             [#'city-effect-carry-branch! [2 3 0 0 0 0 0 0 0 1]]]]
      (let [descriptor (pipeline/compile-gpu-program source :ze:0 :dtype :double)
            step (first (:steps descriptor))
            kernel-body (get-in step [:artifact :attributes :kernel-body])
            loops (filter #(= "ForLoop" (some-> % class .getSimpleName))
                          (nested-operations (:operations kernel-body)))
            visits (int-array 10)]
        (is (= :map-void (:convention step)))
        (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))
        (is (= [1] (mapv (comp count :iter-args) loops))
            "the typed effect loop owns one scalar carry beside its induction index")
        (source loc visits 3 4)
        (is (= expected (vec visits)))))))

(deftest effectful-carried-episode-loop-composes-with-inner-search
  (let [descriptor (pipeline/compile-gpu-program #'city-effect-carry-search!
                                                 :ze:0 :dtype :double)
        step (first (:steps descriptor))
        names (set (map :name (get-in step [:artifact :abi])))]
    (is (= :map-void (:convention step)))
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))
    (is (every? names '[loc weights visits choices]))))

(deftest sibling-effect-carries-remain-distinct-lexical-results
  (let [descriptor (pipeline/compile-gpu-program #'city-two-effect-carry-loops!
                                                 :ze:0 :dtype :double)
        step (first (:steps descriptor))]
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))))

(raster.core/deftm city-conditional-two-arm-effects!
  [visits :- (Array int), counts :- (Array int), n :- Long] :- Void
  (raster.par/map-void! i n
    (if (== (rem i 3) 0)
      (raster.par/atomic-add! counts 2 (int 1))
      (when (< i 5)
        (raster.par/atomic-add! visits i (int 1))
        (raster.par/atomic-add! counts 1 (int 1))))))

(deftest effectful-if-arms-retain-typed-boolean-guards
  (let [descriptor (pipeline/compile-gpu-program #'city-conditional-two-arm-effects!
                                                 :ze:0 :dtype :double)
        step (first (:steps descriptor))
        visits (int-array 8)
        counts (int-array 3)]
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))
    (city-conditional-two-arm-effects! visits counts 8)
    (is (= [0 1 1 0 1 0 0 0] (vec visits)))
    (is (= [0 3 3] (vec counts)))))

(raster.core/deftm city-retail-ordered-scopes!
  [starts :- (Array int), offsets :- (Array int), visits :- (Array int),
   counts :- (Array int), n :- Long] :- Void
  (raster.par/map-void! i n
    (let [start (int (raster.arrays/aget starts i))
          end (int (raster.arrays/aget offsets (unchecked-add-int start 1)))
          key (int (+ start 3))]
      (raster.par/atomic-add! counts 0 (int 1))
      (loop [e (int start) anchor (int 0)]
        (when (< e end)
          (let [loc (int (raster.arrays/aget offsets e))]
            (when (== loc (int 2))
              (let [salt (int (+ key e))
                    lane (int (+ salt anchor))
                    base (int (+ lane loc))
                    slot (int (rem base 8))
                    alpha 1.0 beta 2.0 delta 3.0 lon 4.0 lat 5.0]
                (raster.par/atomic-add! visits
                                        (unchecked-add-int (* slot 2) anchor)
                                        (int 1))
                (raster.par/atomic-add! counts
                                        (unchecked-add-int 2 slot) (int 1))))
            (recur (int (unchecked-add-int e 1)) (int (+ anchor 1)))))))))

(deftest direct-and-loop-atomics-retain-dynamic-coordinates-and-lexical-scopes
  (let [descriptor (pipeline/compile-gpu-program #'city-retail-ordered-scopes!
                                                 :ze:0 :dtype :double)
        step (first (:steps descriptor))]
    (is (= :map-void (:convention step)))
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))
    (let [starts (int-array [0]) offsets (int-array [2 2 2])
          visits (int-array 24) counts (int-array 10)]
      (city-retail-ordered-scopes! starts offsets visits counts 1)
      (is (= [1 0 0 0 0 0 0 1 0 1] (vec counts)))
      (is (= 2 (reduce + visits))))))

(deftest nested-effect-local-shadowing-precedes-flat-type-facts
  (let [source {:locals [{:id 'rstr_local_0 :dtype :int :init 0}]
                :stores [{:out 'counts :index 0 :value 1}]
                :loops [{:index 'e :lower 'rstr_local_0 :extent 2
                         :locals []
                         :stores [{:out 'counts :index 'rstr_local_0 :value 1}]
                         :loops []
                         :order [[:region {:locals [{:id 'rstr_local_0
                                                     :dtype :double :init 2.0}]
                                            :order [[:store 0]]}]]}]
                :order [[:store 0] [:loop 0]]}
        normalized (#'frontend/alpha-rename-source-region source)
        loop (first (:loops normalized))
        inner (second (first (:order loop)))
        inner-id (-> inner :locals first :id)]
    (is (= 'rstr_local_0 (:lower loop)))
    (is (not= 'rstr_local_0 inner-id))
    (is (= :double (-> inner :locals first :dtype)))
    (is (= inner-id (-> loop :stores first :index)))
    (is (= [[:store 0] [:loop 0]] (:order normalized)))))

(deftest recursive-source-regions-never-recycle-generated-local-ids
  ;; Independent recognizer calls used to restart at rstr_local_0. An enclosing region could
  ;; then confuse a captured key with a newly bound splitmix temporary before late alpha-renaming.
  (with-bindings {#'frontend/*region-local-counter* (atom -1)}
    (let [first-region (#'frontend/store-region
                        '(let* [^long key (long 17)]
                           (raster.par/atomic-add! counts key (int 1))) 'i)
          second-region (#'frontend/store-region
                         '(let* [^long mixed (long (clojure.core/+ rstr_local_0 1))]
                            (raster.par/atomic-add! counts mixed (int 1))) 'i)
          first-id (-> first-region :locals first :id)
          second-id (-> second-region :locals first :id)]
      (is (some? first-region))
      (is (some? second-region))
      (is (not= first-id second-id))
      (is (some #{first-id} (tree-seq coll? seq (-> second-region :locals first :init))))
      (is (= first-id (-> first-region :stores first :index)))
      (is (= second-id (-> second-region :stores first :index)))))
  (with-bindings {#'frontend/*region-local-counter* (atom -1)
                  #'frontend/*region-local-source-symbols* #{'rstr_local_0}}
    (is (= 'rstr_local_1 (#'frontend/fresh-region-local-id!)))))

(deftest unchecked-int-loop-step-needs-a-proved-int-exclusive-bound
  (let [tail '(recur (int (unchecked-add-int e 1)) anchor)]
    (is (nil? (#'frontend/split-trailing-recur-many tail 'e 1 false)))
    (is (some? (#'frontend/split-trailing-recur-many tail 'e 1 true)))))

(deftest triangular-effect-domains-retain-dynamic-or-inclusive-boundaries
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 row rows
                 (loop* [j row]
                   (if (clojure.core/<= j last-column)
                     (do
                       (loop* [d 0]
                         (if (clojure.core/< d width)
                           (let* [^long address (clojure.core/+ (clojure.core/* row width) d)
                                  ^float previous (clojure.core/aget out address)]
                             (clojure.core/aset out address
                                                (float (clojure.core/+ previous 1.0)))
                             (recur (clojure.core/inc d)))))
                       (recur (clojure.core/inc j))))))]
               effect)
        result (route/attempt source :float {'out :float}
                              {:scalar-types {'rows :long 'last-column :long 'width :long}})
        program (:program result)
        algorithm (-> program :equations first :algorithm)
        equation (first (dialect/equations algorithm))
        outer (-> equation dialect/operation-parts :lambda dialect/lambda-parts
                  :body-results first dialect/effect-parts)
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[out rows last-column width] (:form jvm)))
        out (float-array 6)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= 'row (:lower outer)))
    (is (= :inclusive (:upper-bound outer)))
    (is (= :independent (-> equation dialect/operation-parts :attributes :iteration-order)))
    (doseq [[field value reason]
            [[:upper-bound :closed :typed-soac-syntax]
             [:lower 'outside :typed-soac-effect-loop]]]
      (let [broken (clojure.walk/postwalk
                    (fn [form]
                      (if (dialect/effect-loop-form? form)
                        (let [[head attributes extent lambda] form]
                          (list head (assoc attributes field value) extent lambda))
                        form))
                    algorithm)]
        (is (= reason
               (try (dialect/validate! broken) :accepted
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
    (doseq [target [:opencl-intel :cuda :hip]]
      (let [artifact (segop-opencl/generate-scheduled-segmap-kernel
                      operation :dtype :float :target-dialect target
                      :array-types {'out :float}
                      :scalar-types {'rows :long 'last-column :long 'width :long})]
        (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
        (is (some #(= :inclusive (get-in % [:attributes :upper-bound]))
                  (nested-operations
                   (get-in artifact [:attributes :kernel-body :operations]))))))
    (is (nil? (execute out 3 2 2)))
    (is (= [3.0 3.0 2.0 2.0 1.0 1.0] (vec out)))))

(deftest production-host-store-loops-preserve-the-continuation
  (let [source '(let* [effect
                      (raster.par/map-void!
                       r rows
                       (loop* [i 0]
                         (if (< i width)
                           (do (aset out (+ (* r width) i) (float i))
                               (recur (inc i))))))]
                 effect)
        result (route/attempt source :float {'out :float}
                              {:scalar-types {'rows :long 'width :long}})
        execute (eval (list 'fn '[out rows width] (get-in result [:program :source])))
        out (float-array (repeat 6 -77))]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (nil? (execute out 2 3)))
    (is (= [0.0 1.0 2.0 0.0 1.0 2.0] (vec out)))))

(deftest effect-loops-must-be-closed-over-the-loop-index
  (let [program (:program (route/attempt row-loop-source :float {'x :float 'out :float}
                                         {:scalar-types {'rows :long 'feat :long}}))
        algorithm (-> program :equations first :algorithm)
        rebound (clojure.walk/postwalk
                 (fn [form]
                   (if (and (seq? form) (= 'effect-loop (first form)))
                     (let [[_ attributes extent [_ _ region]] form]
                       (list 'effect-loop attributes extent (list 'lambda '[other] region)))
                     form))
                 algorithm)]
    (is (= :typed-soac-effect-loop
           (try (dialect/validate! rebound) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest explicit-store-emission-refuses-typed-regions
  (let [program (:program (route/attempt row-loop-source :float {'x :float 'out :float}
                                         {:scalar-types {'rows :long 'feat :long}}))
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))]
    (is (= :explicit-segmap-requires-source-lambda
           (try (segop-opencl/generate-explicit-segmap-kernel
                 operation :dtype :float :array-types {'x :float 'out :float}
                 :scalar-types {'rows :long 'feat :long})
                nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
