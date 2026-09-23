(ns raster.compiler.passes.parallel.scheduled-equation-graph-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.dl.nn :as nn]))

(def ^:private three-map-source
  '(let* [first-effect
          (raster.par/map! tmp i n float (clojure.core/aget x i))
          second-effect
          (raster.par/map! middle j n float
                           (clojure.core/+ (clojure.core/aget tmp j) 1.0))
          third-effect
          (raster.par/map! out k n float
                           (clojure.core/* (clojure.core/aget middle k) 2.0))]
     third-effect))

(def ^:private scalar-gap-source
  '(let* [first-effect
          (raster.par/map! tmp i n float (clojure.core/aget x i))
          ^{:raster.type/tag long} doubled
          (clojure.core/* (clojure.core/long n) (clojure.core/long 2))
          second-effect
          (raster.par/map! out j n float
                           (clojure.core/+ (clojure.core/aget tmp j)
                                           (clojure.core/float doubled)))]
     second-effect))

(defn- scheduled-three-maps []
  (let [options {:dtype :float :target-device :ocl:0
                 :array-types {'x :float 'tmp :float 'middle :float 'out :float}
                 :scalar-types {'n :long}}
        typed (frontend/form->program three-map-source options)
        envelope (route/program-envelope typed)]
    (:form (segop-lower/segop-lower-pass envelope options))))

(defn- scheduled-scalar-gap []
  (let [options {:dtype :float :target-device :ocl:0
                 :array-types {'x :float 'tmp :float 'out :float}
                 :scalar-types {'n :long}}
        typed (frontend/form->program scalar-gap-source options)
        envelope (route/program-envelope typed)]
    (:form (segop-lower/segop-lower-pass envelope options))))

(defn- reason-of [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest contiguous-equations-form-one-exact-semantic-source-graph
  (let [scheduled (scheduled-three-maps)
        equations (subvec (:equations scheduled) 0 2)
        {:keys [algorithm body graph]} (equation-graph/make-for-equations scheduled equations)]
    (is (= [0 1] (get-in body [:attributes :equation-region])))
    (is (= [0 1] (get-in (soac/facts algorithm) [:attributes :equation-region])))
    (is (= 2 (count (soac/equations algorithm))))
    (is (= (:results (peek equations)) (soac/outputs algorithm)))
    (is (= 2 (count (:nodes graph))))
    (is (= #{'x} (set (map :id (:inputs graph)))))
    (is (= #{'middle} (set (map :id (:outputs graph)))))
    (is (= #{'tmp} (set (map :id (:temporaries graph)))))))

(deftest equation-region-must-be-an-exact-contiguous-slice
  (let [scheduled (scheduled-three-maps)
        equations (:equations scheduled)]
    (is (= :scheduled-equation-region
           (reason-of #(equation-graph/make-for-equations
                        scheduled [(first equations) (peek equations)]))))
    (is (= :scheduled-equation-region
           (reason-of #(equation-graph/make-for-equations scheduled []))))))

(deftest pure-scalar-gap-is-hoisted-into-the-region-proof-prefix
  (let [scheduled (scheduled-scalar-gap)
        numerical (filterv (comp seq :operations) (:equations scheduled))
        {:keys [body graph]} (equation-graph/make-for-equations scheduled numerical)]
    (is (= 2 (count numerical)))
    (is (= 3 (count (:equations body))))
    (is (true? (get-in body [:equations 0 :attributes :host-only])))
    (is (= (mapv :id numerical) (mapv :id (subvec (:equations body) 1))))
    (is (= 2 (count (:nodes graph))))))

(deftest later-global-extent-becomes-a-bindable-buffer-capacity
  (let [scheduled (scheduled-three-maps)
        scheduled (-> scheduled
                      (assoc-in [:values 'later-extent] (get-in scheduled [:values 'n]))
                      (assoc-in [:values 'x :shape] ['later-extent]))
        {:keys [body graph]} (equation-graph/make-for-equation
                              scheduled (first (:equations scheduled)))
        graph-input (first (filter #(= 'x (:id %)) (:inputs graph)))]
    (is (= '[(extent x)] (get-in body [:values 'x :shape])))
    (is (= '(extent x) (:elements graph-input))
        "a caller-owned allocation remains resolvable; it never becomes unknown-dimension")))

(deftest earlier-equation-does-not-capture-a-later-global-buffer-extent
  ;; group-norm's final dense map normalizes `batch*channel-span` after three product equations.
  ;; The program-wide value table once leaked that later extent into every earlier graph ABI,
  ;; producing a forward reference at backend reconstruction time.
  (let [report (pipeline/compile-report #'nn/group-norm-jvp-dx
                                        :target-device :ocl:0 :dtype :float)]
    (is (= :typed-soac (get-in report [:route :source-dialect])))
    (is (= {:kernel-body 4} (get-in report [:emission :routes])))
    (is (empty? (get-in report [:route :declines])))))
