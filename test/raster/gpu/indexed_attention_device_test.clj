(ns raster.gpu.indexed-attention-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.indexed-attention-recognize :as recognize]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-route :as route]
            [raster.compiler.reference.segmented-weighted-reduction :as reference]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as array-ops]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.numeric]))

(def ^:private opencl-available?
  (delay
    (try
      (require 'raster.gpu.ocl-runtime)
      ((resolve 'raster.gpu.ocl-runtime/init!))
      true
      (catch Throwable _ false))))

(defn- chain
  []
  '(let* [raw (raster.dl.array-ops/indexed-dot
               Q K dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          weights (raster.dl.array-ops/scale-clamp-exp
                   raw (raster.numeric// 1.0 (raster.numeric/sqrt dk))
                   5.0 (clojure.core/* n-edges n-heads))
          denominator (raster.dl.array-ops/scatter-add
                       weights dst n-nodes n-edges n-heads)
          weighted (raster.dl.array-ops/scatter-mul-add
                    weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          normalized (raster.dl.array-ops/segment-div
                      weighted denominator n-nodes emb-dim n-heads 1.0e-6)]
         normalized))

(deftm resident-indexed-attention-probe
  [Q :- (Array float) K :- (Array float) V :- (Array float)
   dst :- (Array long) src :- (Array long)
   n-nodes :- Long n-edges :- Long emb-dim :- Long n-heads :- Long]
  :- (Array float)
  (let [dk (quot emb-dim n-heads)
        raw (array-ops/indexed-dot
             Q K dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
        weights (array-ops/scale-clamp-exp
                 raw (/ 1.0 (raster.numeric/sqrt dk)) 5.0 (* n-edges n-heads))
        denominator (array-ops/scatter-add weights dst n-nodes n-edges n-heads)
        weighted (array-ops/scatter-mul-add
                  weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
        normalized (array-ops/segment-div
                    weighted denominator n-nodes emb-dim n-heads 1.0e-6)]
    normalized))

(defn- test-case
  []
  (let [shape-env {'n-nodes 3 'n-edges 4 'emb-dim 5 'n-heads 2 'dk 2}
        plan (first (recognize/recognize
                     (chain) :dtype :float :accumulator-dtype :float))
        q (float-array [1 2 3 4 99, 2 1 0 -1 88, -100 100 80 -80 77])
        k (float-array [0 1 1 0 66, 1 1 2 -1 55, 100 -100 -90 90 44])
        v (float-array [1 2 3 4 33, 2 4 6 8 22, -1 1 -2 2 11])
        dst (long-array [0 0 2 2])
        src (long-array [1 1 0 2])
        buffers {'Q q 'K k 'V v 'dst dst 'src src}
        expected (reference/evaluate plan {:buffers buffers :scalars shape-env})]
    {:plan plan :shape-env shape-env :buffers buffers :expected expected}))

(defn- run-case
  [device-id]
  (let [{:keys [plan shape-env buffers expected]} (test-case)
        graph (:graph (route/route-dynamic!
                       plan
                       {:device-type :gpu
                        :subgroup-size 16
                        :max-workgroup-size 256
                        :segmented-weighted-reduction-schedule :subgroup-score-reuse}))
        output-elements (get-in plan [:output :elements])
        scalar-values
        (assoc (into {} (map (fn [[name value]]
                               [name {:type :long :value value}])
                             shape-env))
               output-elements {:type :long :value 15})]
    (gpu/with-gpu-session [session device-id]
      (gpu/alloc! session
                  {:q [:float 15 (get buffers 'Q)]
                   :k [:float 15 (get buffers 'K)]
                   :v [:float 15 (get buffers 'V)]
                   :dst [:long 4 (get buffers 'dst)]
                   :src [:long 4 (get buffers 'src)]
                   :output [:float 15 nil]})
      (let [handle (gpu/bind-kernel-graph!
                    session [:indexed-attention device-id] graph
                    {'Q :q 'K :k 'V :v 'dst :dst 'src :src 'normalized :output}
                    scalar-values)]
        (try
          (gpu/run-kernel-graph! session handle)
          (let [actual ^floats (gpu/download session :output)]
            (is (= (count expected) (alength actual)))
            (is (every? true?
                        (map (fn [wanted got]
                               (< (Math/abs (- (double wanted) (double got))) 2.0e-5))
                             expected actual)))
            (is (= [0.0 0.0 0.0 0.0 0.0]
                   (subvec (vec actual) 5 10))
                "a destination with no incoming edges is zero")
            (is (= 0.0 (double (aget actual 4)))
                "the non-head tail is explicitly zeroed"))
          (finally
            (gpu/release-kernel-graph! session handle)))))))

(deftest level-zero-indexed-attention-matches-independent-plan-oracle
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "fused indexed attention on Level Zero")
    (run-case :ze:0)))

(deftest opencl-indexed-attention-matches-independent-plan-oracle
  (if-not @opencl-available?
    (is true "OpenCL device unavailable")
    (run-case :ocl:0)))

(defn- production-case
  [descriptor total-dim]
  (let [entities 3
        edges 4
        heads 2
        components (quot total-dim heads)
        elements (* entities total-dim)
        values (fn [offset]
                 (float-array
                  (map (fn [i]
                         (float (* 0.01 (- (mod (+ i offset) 17) 8))))
                       (range elements))))
        q (values 0)
        k (values 3)
        v (values 7)
        dst (long-array [0 0 2 2])
        src (long-array [1 1 0 2])
        args [q k v dst src entities edges total-dim heads]
        shape-env {'n-nodes entities 'n-edges edges 'emb-dim total-dim
                   'n-heads heads 'dk components}
        plan (first (recognize/recognize (chain) :dtype :float
                                         :accumulator-dtype :float))
        expected (reference/evaluate
                  plan {:buffers {'Q q 'K k 'V v 'dst dst 'src src}
                        :scalars shape-env})]
    (gpu/with-gpu-session [session :ocl:0]
      (let [handle (gpu/bind-program! session descriptor args)
            selected (get-in @session [:programs :program :bounds 0
                                       :kernel-call :artifact :attributes :strategy])
            result (get (gpu/run-program! session handle args) (:result-sym descriptor))
            max-error (reduce max 0.0
                              (map #(Math/abs (- (double %1) (double %2)))
                                   expected result))]
        {:selected selected :max-error max-error}))))

(deftest resident-compiler-selects-from-runtime-component-width
  (if-not @opencl-available?
    (is true "OpenCL device unavailable")
    (let [descriptor (pipeline/compile-gpu-program
                      #'resident-indexed-attention-probe :ocl:0 :dtype :float)
          small (production-case descriptor 4)
          wide (production-case descriptor 512)]
      (is (= :indexed-segmented-reduction-reference (:selected small)))
      (is (= :indexed-segmented-reduction-subgroup-score-reuse (:selected wide)))
      (is (< (:max-error small) 2.0e-5))
      (is (< (:max-error wide) 2.0e-5)))))
