(ns raster.compiler.passes.parallel.indexed-attention-route-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-graph-call :as kgcall]
            [raster.compiler.passes.parallel.indexed-attention-recognize :as recognize]
            [raster.compiler.passes.parallel.indexed-attention-route :as route]))

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

(def ^:private shape-env
  {'n-nodes 3 'n-edges 4 'dk 2 'emb-dim 5 'n-heads 2})

(defn- plan
  ([] (plan :float))
  ([dtype]
   (first (recognize/recognize (chain) :dtype dtype :accumulator-dtype dtype))))

(deftest direct-reference-has-one-ordered-resident-abi-and-no-intermediates
  (let [{:keys [strategy reference? artifact graph schedule declines]}
        (route/route! (plan) shape-env
                      {:device-type :gpu :subgroup-size 16 :max-workgroup-size 256})]
    (is (= :indexed-segmented-reduction-reference strategy))
    (is reference?)
    (is (empty? declines))
    (is (= '[Q K V dst src normalized] (:arguments artifact)))
    (is (= '[Q K V dst src normalized] (mapv :name (:abi artifact))))
    (is (= [:input :input :input :input :input :output]
           (mapv :kind (:abi artifact))))
    (is (= [:float :float :float :long :long :float]
           (mapv :dtype (:abi artifact))))
    (is (= [5 1] (:workgroup-size schedule)))
    (is (= [1 3] (:group-count schedule)))
    (is (= [] (:temporaries artifact)))
    (is (= [] (get-in artifact [:attributes :materialized-intermediates])))
    (is (= [] (:temporaries graph)))
    (is (= [15 15 15 4 4] (mapv :elements (:inputs graph))))
    (is (= [15] (mapv :elements (:outputs graph))))
    (let [source (:source artifact)]
      (is (str/includes? source "for (long edge = 0; edge < 4L; ++edge)"))
      (is (str/includes? source "fmin((float)5.0f"))
      (is (str/includes? source "isnan(scaled) ? scaled"))
      (is (str/includes? source "denominator + (float)1.0E-6f"))
      (is (str/includes? source "feature >= 4L"))
      (is (not (str/includes? source "raw_scores")))
      (is (not (str/includes? source "weights[")))
      (is (not (str/includes? source "denominator["))))))

(deftest direct-reference-supports-scientific-double-storage
  (let [artifact (:artifact (route/route! (plan :double) shape-env))]
    (is (= [:double :double :double :long :long :double]
           (mapv :dtype (:abi artifact))))
    (is (str/starts-with? (:source artifact)
                          "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"))))

(deftest dynamic-reference-carries-shapes-through-the-ordered-abi
  (let [{:keys [artifact graph]}
        (route/route-dynamic! (plan)
                              {:device-type :gpu :subgroup-size 16
                               :max-workgroup-size 256})
        runtime-values (vec (concat (repeat 6 (Object.))
                                    (map (fn [value] {:type :long :value value})
                                         [3 4 5 2 2 15])))
        call (kcall/make artifact runtime-values)
        output-elements (get-in (plan) [:output :elements])
        graph-output (first (:outputs graph))]
    (is (= '[Q K V dst src normalized
             n_entities n_edges total_dim n_heads n_components output_elements]
           (mapv :name (:abi artifact))))
    (is (= '[Q K V dst src normalized
             n-nodes n-edges emb-dim n-heads dk
             (clojure.core/* n-nodes emb-dim)]
           (:arguments artifact)))
    (is (= [16 1] (get-in call [:geometry :workgroup-size])))
    (is (= [1 3] (get-in call [:geometry :group-count])))
    (is (= output-elements (get-in artifact [:attributes :out-elems])))
    (is (= 15 (kgcall/resolve-integer
               {output-elements {:type :long :value 15}}
               (:elements graph-output))))
    (is (str/includes? (:source artifact) "long n_entities"))
    (is (str/includes? (:source artifact) "sqrt((float)n_components)"))))

(deftest leaf-selection-depends-on-plan-descriptors-not-attention-provenance
  (let [generic-plan (-> (plan)
                         (assoc-in [:provenance :semantic-op]
                                   :generic-segmented-weighted-reduction)
                         (assoc-in [:provenance :lowering] :algebraic-diamond-recognition))
        {:keys [strategy artifact]} (route/route-dynamic! generic-plan)]
    (is (= :indexed-segmented-reduction-reference strategy))
    (is (= :generic-segmented-weighted-reduction
           (get-in artifact [:provenance :semantic-op])))))

(deftest route-declines-unproved-semantics-and-unresolved-shapes
  (testing "a different scalar region cannot silently enter the specialized leaf"
    (let [result (route/route (assoc-in (plan) [:normalization :epsilon] 0.0)
                              shape-env)]
      (is (nil? (:strategy result)))
      (is (= :indexed-segmented-reduction-plan-unsupported
             (get-in result [:declines 0 :reason])))))
  (testing "symbolic extents have to be specialized deliberately"
    (let [result (route/route (plan) (dissoc shape-env 'n-edges))]
      (is (nil? (:strategy result)))
      (is (= :indexed-attention-invalid-shape-extent
             (get-in result [:declines 0 :reason])))))
  (testing "device placement remains a routing decision"
    (let [result (route/route (plan) shape-env {:device-type :cpu})]
      (is (= :indexed-attention-requires-gpu
             (get-in result [:declines 0 :reason]))))))
