(ns raster.compiler.source-free-compilation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-map-element-test :as map-fixture]
            [raster.ad.normalized-map-sum-test :as sum-fixture]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-surface-map-test :as surface]
            [raster.compiler.backend.jvm.typed-scalar :as typed-scalar]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as gpu-link]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- pure-map
  ([] (pure-map :double))
  ([dtype]
   (let [array (av/tensor {:dtype dtype :shape '[n]})
         n (av/tensor {:dtype :long :shape []})
         equation (list '= 'map-0 '[y]
                        (list 'map {:index 'i :extent 'n} '[x] []
                              (soac/lambda-form
                               '[element]
                               [(soac/local-value
                                 'squared dtype
                                 (with-meta '(raster.numeric/* element element)
                                   {:raster.type/tag (case dtype :float 'float :double 'double)}))]
                               '[squared])))]
     (soac/make
      (soac/default-program-facts
       {:values {'x array 'n n 'y array}
        :inputs '[x n]
        :equations {'map-0 (soac/default-equation-facts)}})
      [equation] '[y]))))

(defn- synthetic-descriptor [target]
  {:device-id target})

(defn- arguments-for [compilation]
  (mapv (fn [{:keys [value]}]
          (case [(:dtype value) (empty? (:shape value))]
            [:float true] (float 0.5)
            [:double true] (double 0.5)
            [:int true] (int 2)
            [:long true] (long 2)
            [:float false] (float-array [2.0 -3.0])
            [:double false] (double-array [2.0 -3.0])))
        (get-in compilation [:semantic :attributes :invocation-plan :parameters])))

(deftest source-free-compilation-uses-the-shared-tail-without-a-frontend
  (let [typed (pure-map)
        calls (atom [])
        reject-frontend (fn [& arguments]
                          (swap! calls conj arguments)
                          (throw (ex-info "frontend must remain unreachable" {:reason :test-frontend})))
        compilation
        (with-redefs [hardware/descriptor-for synthetic-descriptor
                      pipeline/deftm-frontend-facts reject-frontend
                      pipeline/get-walked-body reject-frontend
                      pipeline/run-passes reject-frontend]
          (equation-first/compile-typed
           typed {:target :cuda:source-free
                  :schedule {:precision :f32-scalar}}))]
    (is (empty? @calls))
    (is (equation-first/equation-first-compilation? compilation))
    (is (= :double (:dtype compilation)))
    (is (nil? (:source-ns compilation)))
    (is (= :typed-parallel (get-in compilation [:semantic :dialect])))
    (is (= :scheduled-parallel (get-in compilation [:scheduled :dialect])))
    (is (seq (:kernels compilation)))
    (is (= [:raster.compiler.equation-first/typed-algorithm
            (semantic/fingerprint typed)]
           (:function compilation)))
    (is (= [:raster.compiler.equation-first/typed-compilation
            (semantic/fingerprint typed) :cuda:source-free :double
            {:schedule {:precision :f32-scalar}}]
           (:id compilation)))
    (is (= :equation-artifact-compilation-boundary
           (reason #(#'artifact/validate-compilation! compilation nil)))
        "source-free compilations do not forge a persistent source Var identity")))

(deftest source-free-lowering-materializes-the-exact-public-interface-without-host-evaluation
  (let [input (double-array [2.0 -3.0])
        compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                      (equation-first/compile-typed
                       (pure-map) {:target :ocl:source-free
                                   :schedule {:precision :f32-scalar}}))
        host-calls (atom 0)
        plan (with-redefs [typed-scalar/evaluate-invocation-step
                           (fn [& _] (swap! host-calls inc) (throw (Exception. "host step")))
                           typed-scalar/evaluate-host-equation
                           (fn [& _] (swap! host-calls inc) (throw (Exception. "host equation")))]
               (equation-first/lower compilation [input 2]))]
    (is (zero? @host-calls))
    (is (= plan (link-plan/validate! plan)))
    (is (= '[x n] (mapv :symbol
                         (get-in compilation [:semantic :attributes
                                              :invocation-plan :parameters]))))
    (let [input-node (get-in plan [:attributes :public-buffer-bindings 'x])]
      (is (identical? input (get-in plan [:nodes input-node :source]))))
    (is (zero? (get-in plan [:attributes :driver-allocations])))
    (is (= #{'y} (set (keys (get-in plan [:attributes :semantic-outputs])))))
    (is (= 1 (count (:outputs plan))))))

(deftest source-free-compilation-retains-target-policy-and-rejects-interface-overrides
  (doseq [target [:ocl:source-free :ze:source-free :cuda:source-free :hip:source-free]]
    (let [compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                        (equation-first/compile-typed
                         (pure-map :float)
                         {:target target :schedule {:precision :f32-scalar}
                          :scalar-math {:overrides {}}}))]
      (is (= target (:target compilation)))
      (is (= :float (:dtype compilation)))
      (is (seq (:kernels compilation)))
      (is (= {:overrides {}} (get-in compilation [:options :scalar-math])))))
  (doseq [options [{:dtype :double}
                   {:source-ns 'forged.source}
                   {:public-parameters '[other]}
                   {:array-types {'x :float}}
                   {:target :cuda:source-free :frontend :forged}]]
    (is (= :equation-first-typed-options
           (reason #(equation-first/compile-typed (pure-map) options)))))
  (testing "mixed retained floating facts cannot be hidden by a requested carrier"
    (let [base (pure-map)
          typed (apply list
                       (assoc (vec base) 1
                              (update (soac/facts base) :values assoc
                                      'unrelated (av/tensor {:dtype :float :shape []}))))]
      (is (some? (soac/validate! typed)))
      (is (= :equation-first-typed-dtype
             (with-redefs [hardware/descriptor-for synthetic-descriptor]
               (reason #(equation-first/compile-typed
                         typed {:target :cuda:source-free
                                :schedule {:precision :f32-scalar}}))))))))

(deftest existing-deftm-compilation-retains-its-public-boundary-after-tail-factoring
  (doseq [[operation dtype array] [[#'surface/square-double :double double-array]
                                   [#'surface/square-float :float float-array]]
          target [:cuda:source-free :hip:source-free]
          :let [input (array [2.0 -3.0])
                compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                              (equation-first/compile
                               operation {:target target :dtype dtype
                                          :schedule {:precision :f32-scalar}}))
                plan (equation-first/lower compilation [input 2])
                input-node (get-in plan [:attributes :public-buffer-bindings 'xs])]]
    (is (= dtype (:dtype compilation)))
    (is (= dtype (get-in compilation [:semantic :values 'xs :dtype])))
    (is (= 'raster.ad.normalized-surface-map-test (:source-ns compilation)))
    (is (= '[xs size]
           (mapv :symbol (get-in compilation [:semantic :attributes
                                              :invocation-plan :parameters]))))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (identical? input (get-in plan [:nodes input-node :source])))
    (is (= plan (link-plan/validate! plan)))))

(deftest canonical-normalized-phases-compile-and-materializable-inputs-lower
  (doseq [dtype [:float :double]
          :let [map-bundle (normalized/linearize-map-program
                            (#'map-fixture/map-program dtype {})
                            (the-ns 'raster.compiler.source-free-compilation-test))
                sum-bundle (#'sum-fixture/build (#'sum-fixture/source dtype) 'xs)]
          [family bundle facet] [[:map map-bundle :map-programs]
                                 [:sum sum-bundle :sum-programs]]
          phase [:primal :linear :transpose]
          :let [typed (get-in bundle [facet phase])
                borrowed-input? (some #(= :borrowed (:ownership %))
                                      (vals (select-keys (:values (soac/facts typed))
                                                        (:inputs (soac/facts typed)))))
                compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                              (equation-first/compile-typed
                               typed {:target :cuda:source-free
                                      :schedule {:precision :f32-scalar}}))]]
    (is (= dtype (:dtype compilation)) (str family " " phase " carrier"))
    (is (seq (:kernels compilation)) (str family " " phase " kernels"))
    ;; Host primitive arrays materialize as owned uploads. A retained borrowed input needs the
    ;; later same-session/module owner; this front door must expose that boundary, not retag it.
    (if borrowed-input?
      (is (= :link-value-abstract-ownership
             (reason #(equation-first/lower compilation (arguments-for compilation))))
          (str family " " phase " requires a future retained borrowed-buffer owner"))
      (let [plan (equation-first/lower compilation (arguments-for compilation))]
        (is (= plan (link-plan/validate! plan)) (str family " " phase " link plan"))))))

(deftest source-free-compilation-replays-changed-inputs-on-opencl
  (if @opencl/opencl-available?
    (let [initial (float-array [2.0 -3.0 0.5])
          changed (float-array [-1.0 4.0 -2.0])
          initial-before (vec initial)
          changed-before (vec changed)
          compilation (equation-first/compile-typed
                       (pure-map :float)
                       {:target :ocl:0 :schedule {:precision :f32-scalar}})
          plan (equation-first/lower compilation [initial 3])
          input-node (get-in plan [:attributes :public-buffer-bindings 'x])
          output-node (first (:outputs plan))
          executable (gpu-link/instantiate! plan)]
      (try
        (gpu-link/run! executable)
        (is (= [4.0 9.0 0.25]
               (vec (gpu-link/download executable output-node)))
            "the initial source-free invocation executes through the public LinkPlan")
        (gpu-link/upload! executable input-node changed)
        (gpu-link/run! executable)
        (is (= [1.0 16.0 4.0]
               (vec (gpu-link/download executable output-node))))
        (is (= initial-before (vec initial)))
        (is (= changed-before (vec changed)))
        (is (= :none (get-in compilation [:stats :fallback])))
        (is (= 1 (count (:kernels compilation))))
        (finally
          (gpu-link/close! executable)
          (gpu-link/close! executable))))
    (opencl/opencl-skip! "source-free TypedSOAC compilation and changed-input replay")))
