(ns raster.compiler.scalar-intrinsic-identity-test
  "Public compiler controls: a user function name is not a primitive semantic identity."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.backend.intrinsics :as intrinsic]
            [raster.compiler.backend.gpu.c-emit :as c]
            [raster.compiler.core.op-descriptor :as operation]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.gpu.core :as gpu]))

(deftm sin [x :- Double] :- Double (clojure.core/+ x 100.0))
(deftm custom-sine [x :- Double] :- Double (clojure.core/+ x 100.0))
(deftm identity-reference [x :- Double] :- Double (clojure.core/+ x 100.0))

(deftm sin-map!
  [a :- (Array double) out :- (Array double) n :- Long] :- (Array double)
  (par/map! out i n double (sin (clojure.core/aget a i))))

(deftm control-map!
  [a :- (Array double) out :- (Array double) n :- Long] :- (Array double)
  (par/map! out i n double (custom-sine (clojure.core/aget a i))))

(defn- compiled-source [operation]
  (let [session (atom {:device-id :ocl:0 :closed? false :kernels {}
                      :dispatches {} :kernel-cache {}})]
    ;; Only registration is stubbed. Source analysis, dispatch, inlining, typed scheduling,
    ;; ABI construction and emission run through the public compile! entry unchanged.
    (with-redefs-fn
      {#'gpu/rt-resolve
       (fn [_ name]
         (if (= name "register-kernel!")
           (fn [& _] nil)
           (throw (ex-info "compiler control unexpectedly requested native execution"
                           {:name name}))))}
      #(apply str (map :source (gpu/compile! session :identity-control operation
                                           {:dtype :double}))))))

(deftest own-namespaced-helper-does-not-acquire-sine-semantics
  (doseq [operation [#'sin-map! #'control-map!]]
    (let [a (double-array [0.5 -1.0 0.0 2.0])
          output (double-array 4)
          source (compiled-source operation)]
      (operation a output 4)
      (is (= [100.5 99.0 100.0 102.0] (vec output)))
      (is (re-find #"\+ 100\.0" source) "the user helper's actual arithmetic reaches emission")
      (is (not (re-find #"\bsin\(" source))
          "a qualified user function must not be reinterpreted as mathematical sine"))))

(deftest qualified-identity-is-not-a-basename-contract
  (doseq [identity '[user.helpers/sin user.helpers/sin_m_double-impl
                    user.helpers/sqrt user.helpers/sqrt_m_double-impl
                    user.helpers/unchecked-add user.helpers/unchecked-add_m_long_long-impl
                    user.helpers/_plus__m_double_double-impl clojure.core/sin
                    raster.numeric/sin raster.numeric/tanh Math/asinh Math/log2 raster.quant.kernels/wi8-dot
                    sin-impl sqrt-impl unchecked-add-impl]]
    (is (nil? (intrinsic/canonical identity)) (str identity))
    (is (nil? (intrinsic/source-overflow-policy identity)) (str identity)))
  (doseq [[identity expected] [['raster.math/sin :sin]
                              ['raster.math/sin_m_double-impl :sin]
                              ['raster.numeric/_plus__m_double_double-impl :+]
                              ['raster.numeric/sqrt :sqrt] ['Math/sin :sin]
                              ['java.lang.Math/sin :sin] ['raster.par/dp4a :dp4a]
                              ['clojure.core/unchecked-add :+] ['_plus__m_double_double :+]
                              ['sin :sin] [:sin :sin]]]
    (is (= expected (intrinsic/canonical identity)) (str identity)))
  (is (= :wrap (intrinsic/source-overflow-policy 'clojure.core/unchecked-add)))
  (is (= :wrap (intrinsic/source-overflow-policy 'clojure.core/unchecked-add_m_long_long-impl)))
  (is (some? (intrinsic/c-lowering 'raster.math/sin_m_double-impl false)))
  (is (nil? (intrinsic/c-lowering 'user.helpers/sin_m_double-impl false)))
  (is (nil? (intrinsic/c-lowering "user.helpers/sin_m_double-impl" false))))

(deftest explicit-intrinsic-contract-uses-existing-definition-revision
  (let [identity 'raster.compiler.scalar-intrinsic-identity-test/explicit-contract
        before (dispatch/compiler-definition-revision)]
    (operation/register-op-descriptor! identity {:intrinsic {:key :sin}})
    (is (< before (dispatch/compiler-definition-revision)))
    (is (= :sin (intrinsic/canonical identity)))
    (let [registered (dispatch/compiler-definition-revision)]
      (operation/register-op-descriptor! identity {:intrinsic {:key :sin}})
      (is (= registered (dispatch/compiler-definition-revision)))
      (binding [dispatch/*installing-derived-specialization* true]
        (operation/register-op-descriptor! identity {:intrinsic {:key :sqrt}}))
      (is (< registered (dispatch/compiler-definition-revision)))
      (is (= :sqrt (intrinsic/canonical identity))))
    (let [registered (dispatch/compiler-definition-revision)]
      (operation/register-op-descriptor! identity {:intrinsic nil})
      (is (< registered (dispatch/compiler-definition-revision)))
      (is (nil? (intrinsic/canonical identity))))))

(deftest exact-concrete-intrinsic-contract-precedes-derived-base
  (let [base 'raster.compiler.scalar-intrinsic-identity-test/exact-operation
        concrete 'raster.compiler.scalar-intrinsic-identity-test/exact-operation_m_double-impl]
    (operation/register-op-descriptor! base {:intrinsic {:key :sin}})
    (operation/register-op-descriptor! concrete {:intrinsic {:key :sqrt}})
    (is (= :sqrt (intrinsic/canonical concrete)))
    (operation/register-op-descriptor! concrete {:intrinsic nil})
    (is (nil? (intrinsic/canonical concrete)) "explicit withdrawal does not resurrect the base contract")
    (operation/register-op-descriptor! base {:intrinsic nil})))

(deftest withdrawing-inherited-contract-advances-revision
  (let [base (symbol "raster.compiler.scalar-intrinsic-identity-test" (str (gensym "inherited-operation")))
        concrete (symbol (namespace base) (str (name base) "_m_double-impl"))]
    (operation/register-op-descriptor! base {:intrinsic {:key :sin}})
    (is (= :sin (intrinsic/canonical concrete)))
    (let [before (dispatch/compiler-definition-revision)]
      (operation/register-op-descriptor! concrete {:intrinsic nil})
      (is (< before (dispatch/compiler-definition-revision)))
      (is (nil? (intrinsic/canonical concrete))))
    (operation/register-op-descriptor! base {:intrinsic nil})))

(deftest retained-c-helper-preserves-qualified-identity
  (doseq [semantic [nil 'raster.compiler.scalar-intrinsic-identity-test/sin]]
    (let [form (with-meta '(.invk raster.compiler.scalar-intrinsic-identity-test/sin_m_double-impl 0.5)
                 (cond-> {} semantic (assoc :raster.op/original semantic)))
          source (c/emit-expr form 'i #{} "get_global_id(0)")]
      (is (re-find #"gpufn_" source))
      (is (not (re-find #"^sin\(" source)))))
  (let [form (with-meta '(.invk raster.math/sin_m_double-impl 0.5)
               {:raster.op/original 'user.helpers/sin})
        source (c/emit-expr form 'i #{} "get_global_id(0)")]
    (is (re-find #"gpufn_" source) "rejected semantic metadata must not retry a builtin basename")))

(deftest session-acquisition-observes-semantic-revision
  (let [session (atom {:device-id :ocl:0 :closed? false :kernels {} :dispatches {} :kernel-cache {}})
        calls (atom 0)
        identity 'raster.compiler.scalar-intrinsic-identity-test/session-contract]
    (with-redefs-fn
      {#'gpu/compile-deftm-internal!
       (fn [& _] {:kernels [{:generation (swap! calls inc)}]})}
      #(let [old (gpu/compile! session :old #'sin-map! {:dtype :double})]
         (is (= old (gpu/compile! session :same #'sin-map! {:dtype :double})))
         (is (= 1 @calls))
         (operation/register-op-descriptor! identity {:intrinsic {:key :sin}})
         (is (not= old (gpu/compile! session :fresh #'sin-map! {:dtype :double})))
         (is (= 2 @calls))
         (is (= 1 (count (:kernel-cache @session))) "old acquisition epochs do not accumulate")
         (is (= old (get-in @session [:kernels :old])) "already-published phases retain their artifact")))
    (operation/register-op-descriptor! identity {:intrinsic nil})))

(deftest helper-collection-and-emission-share-call-identity
  (let [base 'raster.compiler.scalar-intrinsic-identity-test/identity-reference
        concrete (symbol (namespace base) (str (name base) "_m_double-impl"))
        descriptor operation/get-op-descriptor
        direct (list (symbol (namespace base) (str (name base) "_m_double")) 0.5)
        dispatched (list '.invk concrete 0.5)]
    (doseq [[label concrete-facet expected]
            [[:inherited {} "sin("]
             [:overridden {:intrinsic {:key :sqrt}} "sqrt("]
             [:withdrawn {:intrinsic nil} "gpufn_"]]]
      ;; Overlay only the semantic facets of a real deftm; no owner/cache/epoch reset.
      (with-redefs [operation/get-op-descriptor
                    (fn [identity]
                      (cond
                        (= identity base) (assoc (descriptor base) :intrinsic {:key :sin})
                        (= identity concrete) (merge (dissoc (descriptor concrete) :intrinsic)
                                                    concrete-facet)
                        :else (descriptor identity)))]
        (let [call (c/emit-expr dispatched 'i #{} "get_global_id(0)")
              helpers (c/collect-gpu-fn-calls dispatched)]
          (is (.startsWith call expected) (str label))
          (is (= (if (= label :withdrawn) 1 0) (count helpers)) (str label))
          (when (= label :withdrawn)
            (is (and (seq helpers) (.startsWith call (c/gpu-helper-c-name (:sym (first helpers))))))
            (is (and (seq helpers) (.contains (:source (c/generate-c-helper (first helpers))) "100.0")))))
        (doseq [form [direct dispatched]
                semantic [nil 'user.helpers/non-intrinsic]]
          (let [retained (with-meta form {:raster.op/original semantic})
                call (c/emit-expr retained 'i #{} "get_global_id(0)")
                helpers (c/collect-gpu-fn-calls retained)]
            (is (= 1 (count helpers)) (str label " explicit non-intrinsic " semantic))
            (is (and (seq helpers) (.startsWith call (c/gpu-helper-c-name (:sym (first helpers))))))))
        (let [intrinsic-call (with-meta dispatched {:raster.op/original base})
              helper-call (with-meta dispatched {:raster.op/original nil})]
          (doseq [forms [[intrinsic-call helper-call] [helper-call intrinsic-call]]]
            (is (= 1 (count (c/collect-gpu-fn-calls (cons 'do forms))))
                "intrinsic ownership does not mark a required helper as already collected")))))))

(deftest acquisition-does-not-relabel-an-unstable-compilation
  (let [session (atom {:device-id :ocl:0 :closed? false :kernels {} :dispatches {} :kernel-cache {}})
        calls (atom 0)]
    (with-redefs-fn
      {#'gpu/compile-deftm-internal!
       (fn [& _]
         (let [generation (swap! calls inc)]
           (when (= 1 generation) (dispatch/bump-compiler-definition-revision!))
           {:kernels [{:generation generation}]}))}
      #(do
         (is (= [{:generation 1}] (gpu/compile! session :unstable #'sin-map!)))
         (is (empty? (:kernel-cache @session)))
         (is (= [{:generation 2}] (gpu/compile! session :stable #'sin-map!)))
         (is (= [{:generation 2}] (gpu/compile! session :reuse #'sin-map!)))
         (is (= 2 @calls))))))
