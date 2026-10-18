(ns raster.compiler.passes.scalar.numeric-regions-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]
            [raster.compiler.passes.scalar.numeric-regions :as regions]))

(deftm region-square [x :- Double] :- Double (clojure.core/* x x))

(defn- project [source & [ports result]]
  (method-entry/call-with-definition-inputs
   #(regions/project-closed-double-region source (or ports {'x Double/TYPE}) (or result Double/TYPE))))

(defn- execute [source x]
  ((eval (list 'fn* [(with-meta 'x {:tag 'double})]
               (completion/project-to-jvm-source source))) x))

(deftest explicit-primitive-conversion-uses-the-shared-source-proof
  (is (= {:type :double}
         (scalar-body/checked-primitive-result '(clojure.core/double x) {'x :double})))
  (is (nil? (scalar-body/checked-primitive-result '(clojure.core/double x) {}))))

(deftest closed-object-fields-project-only-onto-an-owned-primitive-abi
  (let [source (list 'let* ['saved (completion/complete-value
                                  completion/object-input-contract '(clojure.core/* x 2.0))
                           'alias 'saved]
                     '(clojure.core/+ saved alias))
        result (project source)]
    (is (= :double (:type result)))
    (is (not (completion/contains-completion? (:form result))))
    (doseq [x [-3.5 -0.0 0.0 Math/PI Double/POSITIVE_INFINITY]]
      (is (= (Double/doubleToRawLongBits (execute source x))
             (Double/doubleToRawLongBits (execute (:form result) x)))))
    (is (nil? (regions/project-closed-double-region source {'x Double/TYPE} Double/TYPE)))
    (is (nil? (project source {'x Float/TYPE})))
    (is (nil? (project source {'x Double/TYPE} Float/TYPE)))))

(deftest escaping-and-effectful-uses-decline-the-whole-region
  (let [value (completion/complete-value completion/object-input-contract 'x)]
    (doseq [body ['(clojure.core/identical? saved saved)
                  '[saved]
                  '(clojure.core/identity saved)
                  '(clojure.core/aset array 0 saved)
                  '(unknown saved)
                  '(if (clojure.core/= saved saved) 1.0 0.0)
                  '(if (clojure.core/boolean saved) 1.0 0.0)
                  '(Math/sqrt saved)
                  '(loop* [i 0 acc saved] (if (< i 2) (recur (inc i) (+ acc saved)) acc))]]
      (let [source (list 'let* ['saved value] body)]
        (is (nil? (project source)) (str "unclosed use: " body)))))
  (let [source (list 'let* ['saved (completion/complete-value
                                  (completion/primitive-input-contract Double/TYPE {:unchecked-math false})
                                  'x)] 'saved)]
    (is (nil? (project source)) "primitive completion is not an Object alias"))
  (doseq [tag ['float 'Float 'long 'Object]]
    (let [saved (with-meta 'saved {:raster.type/tag tag})
          source (list 'let* [saved (completion/complete-value completion/object-input-contract 'x)] saved)]
      (is (nil? (project source)) "a consumer hint is not a conversion proof"))))

(deftest ordered-initializers-and-strict-floating-association-survive
  (let [source (list 'let* ['a (completion/complete-value completion/object-input-contract 'x)
                           'b '(clojure.core/+ a 1.0)
                           'c '(clojure.core/+ b -1.0e16)] 'c)
        result (project source)]
    (is (= ['a 'b 'c] (vec (take-nth 2 (second (:form result))))))
    (is (= 0.0 (execute (:form result) 1.0e16)))
    (is (= (execute source 1.0e16) (execute (:form result) 1.0e16)))))

(deftest optional-scalar-sum-and-its-proved-nil-branch-project-together
  (let [source (list 'let* ['saved (completion/complete-value completion/object-input-contract 'x)
                           'sum '(raster.ad.reverse/grad-acc saved saved)]
                     '(if (clojure.core/nil? sum) 0.0 sum))
        result (project source)]
    (is (= :double (:type result)))
    (is (not-any? #{'raster.ad.reverse/grad-acc 'clojure.core/nil?}
                  (tree-seq coll? seq (:form result))))
    (doseq [x [-3.5 -0.0 0.0 Math/PI]]
      (is (= (execute source x) (execute (:form result) x))))))

(deftest selected-producer-needs-its-published-actual-entrypoint
  (let [source-var (core/resolve-deftm-var #'region-square)
        receiver (ns-resolve (:ns (meta source-var)) (symbol (str (:name (meta source-var)) "-impl")))
        _ (core/ensure-bytecode-implementation! receiver)
        call (with-meta (list '.invk
                              (symbol (str (:ns (meta receiver))) (str (:name (meta receiver)))) 'x)
               {:raster.op/original 'raster.compiler.passes.scalar.numeric-regions-test/region-square})
        source (list 'let* ['saved (completion/complete-value completion/object-input-contract call)]
                     '(clojure.core/+ saved x))
        result (project source)]
    (is (= :double (:type result)))
    (doseq [x [-3.5 -0.0 0.0 Math/PI]]
      (is (= (+ (* x x) x) (execute (:form result) x))))
    (with-redefs-fn {receiver (fn [_] nil)}
      #(is (nil? (project source)) "replacement cannot inherit an old emitted carrier proof"))))

(deftest cold-source-projection-must-not-require-jvm-warmup
  ;; An isolated declaration prevents other tests' arithmetic calls from hiding
  ;; the cold compilation gap. The native proof must eventually use a retained
  ;; source-entry contract, not force JVM emission merely to manufacture proof.
  (doseq [jit? [true false]]
   (let [scope (create-ns (gensym "raster.cold-numeric-region."))]
    (try
      (binding [*ns* scope core/*compile-deftm-bytecode?* jit?]
        (refer 'clojure.core)
        (eval '(raster.core/deftm cold-square [x :- Double] :- Double (clojure.core/* x x))))
      (let [generic (ns-resolve scope 'cold-square)
            source-var (core/resolve-deftm-var generic)
            receiver (ns-resolve scope (symbol (str (:name (meta source-var)) "-impl")))
            implementation @receiver
            call (with-meta (list '.invk
                                  (symbol (str (ns-name scope)) (str (:name (meta receiver)))) 'x)
                   {:raster.op/original (symbol (str (ns-name scope)) "cold-square")})
            source (completion/complete-value completion/object-input-contract call)]
        (is (= jit? (some? (:raster.core/jit-spec (meta receiver)))))
        (is (identical? implementation (get-in (meta receiver) [:raster.core/source-entry :implementation])))
        (when jit?
          (is (identical? (:raster.core/source-entry (meta receiver))
                          (:raster.core/jit-spec (meta receiver)))
              "source ownership and lazy execution share one immutable spec"))
        (is (nil? (:raster.core/bytecode-emission (meta receiver))))
        (let [proof (method-entry/call-with-definition-inputs
                     #(scalar-body/checked-selected-source-double-result call))]
          (is (= :source-scalar (:kind proof)))
          (is (nil? (get-in proof [:owner :entrypoint])))
          (is (nil? (method-entry/call-with-definition-inputs
                     #(scalar-body/checked-selected-object-result call)))))
        (is (= :double (:type (project source)))
            "cold source is compilable without a JVM emission artifact")
        (let [inputs (atom {:static [] :values []})
              proof (binding [method-entry/*emission-inputs* inputs]
                      (scalar-body/checked-selected-source-double-result call))]
          (is (= :source-scalar (:kind proof)))
          (is (= 3 (count (:definition-proofs @inputs))))
          (doseq [changed [generic source-var receiver]]
            (with-redefs-fn {changed (fn [& _] :changed-after-query)}
              #(is (thrown? clojure.lang.ExceptionInfo
                            (method-entry/assert-definition-inputs-current! @inputs)))))
          (is (true? (method-entry/assert-definition-inputs-current! @inputs))))
        (doseq [changed [source-var receiver]]
          (with-redefs-fn {changed (fn [_] nil)}
            #(is (nil? (project source)) "a replaced definition cannot inherit source ownership")))
        (let [before (meta source-var)]
          (try
            (alter-meta! source-var assoc :raster.core/deftm-source-body [99.0])
            (is (nil? (project source)) "a source entry cannot certify a different retained body")
            (finally (reset-meta! source-var before))))
        (is (identical? implementation @receiver))
        (is (nil? (:raster.core/bytecode-emission (meta receiver))))
        (is (= jit? (some? (:raster.core/jit-spec (meta receiver))))))
      (finally (remove-ns (ns-name scope)))))))

(deftest generic-projection-requires-the-published-dispatcher
  (let [source (list 'let* ['saved (completion/complete-value completion/object-input-contract 'x)]
                     '(raster.numeric/+ saved saved))
        calls (atom 0)]
    (is (= :double (:type (project source))))
    (with-redefs [numeric/+ (fn [& _] (swap! calls inc) 0.0)]
      (is (nil? (project source))))
    (is (zero? @calls))
    (is (= :double (:type (project source))))))
