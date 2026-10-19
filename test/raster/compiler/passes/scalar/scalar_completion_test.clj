(ns raster.compiler.passes.scalar.scalar-completion-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.numeric]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.normal-return :as normal]))

(use-fixtures :each
  (fn [test!]
    (binding [entry/*emission-inputs* (atom {:static [] :values []})]
      (test!))))

(defn- selected [op implementation & args]
  (with-meta (list* '.invk implementation args) {:raster.op/original op}))

(deftest facts-describe-successful-evaluation-not-its-safety
  (let [cast '(clojure.core/double nullable)
        fact (normal/completed-scalar-fact cast)]
    (is (= {:carrier Double/TYPE :presence :present :builtin-class Double} fact))
    (is (not (contains? fact :nonthrowing?)))
    (is (thrown? NullPointerException (eval (list 'let* ['nullable nil] cast))))
    (is (= {:carrier Object :presence :present :builtin-class Double}
           (normal/completed-scalar-fact
            (completion/complete-value completion/object-input-contract cast))))
    (is (= {:presence :unknown} (normal/result-fact 'unknown)))
    (is (nil? (normal/completed-scalar-fact (with-meta 'unknown {:tag 'double}))))
    (is (nil? (normal/completed-scalar-fact '(double nullable)))
        "bare casts need original source namespace authority")
    (is (nil? (normal/completed-scalar-fact cast {'clojure.core/double nil}))
        "a lexical call cannot borrow core cast semantics")))

(deftest primitive-literals-do-not-borrow-abstract-numeric-tags
  (doseq [[value carrier boxed] [[1.0 Double/TYPE Double]
                                 [(float 1) Float/TYPE Float]
                                 [1 Long/TYPE Long]]]
    (is (= {:carrier carrier :presence :present :builtin-class boxed}
           (normal/completed-scalar-fact value))))
  (doseq [value [1N (biginteger 1) 1/3 1M nil 'unknown]]
    (is (nil? (normal/completed-scalar-fact value))))
  (is (nil? (normal/completed-scalar-fact '(let [x 1.0] x)))
      "an open source macro spelling is not an expanded lexical region")
  (is (= {'x nil 'y nil}
         (normal/completed-scalar-bindings [['x 1.0] ['x 'unknown] ['y 'x]]))
      "an unknown sequential producer shadows an earlier known carrier"))

(deftest sequential-selected-calls-retain-their-object-carrier
  (let [multiply (selected 'raster.numeric/* 'raster.numeric/_star__m_double_double-impl
                           2.0 'seed)
        divide (selected 'raster.numeric// 'raster.numeric/_div__m_double_double-impl
                         'scaled '(clojure.core/double count))
        source (list 'let* ['seed '(clojure.core/double 1.0)
                            'scaled multiply 'result divide] 'result)
        owner (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* owner]
      (is (= {:carrier Object :presence :present :builtin-class Double}
             (normal/completed-scalar-fact source)))
      (is (seq (:definition-proofs @owner)))
      (entry/assert-definition-inputs-current! @owner))
    (binding [entry/*emission-inputs* nil]
      (is (nil? (normal/completed-scalar-fact source))))
    (binding [entry/*emission-inputs* (atom {:static [] :values []})]
      (is (nil? (normal/completed-scalar-fact divide {'scaled {:tag 'double}})))
      (is (nil? (normal/completed-scalar-fact
                 source {'raster.numeric/_star__m_double_double-impl nil}))))))

(deftest failed-outer-proof-retains-no-speculative-helper-dependencies
  (let [inner (selected 'raster.numeric/* 'raster.numeric/_star__m_double_double-impl 2.0 3.0)
        outer (selected 'raster.numeric// 'raster.numeric/_div__m_double_double-impl inner 'unknown)
        owner (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* owner]
      (is (nil? (normal/completed-scalar-fact outer)))
      (is (= {:static [] :values []} @owner)))))

(deftest input-context-is-not-a-physical-result-carrier
  (let [policy {:unchecked-math false}
        input (completion/complete-value
               (completion/primitive-input-contract Double/TYPE policy) 1.0)
        projected (completion/project-to-jvm-source input policy)
        box (.getMethod clojure.lang.RT "box" (into-array Class [Double/TYPE]))]
    (is (= Number (.getReturnType box)))
    (is (false? (.isPrimitive (.getReturnType box))))
    (is (instance? Double (eval projected)))
    (is (nil? (normal/completed-scalar-fact input)))
    (is (nil? (normal/completed-scalar-fact
               (completion/complete-value
                (completion/bytecode-floating-input-contract
                 Float/TYPE (completion/bytecode-operand-context {})) 1.0))))))

(deftest cast-proof-requires-bootstrap-definition-and-owner
  (let [expression '(clojure.core/double nullable)]
    (binding [entry/*emission-inputs* nil]
      (is (nil? (normal/completed-scalar-fact expression))))
    (let [original (meta #'clojure.core/double)]
      (try
        (alter-meta! #'clojure.core/double assoc :inline (fn [& _] nil))
        (is (= :intrinsic-source-definition-mismatch
               (try (normal/completed-scalar-fact expression) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
        (finally (reset-meta! #'clojure.core/double original))))))
