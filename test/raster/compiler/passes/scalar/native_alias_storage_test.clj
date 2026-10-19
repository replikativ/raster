(ns raster.compiler.passes.scalar.native-alias-storage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.local-products :as products]))

(defn- snapshot [value]
  (completion/complete-value completion/object-input-contract value))

(defn- project [source]
  (products/project-native-alias-snapshots source {} Double/TYPE))

(defn- invoke [source x calls]
  ;; This pass targets C declarations, not Clojure primitive-local hint syntax.
  ;; Let Clojure infer its own carriers for the independent numerical oracle.
  (let [source (util/postwalk-preserving-meta
                #(if (symbol? %) (vary-meta % dissoc :tag) %) source)]
    ((eval (list 'fn* '[x calls] (completion/project-to-jvm-source source))) x calls)))

(deftest already-completed-aliases-can-use-primitive-consumers
  (let [source (list 'let* ['value '(clojure.core/double x)
                            'saved (snapshot 'value) 'copy 'saved]
                     '(clojure.core/+ (clojure.core/double saved) (clojure.core/double copy)))
        lowered (project source)]
    (is (not (completion/contains-completion? lowered)))
    (is (= 'double (:raster.type/tag (meta (nth (second lowered) 2)))))
    (is (= 'double (:raster.type/tag (meta (nth (second lowered) 4)))))
    (doseq [x [-0.0 0.0 1.5 Math/PI Double/POSITIVE_INFINITY Double/NaN]]
      (is (= (Double/doubleToLongBits (invoke source x nil))
             (Double/doubleToLongBits (invoke lowered x nil)))))))

(deftest direct-producers-and-unknown-physical-payloads-stay-object-values
  (doseq [source [(list 'let* ['saved (snapshot '(clojure.core/double x))]
                        '(clojure.core/double saved))
                  (list 'let* ['saved (snapshot (with-meta 'x {:raster.type/tag 'double}))]
                        '(clojure.core/double saved))
                  (list 'let* ['value nil 'saved (snapshot 'value)]
                        '(clojure.core/double saved))]]
    (is (completion/contains-completion? (project source)))))

(deftest object-consumers-cannot-borrow-a-payload-carrier
  (doseq [body ['(clojure.core/identical? saved saved)
                '(identity saved)
                '(clojure.core// saved saved)
                '(if saved 1.0 2.0)
                '(clojure.core/float saved)
                '(let* [f (fn* [] (clojure.core/double saved))] (f))]]
    (let [source (list 'let* ['value '(clojure.core/double x) 'saved (snapshot 'value)] body)]
      (is (completion/contains-completion? (project source)))))
  (let [source (list 'let* ['value '(clojure.core/double x) 'saved (snapshot 'value)]
                     '(clojure.core// saved saved))]
    (is (thrown? ArithmeticException (invoke source 0.0 nil)))
    (is (thrown? ArithmeticException (invoke (project source) 0.0 nil)))))

(deftest storage-erasure-preserves-producer-position-and-failure
  (let [source (list 'let* ['value '(clojure.core/double (do (swap! calls conj :producer) x))
                            'saved (snapshot 'value)]
                     '(do (swap! calls conj :consumer) (clojure.core/double saved)))
        lowered (project source)
        failure (fn [form calls]
                  (try (invoke form nil calls) nil (catch Throwable e (class e))))]
    (is (not (completion/contains-completion? lowered)))
    (let [a (atom []) b (atom [])]
      (is (= 3.0 (invoke source 3.0 a) (invoke lowered 3.0 b)))
      (is (= [:producer :consumer] @a @b)))
    (let [a (atom []) b (atom [])]
      (is (= NullPointerException (failure source a) (failure lowered b)))
      (is (= [:producer] @a @b)))))

(deftest alias-closure-respects-shadowing-and-rejects-nullable-joins
  (let [source (list 'let* ['value '(clojure.core/double x) 'saved (snapshot 'value)]
                     '(let* [saved nil] (identity saved)))]
    (is (not (completion/contains-completion? (project source))))
    (is (nil? (invoke (project source) 1.0 nil))))
  (let [source (list 'let* ['value '(clojure.core/double x) 'saved (snapshot 'value)
                            'joined '(if (pos? x) saved nil)]
                     '(clojure.core/double joined))]
    (is (completion/contains-completion? (project source)))))

(deftest core-named-locals-still-count-as-reference-observations
  (doseq [body ['(if inc 1.0 2.0)
                '(fn* [] (clojure.core/identical? inc inc))]]
    (let [source (list 'let* ['value '(clojure.core/double x) 'inc (snapshot 'value)] body)]
      (is (completion/contains-completion? (project source))))))

(deftest completion-rewalk-retains-operand-context-and-separate-types
  (let [operand '(clojure.core// x y)
        source (snapshot operand)
        walked (walker/walk-body source {:type-env {'x {:tag 'double} 'y {:tag 'double}}})]
    (is (identical? operand (:operand (completion/canonical-parts walked))))
    (is (= 'double (inference/infer-arg-tag walked {'x 'double 'y 'double}))))
  (let [source (list 'let* ['saved (snapshot 'x) 'copy 'saved] 'copy)
        tagged (#'pipeline/tag-binding-types source {'x 'double})
        again (#'pipeline/tag-binding-types tagged {'x 'double})]
    (doseq [binding [(first (second tagged)) (nth (second tagged) 2)]]
      (is (= 'Object (:tag (meta binding))))
      (is (= 'double (:raster.type/tag (meta binding)))))
    (is (= (map meta (take-nth 2 (second tagged)))
           (map meta (take-nth 2 (second again)))))))

(deftest malformed-completion-terms-never-reenter-source-dispatch
  (doseq [source [(list completion/completion-symbol)
                  (list completion/completion-symbol {:invalid true} 'x)
                  (list completion/completion-symbol completion/object-input-contract 'x 'extra)]]
    (is (= :unsupported-value-completion
           (try (walker/walk-body source {})
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(deftest flattened-initializers-retain-object-storage-through-aliases
  (let [source (list 'let* ['result (list 'let* ['saved (snapshot 'value)] 'saved)
                            'copy 'result]
                     '(clojure.core/identical? result copy))
        walked (walker/walk-body source {:type-env {'value {:tag 'double}}})]
    (doseq [id (take-nth 2 (second walked))]
      (is (= 'Object (:tag (meta id))))
      (is (= 'double (:raster.type/tag (meta id)))))))

(deftest exhausted-storage-analysis-never-publishes-a-partial-rewrite
  (let [source (list 'let* ['value '(clojure.core/double x) 'saved (snapshot 'value)]
                     '(clojure.core/double saved))]
    (with-redefs-fn {#'products/native-snapshot-visit-limit 1}
      #(is (= :native-object-storage-budget
              (try (project source)
                   (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))

(deftest transitive-aliases-cannot-demand-an-incompatible-payload-width
  (let [copy (with-meta 'copy {:tag 'Object :raster.type/tag 'float})
        source (list 'let* ['value '(clojure.core/double x) 'saved (snapshot 'value) copy 'saved]
                     '(clojure.core/double copy))]
    (is (completion/contains-completion? (project source)))
    (is (= 16777217.0 (invoke source 16777217.0 nil) (invoke (project source) 16777217.0 nil)))))

(deftest unused-bound-alias-snapshots-do-not-erase-their-producers
  (let [source (list 'let* ['value '(do (swap! calls inc) x) 'unused (snapshot 'value)] 7.0)
        lowered (project source)
        calls (atom 0)]
    (is (not (completion/contains-completion? lowered)))
    (is (= 7.0 (invoke lowered nil calls)))
    (is (= 1 @calls))))
