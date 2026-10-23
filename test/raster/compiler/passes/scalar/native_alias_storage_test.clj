(ns raster.compiler.passes.scalar.native-alias-storage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.dce :as dce]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.scalar.local-products :as products]))

(defn- snapshot [value]
  (completion/complete-value completion/object-input-contract value))

(defn- project [source]
  (products/project-storage-alias-snapshots source {} Double/TYPE))

(deftm nil-product-numeric-consumer [x :- Double] :- Double
  (let [values [nil x]] (double (nth values 0))))

(deftest native-nil-consumers-cannot-acquire-numeric-zero
  (is (thrown? NullPointerException (nil-product-numeric-consumer 1.0)))
  (is (= :unclosed-c-family-value-completion
         (try (pipeline/compile-aot #'nil-product-numeric-consumer :target :c)
              nil
              (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))

(deftest literal-nil-object-completion-has-no-storage-or-conversion
  (is (= (snapshot nil) (project (snapshot nil)))
      "a root or escaping nil is not a native numeric value")
  (let [source (list 'let* ['unused (snapshot nil)] 3.0)
        lowered (project source)]
    (is (not (completion/contains-completion? lowered)))
    (is (= '(let* [] 3.0) (:form (dce/eliminate-dead-bindings lowered)))))
  (doseq [source [(snapshot '(do (side-effect) nil))
                  (snapshot 1.0)
                  (snapshot 'unknown)
                  (list 'clojure.core/double (snapshot nil))
                  (list 'let* ['saved (snapshot nil)] '(clojure.core/double saved))
                  (list 'let* ['saved (snapshot nil)] 'saved)
                  (completion/complete-value
                   (completion/primitive-input-contract Double/TYPE {:unchecked-math false}) nil)
                  (vary-meta (snapshot nil) assoc :raster.effect/preserve-evaluation true)
                  (vary-meta (snapshot nil) assoc :raster.effect/source-reference-cast true)
                  (list completion/completion-symbol completion/object-input-contract)]]
    (is (= source (project source)))
    (is (completion/contains-completion? (project source))))
  (let [quoted (list 'quote (snapshot nil))]
    (is (identical? quoted (project quoted))))
  (let [source (list 'let* ['a (snapshot nil) 'b (snapshot 'a)] 3.0)
        lowered (project source)]
    (is (completion/contains-completion? lowered))
    (is (= 3.0 (eval (completion/project-to-jvm-source lowered))))
    (is (:raster.effect/preserve-evaluation (meta (second (second lowered))))
        "an alias can consume the exact nil producer's evaluation token")))

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

(deftest representation-erasure-preserves-exceptional-evaluation-through-dce
  (doseq [source ['(let* [a (clojure.core// x 0) p [a 7.0]]
                     (clojure.core/nth p 1))
                  '(let* [a (clojure.core/+ x 1) p [a 7.0]]
                     (clojure.core/nth p 1))
                  '(let* [a {:value (clojure.core// x 0)} p [a 7.0]]
                     (clojure.core/nth p 1))
                  '(let* [a #{(clojure.core// x 0)} p [a 7.0]]
                     (clojure.core/nth p 1))
                  '(let* [a [(clojure.core// x 0)] p [a 7.0]]
                     (clojure.core/nth p 1))
                  '(let* [a (clojure.core// x 0) b (do a 8.0) p [b 7.0]]
                     (clojure.core/nth p 1))
                  '(let* [a (clojure.core/double x) p [a 7.0]]
                     (clojure.core/nth p 1))]]
    (let [lowered (products/lower-local-products source)
          projected (project lowered)
          repeated (project (products/lower-local-products projected))
          optimized (:form (dce/eliminate-dead-bindings repeated))
          x (if (= 'clojure.core/+ (first (nth (second source) 1))) Long/MAX_VALUE nil)
          ;; Division receives a real number; the nullable conversion receives
          ;; nil. Each source must throw, independently of compiler inspection.
          x (if (and (nil? x) (not= '(clojure.core/double x) (nth (second source) 1))) 1 x)
          outcome (fn [form]
                    (try (invoke form x nil)
                         (catch Throwable e (class e))))
          expected (outcome source)]
      (is (instance? Class expected) (str source))
      (doseq [form [lowered projected repeated optimized]]
        (is (= expected (outcome form)) (str form))))))

(deftest evaluation-denials-apply-to-collection-roots
  (doseq [value [[1.0] {:value 1.0} #{1.0}]]
    (let [marked (vary-meta value assoc :raster.effect/preserve-evaluation true)]
      (is (contains? (:flags (effects/descriptor marked)) :preserve-source-evaluation))
      (is (false? (effects/removable-expr? marked)))
      (is (false? (effects/cse-safe-expr? marked))))))

(deftest transitive-obligations-are-owned-by-the-visible-lexical-producer
  (let [source '(let* [inc (clojure.core// x 0)
                      b (do inc 8.0)
                      p [b 7.0]]
                 (clojure.core/nth p 1))
        projected (project (products/lower-local-products source))
        bindings (into {} (map vec) (partition 2 (second projected)))]
    (is (true? (:raster.effect/preserve-evaluation (meta (get bindings 'inc)))))
    (is (true? (:raster.effect/preserve-evaluation (meta (get bindings 'b)))))
    ;; Independent removal of a discarded read must not erase the earlier
    ;; throwing producer. Its own obligation, not a live use in b, retains it.
    (let [without-read (util/postwalk-preserving-meta
                        #(if (= % '(do inc 8.0)) (with-meta '(do 8.0) (meta %)) %) projected)
          optimized (:form (dce/eliminate-dead-bindings without-read))]
      (is (thrown? ArithmeticException (invoke optimized 1 nil))))))

(deftest erased-snapshots-preserve-first-failure-and-source-order
  (let [source '(let* [a (do (swap! calls conj :first) (clojure.core// x 0))
                      b (do (swap! calls conj :second) (clojure.core/double nil))
                      p [a b 7.0]]
                 (clojure.core/nth p 2))
        optimized (:form (dce/eliminate-dead-bindings
                          (project (products/lower-local-products source))))]
    (doseq [form [source optimized]]
      (let [calls (atom [])]
        (is (thrown? ArithmeticException (invoke form 1 calls)))
        (is (= [:first] @calls))))))

(deftest nonlexical-static-field-producers-are-not-total-lexical-aliases
  ;; System/out is a real, nonconstant GETSTATIC, not a lexical reference.
  ;; Erasing its storage does not authorize erasing its evaluation (including
  ;; initialization for an as-yet uninitialized declaring class).
  (let [source '(let* [a java.lang.System/out p [a 7.0]] (clojure.core/nth p 1))
        projected (project (products/lower-local-products source))
        optimized (:form (dce/eliminate-dead-bindings projected))
        bindings (into {} (map vec) (partition 2 (second optimized)))]
    (is (contains? bindings 'a))
    (is (= 'java.lang.System/out (get bindings 'a)))
    (is (true? (:raster.effect/preserve-evaluation (meta (get bindings 'a)))))
    (is (= 7.0 (invoke source nil nil) (invoke optimized nil nil)))))

(deftest evaluation-dependency-bookkeeping-is-direct-and-invocation-owned
  (let [graph (atom {}) consumed (atom #{})]
    (with-bindings {#'products/*evaluation-dependencies* graph
                   #'products/*consumed-evaluation-obligations* consumed}
      (let [scope (reduce (fn [scope i]
                            (let [id (symbol (str "value_" i))
                                  previous (symbol (str "value_" (dec i)))
                                  [known _] (#'products/owned-binding-fact
                                             scope id (list 'identity previous))]
                              (assoc scope id known)))
                          {} (range 128))
            [_ alias-token] (#'products/owned-binding-fact scope 'alias 'value_127)
            [diamond _] (#'products/owned-binding-fact scope 'diamond
                         '[value_126 value_127])]
        (is (nil? alias-token))
        (is (every? #(= 1 (count (:evaluation-obligations %))) (vals scope)))
        (is (= 129 (count @graph)))
        (is (= 129 (reduce + (map count (vals @graph)))))
        (#'products/consume-evaluation! diamond)
        (is (= 129 (count @consumed)))
        (#'products/consume-evaluation! diamond)
        (is (= 129 (count @consumed)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no producer"
              (#'products/consume-evaluation! {:evaluation-obligations #{(Object.)}})))))))

(deftest array-snapshots-preserve-reference-identity-and-nullability
  (let [source (list 'let* ['a 'x 'saved (snapshot 'a) 'copy 'saved]
                     '(clojure.core/identical? x copy))
        projected (products/project-storage-alias-snapshots
                   source {'x {:presence :unknown :array-tag 'doubles}} nil)]
    (is (not (completion/contains-completion? projected)))
    (is (= 'doubles (:tag (meta (nth (second projected) 2)))))
    (doseq [x [nil (double-array 0) (double-array [1.0 2.0])]]
      (is (true? (invoke source x nil)))
      (is (true? (invoke projected x nil)))))
  (let [source (list 'let* ['saved (snapshot 'x)]
                     '(clojure.core/aget saved 0))
        projected (products/project-storage-alias-snapshots
                   source {'x {:presence :unknown :array-tag 'doubles}} nil)]
    (doseq [form [source projected]]
      (is (thrown? NullPointerException (invoke form nil nil)))
      (is (thrown? ArrayIndexOutOfBoundsException (invoke form (double-array 0) nil))))))

(deftest array-storage-evidence-is-not-a-consumer-hint-or-a-nonnull-claim
  (let [hinted (with-meta 'x {:tag 'doubles :raster.type/tag 'doubles})
        source (list 'let* ['saved (snapshot hinted)] 'saved)]
    (is (completion/contains-completion?
         (products/project-storage-alias-snapshots source {} nil))))
  (let [source (list 'let* ['saved (snapshot 'x)] '(clojure.core/nil? saved))
        projected (products/project-storage-alias-snapshots
                   source {'x {:presence :unknown :array-tag 'doubles}} nil)]
    (is (true? (invoke projected nil nil)))
    (is (false? (invoke projected (double-array 0) nil)))))

(deftest conflicting-array-binders-do-not-get-silently-retyped
  (doseq [saved [(with-meta 'saved {:tag 'floats})
                 (with-meta 'saved {:tag 'Object :raster.type/tag 'floats})]]
    (let [source (list 'let* [saved (snapshot 'x)] saved)]
      (is (= :storage-alias-carrier-conflict
             (try (products/project-storage-alias-snapshots
                   source {'x {:presence :unknown :array-tag 'doubles}} nil)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest array-producer-effects-remain-at-their-source-position
  (let [source (list 'let* ['a '(do (swap! calls conj :allocate) (clojure.core/double-array x))
                            'saved (snapshot 'a)]
                     '(do (swap! calls conj :observe) (clojure.core/identical? a saved)))
        projected (project source)]
    (is (not (completion/contains-completion? projected)))
    (doseq [form [source projected]]
      (let [calls (atom [])]
        (is (true? (invoke form 2 calls)))
        (is (= [:allocate :observe] @calls)))
      (let [calls (atom [])]
        (is (thrown? NegativeArraySizeException (invoke form -1 calls)))
        (is (= [:allocate] @calls))))))
