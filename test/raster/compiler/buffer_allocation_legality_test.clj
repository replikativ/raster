(ns raster.compiler.buffer-allocation-legality-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.scalar.buffer-fuse :as buffer-fuse]))

(defn- with-source [f]
  (let [source (create-ns (gensym "raster.test.buffer_legality_"))]
    (try
      (binding [*ns* source]
        (refer 'clojure.core)
        (eval '(def calls (atom 0)))
        (eval '(def failure (ex-info "size failure" {:test :size})))
        (eval '(defn ^{:tag 'long} tick [n] (swap! calls inc) n))
        (eval '(defn ^{:tag 'long} fail [n] (swap! calls inc) (throw failure)))
        (f source))
      (finally (remove-ns (ns-name source))))))

(defn- operation [source name tags]
  (symbol (str (ns-name source)) (str name "_m_" tags)))

(defn- define-allocator [name params body]
  (eval (list 'raster.core/deftm name params ':- '(Array double) body)))

(defn- call-form [op arguments]
  (list 'let* ['result (apply list op arguments)] 'result))

(deftest effectful-and-throwing-size-prefixes-are-not-replayed
  (with-source
    (fn [source]
      (doseq [helper '[tick fail]]
        (define-allocator 'allocate '[n :- Long]
          (list 'let ['m (list helper 'n) 'out '(double-array m)] 'out))
        (let [op (operation source "allocate" "long")
              original (call-form op [3])
              fused (:form (buffer-fuse/fuse-let original))
              calls @(ns-resolve source 'calls)
              failure @(ns-resolve source 'failure)]
          (is (nil? (descriptor/resolve-buffer-semantics op)))
          (is (= original fused))
          (doseq [form [original fused]]
            (reset! calls 0)
            (if (= helper 'fail)
              (is (identical? failure (try (eval form) nil (catch clojure.lang.ExceptionInfo e e))))
              (is (= 3 (alength ^doubles (eval form)))))
            (is (= 1 @calls))))))))

(deftest unrelated-preallocation-effects-and-open-helpers-retain-order
  (with-source
    (fn [source]
      (define-allocator 'allocate '[n :- Long]
        '(let [_ (tick n) out (double-array n)] out))
      (let [op (operation source "allocate" "long")
            original (call-form op [3])]
        (is (nil? (descriptor/resolve-buffer-semantics op)))
        (is (= original (:form (buffer-fuse/fuse-let original)))))
      (eval '(defn ^{:tag 'long} helper [n] n))
      (define-allocator 'open '[n :- Long] '(let [m (helper n) out (double-array m)] out))
      (let [op (operation source "open" "long")]
        (is (nil? (descriptor/resolve-buffer-semantics op)) "current purity is not a dependency certificate")
        (eval '(defn ^{:tag 'long} helper [n] (swap! calls inc) n))
        (is (nil? (descriptor/resolve-buffer-semantics op)))
        (let [form (:form (buffer-fuse/fuse-let (call-form op [3])))]
          (eval form)
          (is (= 1 @@(ns-resolve source 'calls))))))))

(deftest actual-argument-effects-are-neither-duplicated-nor-dropped
  (with-source
    (fn [source]
      (doseq [[name params body tags arguments]
              [['unused '[n :- Long extra :- Long] '(double-array n) "long_long" '[3 (tick 1)]]
               ['repeated '[n :- Long] '(let [out (double-array n)] (aset out 0 (double n)) out)
                "long" '[(tick 3)]]]]
        (define-allocator name params body)
        (let [op (operation source (str name) tags)
              original (call-form op arguments)
              fused (:form (buffer-fuse/fuse-let original))
              calls @(ns-resolve source 'calls)]
          (is (some? (descriptor/resolve-buffer-semantics op)) "safe body still has an auto contract")
          (is (= original fused) "complex actuals need evaluate-once bindings")
          (doseq [form [original fused]]
            (reset! calls 0)
            (is (= 3 (alength ^doubles (eval form))))
            (is (= 1 @calls))))))))

(deftest constructor-identity-and-overload-are-not-guessed
  (with-source
    (fn [source]
      (eval '(defn bogus-array [n] (double-array n)))
      (doseq [[name params body tags]
              [['impostor '[n :- Long] '(bogus-array n) "long"]
               ['initializer '[n :- Long] '(double-array n 4.0) "long"]
               ['copy-input '[a :- (Array double)] '(double-array a) "doubles"]]]
        (define-allocator name params body)
        (is (nil? (descriptor/resolve-buffer-semantics (operation source (str name) tags))))))))

(deftest namespaced-core-constructor-name-is-not-a-core-constructor
  (with-source
    (fn [source]
      (let [foreign (create-ns (gensym "raster.test.foreign_allocator_"))]
        (try
          (intern foreign (with-meta 'double-array {:tag 'doubles})
                  (fn [n] (double-array n)))
          (define-allocator 'foreign '[n :- Long]
            (list (symbol (str (ns-name foreign)) "double-array") 'n))
          (is (nil? (descriptor/resolve-buffer-semantics (operation source "foreign" "long"))))
          (finally (remove-ns (ns-name foreign))))))))

(deftest stable-size-and-harmless-prefix-still-fuse
  (with-source
    (fn [source]
      (define-allocator 'stable '[n :- Long] '(let [m n out (double-array m)] out))
      (let [op (operation source "stable" "long")
            original (call-form op [3])
            result (buffer-fuse/fuse-let original)]
        (is (some? (descriptor/resolve-buffer-semantics op)))
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (is (= (vec (eval original)) (vec (eval (:form result)))))))))

(deftest lexical-substitution-preserves-shadowing-capture-and-quoted-data
  (with-source
    (fn [source]
      (doseq [[name body expected]
              [['shadow '(let [out (double-array n)]
                           (let [n 2] (aset out 0 (double n))) out) 2.0]
               ['capture '(let [out (double-array n)]
                            (let [m 2] (aset out 0 (double n))) out) 3.0]
               ['quoted '(let [out (double-array n)]
                           (aset out 0 (if (= 'n (symbol "n")) 1.0 2.0)) out) 1.0]]]
        (define-allocator name '[n :- Long] body)
        (let [op (operation source (str name) "long")
              original (list 'let* ['m 3 'result (list op 'm)] 'result)
              result (buffer-fuse/fuse-let original)]
          (is (= 1 (get-in result [:stats :fresh-allocs])))
          (is (= expected (aget ^doubles (eval (:form result)) 0)))
          (is (= (vec (eval original)) (vec (eval (:form result))))))))))

(deftest global-actuals-are-captured-by-the-original-call
  (with-source
    (fn [source]
      (eval '(def global-size 3))
      (eval '(defn change-global [] (alter-var-root #'global-size (constantly 7))))
      (define-allocator 'global '[n :- Long]
        '(let [out (double-array n)] (change-global) (aset out 0 (double n)) out))
      (let [original (call-form (operation source "global" "long") '[global-size])
            result (buffer-fuse/fuse-let original)]
        (is (= original (:form result)))
        (doseq [form [original (:form result)]]
          (alter-var-root (ns-resolve source 'global-size) (constantly 3))
          (is (= [3.0 0.0 0.0] (vec (eval form)))))))))

(deftest caller-cast-names-cannot-capture-helper-operations
  (with-source
    (fn [source]
      (define-allocator 'casted '[n :- Long]
        '(let [m (long n) out (double-array m)] (aset out 0 (double n)) out))
      (let [original (list 'let* ['long '(fn [x] (tick x))
                                 'double '(fn [x] (tick x))
                                 'result (list (operation source "casted" "long") 3)] 'result)
            result (buffer-fuse/fuse-let original)
            calls @(ns-resolve source 'calls)]
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (doseq [form [original (:form result)]]
          (reset! calls 0)
          (is (= [3.0 0.0 0.0] (vec (eval form))))
          (is (zero? @calls)))))))

(deftest replay-admission-requires-local-and-unshadowed-operation-evidence
  (is (not (effects/replay-safe-value? 'global-value)))
  (is (effects/replay-safe-value? 'local-value {'local-value nil}))
  (is (not (effects/replay-safe-value? '(long n) {'long nil 'n 'long})))
  (is (effects/replay-safe-value? '(long n) {'n 'long})))
