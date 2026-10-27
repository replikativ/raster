(ns raster.compiler.core.source-declaration-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.core.source-declaration :as declaration]
            [raster.compiler.core.macroexpand :as mex]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.core.util :as util]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.cse :as cse]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.compiler.passes.scalar.rewalk :as rewalk]
            [raster.gpu.core :as gpu]))

(deftm incompatible-clone!
  [input :- (Array double), out :- (Array double)] :- (Array double)
  (let [^objects cloned (clojure.core/aclone input)]
    (par/map! out i 1 double (aget cloned i))))

(deftm compatible-clone!
  [input :- (Array double), out :- (Array double)] :- (Array double)
  (let [^doubles cloned (clojure.core/aclone input)]
    (par/map! out i 1 double (aget cloned i))))

(deftm incompatible-constructor!
  [out :- (Array double)] :- (Array double)
  (let [^objects data (double-array 1)]
    (par/map! out i 1 double (aget data i))))

(deftm forged-declaration!
  [out :- (Array double)] :- (Array double)
  (let [^{:tag objects :raster.source/binding-declaration {:present? false}} data (double-array 1)]
    (par/map! out i 1 double (aget data i))))

(deftm incompatible-element [input :- (Array double)] :- Double
  (let [^objects data (clojure.core/aclone input)] (aget data 0)))

(defmacro macro-declared-element [input]
  `(let [~(with-meta 'data {:tag 'objects}) (clojure.core/aclone ~input)]
     (clojure.core/aget ~'data 0)))

(defmacro macro-splice [body] `(do ~body))

(defmacro nested-macro-declared-element [input] `(macro-declared-element ~input))

(deftm incompatible-macro-helper!
  [input :- (Array double), out :- (Array double)] :- (Array double)
  (par/map! out i 1 double (macro-declared-element input)))

(deftm incompatible-inlined-helper!
  [input :- (Array double), out :- (Array double)] :- (Array double)
  (let [value (incompatible-element input)]
    (par/map! out i 1 double value)))

(deftm declared-output! [out :- (Array double)] :- (Array double)
  (let [^objects target out]
    (par/map! target i 1 double 3.0)))

(deftm incompatible-store! [input :- (Array double)] :- (Array double)
  (let [^objects target (clojure.core/aclone input)]
    (par/map-void! i 1 (clojure.core/aset target i 3.0))
    target))

(deftm mixed-shadow!
  [wide :- (Array double), narrow :- (Array float), out :- (Array double)] :- (Array double)
  (let [^doubles value (clojure.core/aclone wide)]
    (par/map! out i 1 double (aget value i))
    (let [^floats value (clojure.core/aclone narrow)]
      (par/map! out i 1 double (+ (aget out i) (double (aget value i)))))))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest original-capture-is-scoped-and-cannot-trust-a-forged-absence
  (let [outer (with-meta 'value {:tag 'objects :raster.source/binding-declaration {:present? false}})
        inner (with-meta 'value {:tag 'doubles})
        source (list 'let* [outer 'input] (list 'let* [inner 'input] 'value))
        captured (declaration/capture-form source)
        outer' (first (second captured))
        inner' (first (second (last captured)))]
    (is (= {:present? true :hint 'objects} (declaration/declaration outer')))
    (is (= {:present? true :hint 'doubles} (declaration/declaration inner')))
    (is (= 'objects (:tag (meta (declaration/retain-hint
                                 (vary-meta outer' assoc :tag 'doubles :raster.type/tag 'doubles))))))
    (is (= 'doubles (:tag (meta inner'))))
    (is (= '(quote (let* [value input] value))
           (declaration/capture-form '(quote (let* [value input] value)))))
    (let [renamed (util/alpha-convert captured)]
      (is (= {:present? true :hint 'objects}
             (declaration/declaration (first (second renamed))))))))

(deftest rewalking-does-not-turn-inference-into-a-declaration
  (let [source (declaration/capture-form '(let* [value (clojure.core/aclone input)] value))
        options {:simplify? true :dtype :double :source-ns 'raster.compiler.core.source-declaration-test
                 :param-env {'input 'doubles}}
        once (:form (rewalk/pe-rewalk source options))
        twice (:form (rewalk/pe-rewalk once options))]
    (doseq [walked [once twice]]
      (is (= {:present? false :hint nil} (declaration/declaration (first (second walked)))))
      (is (= 'doubles (:raster.type/tag (meta (first (second walked))))))
      (is (= walked (declaration/validate-native-array-declarations! walked))))))

(deftest expansion-captures-new-declarations-without-reclassifying-caller-inference
  (let [caller (vary-meta 'caller assoc :tag 'doubles :raster.type/tag 'doubles)
        incoming (list 'let* [caller 'input] 'caller)
        fresh (with-meta 'fresh {:tag 'objects :raster.source/binding-declaration {:present? false}})
        expanded (list 'let* [fresh '(clojure.core/aclone input)] incoming)
        captured (declaration/capture-expansion incoming expanded)]
    (is (= {:present? true :hint 'objects} (declaration/declaration (first (second captured)))))
    (is (nil? (declaration/declaration (first (second (last captured)))))
        "already inferred caller tags are not new source declarations"))
  (binding [*ns* (the-ns 'raster.compiler.core.source-declaration-test)]
    (doseq [source ['(macro-declared-element input) '(nested-macro-declared-element input)]]
      (let [expanded (mex/macroexpand-core source)]
        (is (= {:present? true :hint 'objects}
               (declaration/declaration (first (second expanded)))))))))

(deftest owned-completions-remain-opaque-through-parent-macro-expansion
  (let [operand '(let* [value input] value)
        completed (completion/complete-value completion/object-input-contract operand)]
    (is (identical? completed (declaration/capture-form completed)))
    (is (identical? completed (last (declaration/capture-expansion
                                    (list 'macro-splice completed) (list 'do completed)))))
    (binding [*ns* (the-ns 'raster.compiler.core.source-declaration-test)]
      (is (identical? completed (last (mex/macroexpand-core (list 'macro-splice completed))))))))

(deftest compatible-declarations-keep-the-vector-schedule
  (doseq [hint [nil 'doubles]]
    (let [binder (cond-> 'data hint (vary-meta assoc :tag hint))
          captured (declaration/capture-form
                    (list 'let* [binder 'input]
                          '(raster.par/map! out i 100 double (+ (aget data i) 1.0))))
          selected (vary-meta (first (second captured)) assoc :raster.type/tag 'doubles)
          source (list 'let* [selected 'input] (last captured))
          {:keys [stats]} (par-simd/simd-pass source :min-elements 0 :dtype :double)]
      (is (= 1 (:simd-maps stats)))
      (is (zero? (get stats :fallback 0))))))

(deftest native-constraints-refuse-conflicts-and-unresolved-declarations
  (doseq [[hint expected] [['objects :source-array-declaration-conflict]
                           ['floats :source-array-declaration-conflict]
                           ['String :source-array-declaration-unsupported]
                           ['Object :source-array-declaration-unsupported]
                           ["[Ljava.lang.Object;" :source-array-declaration-unsupported]]]
    (let [source (declaration/capture-form
                  (list 'let* [(with-meta 'value {:tag hint}) '(clojure.core/double-array 1)] 'value))]
      (is (= expected
             (reason #(declaration/validate-native-array-declarations!
                        (list 'let* [(vary-meta (first (second source)) assoc :raster.type/tag 'doubles)
                                    '(clojure.core/double-array 1)] 'value))))))))

(deftest declaration-bearing-aliases-are-not-folded-away
  (let [source (declaration/capture-form
                (list 'let* ['first 'input (with-meta 'checked {:tag 'objects}) 'input] 'checked))]
    (doseq [result [(pe/pe source) (:form (cse/cse-let source))]]
      (is (some #(= {:present? true :hint 'objects} (declaration/declaration %))
                (take-nth 2 (second result)))))))

(deftest public-gpu-compilation-refuses-before-source-checks-can-be-erased
  (let [forbidden (fn [& _] (throw (AssertionError. "declaration checking acquired GPU resources")))]
    (with-redefs [gpu/make-session forbidden gpu/alloc! forbidden]
      (doseq [source [#'incompatible-clone! #'incompatible-constructor!
                      #'forged-declaration! #'incompatible-inlined-helper! #'incompatible-store!]]
        (is (= :source-array-declaration-conflict
               (reason #(equation-first/compile source {:target :ze:debug :dtype :double})))))
      (is (map? (equation-first/compile #'compatible-clone! {:target :ze:debug :dtype :double}))))))

(deftest macro-local-clone-remains-an-explicit-coverage-refusal
  ;; A clone inside a map lambda is not the admitted top-level copy schedule. The
  ;; declaration is retained by expansion, but does not manufacture a schedule.
  (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "acquired device")))
                gpu/alloc! (fn [& _] (throw (AssertionError. "allocated device buffer")))]
    (is (= :equation-first-coverage
           (reason #(equation-first/compile #'incompatible-macro-helper!
                                           {:target :ze:debug :dtype :double})))))
  (doseq [simd? [false true]]
    (let [compiled (pipeline/compile-aot #'incompatible-macro-helper! :dtype :double :simd? simd?)]
      (is (thrown? ClassCastException (compiled (double-array 1) (double-array 1)))))))

(deftest source-declarations-retain-jvm-check-behavior
  (is (thrown? ClassCastException (incompatible-clone! (double-array [3.0]) (double-array 1))))
  (doseq [simd? [false true]]
    (let [compiled (pipeline/compile-aot #'incompatible-clone! :dtype :double :simd? simd?)]
      (is (thrown? ClassCastException (compiled (double-array [3.0]) (double-array 1)))))
    (let [compiled (pipeline/compile-aot #'incompatible-store! :dtype :double :simd? simd?)]
      (is (thrown? ClassCastException (compiled (double-array 1)))))
    (let [compiled (pipeline/compile-aot #'declared-output! :dtype :double :simd? simd?)]
      (is (= [3.0] (vec (declared-output! (double-array 1))) (vec (compiled (double-array 1))))
          "the typed map's explicit store contract can preserve the ordinary store result"))))

(deftest native-targets-reject-incompatible-declarations-before-emission
  (doseq [compile [#(pipeline/compile-aot % :target :c :dtype :double)
                  #(pipeline/compile-wasm % :dtype :double)]
          source [#'incompatible-clone! #'incompatible-constructor! #'incompatible-store!]]
    (is (= :source-array-declaration-conflict
           (reason #(compile source))))))

(deftest shadowed-array-declarations-remain-lexically-distinct-in-public-compilation
  (is (map? (equation-first/compile #'mixed-shadow!
                                  {:target :ze:debug :dtype :double
                                   :preserve-declared-array-storage? true}))))

(deftest semantic-compatibility-is-not-vector-operand-compatibility
  (doseq [hint [Object Cloneable java.io.Serializable]]
    (let [source (declaration/capture-form
                  (list 'let* [(with-meta 'value {:tag hint}) 'input] 'value))
          binder (vary-meta (first (second source)) assoc :raster.type/tag 'doubles)
          source (list 'let* [binder 'input] 'value)]
      (is (nil? (declaration/array-declaration-violation source :native)))
      (is (= :source-array-declaration-conflict
             (:reason (declaration/array-declaration-violation source :simd)))))))
