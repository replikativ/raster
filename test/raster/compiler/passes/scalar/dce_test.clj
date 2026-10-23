(ns raster.compiler.passes.scalar.dce-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.dce :as dce]
            [raster.compiler.passes.scalar.effects :as effects]))

;; ================================================================
;; Dead binding removal
;; ================================================================

(deftest per-binding-syntactic-free-symbols-are-derived-once
  (let [store '(clojure.core/aset buf 0 1.0)
        compute '(+ x 2)
        form (with-meta (list 'let* ['effect store 'result compute] 'result)
               {:source :dce-once})
        free-var #'dce/free-syms
        original-free @free-var
        free-calls (atom {})
        observe (fn [calls original expr]
                  (when (or (identical? expr store) (identical? expr compute))
                    (swap! calls update expr (fnil inc 0)))
                  (original expr))
        result (with-redefs-fn
                 {free-var #(observe free-calls original-free %)}
                 #(dce/eliminate-dead-bindings form))]
    (is (= form (:form result)))
    (is (= (meta form) (meta (:form result))))
    (is (= {store 1 compute 1} @free-calls))
    (is (= {:bindings-removed 0} (:stats result)))))

(deftest mutation-facts-remain-fresh-after-effect-analysis
  (let [store '(late-effect buf)
        descriptor-var #'effects/descriptor
        mutation-var #'dce/extract-mutation-targets
        original-descriptor @descriptor-var original-mutation @mutation-var
        initialized? (atom false) queries (atom [])
        form (list 'let* ['effect store 'result 0] 'result)
        result (with-redefs-fn
                 {descriptor-var
                  (fn [expr]
                    (if (identical? expr store)
                      (do (reset! initialized? true) {:effect :local :flags #{}})
                      (original-descriptor expr)))
                  mutation-var
                  (fn [expr]
                    (if (identical? expr store)
                      (do (swap! queries conj @initialized?)
                          (if @initialized? '#{buf} #{}))
                      (original-mutation expr)))}
                 #(dce/eliminate-dead-bindings form))]
    (is (= form (:form result)))
    (is (= {:bindings-removed 0} (:stats result)))
    (is (= [false true true] @queries))))

(deftest mutation-targets-traverse-maps-test
  ;; Regression: extract-mutation-targets must descend into map literals. A `case*`
  ;; clause map {hash [test (aset buf …)]} can hold a mutation; missing it made DCE
  ;; fail to seed the mutated buffer live → it could eliminate the live mutating
  ;; binding (same map-blind class as the free-syms-flat / case* fix in #18).
  (let [emt @#'dce/extract-mutation-targets
        form '(case* g 0 0 (throw e)
                     {0 [0 (clojure.core/aset buf (long i) 1.0)]
                      1 [1 (clojure.core/aset other (long i) 2.0)]}
                     :compact :int)]
    (testing "a mutation inside a case* clause map is seen as a mutation target"
      (is (contains? (emt form) 'buf))
      (is (contains? (emt form) 'other)))))

(deftest dead-binding-removal-test
  (testing "unused pure binding is removed"
    (let [form '(let* [x 1 y 2] x)
          {:keys [form stats]} (dce/eliminate-dead-bindings form)]
      (is (= 1 (:bindings-removed stats))
          "y is dead and should be removed")
      (let [[_ bindings _] form]
        (is (= '[x 1] (vec bindings))
            "only x binding should remain"))))

  (testing "all-live bindings are preserved"
    (let [form '(let* [x 1 y (+ x 2)] y)
          {:keys [form stats]} (dce/eliminate-dead-bindings form)]
      (is (= 0 (:bindings-removed stats)))
      (let [[_ bindings _] form]
        (is (= 4 (count bindings))
            "both bindings should remain"))))

  (testing "transitive liveness: y keeps x alive"
    (let [form '(let* [x 1 y (+ x 2) z 3] y)
          {:keys [form stats]} (dce/eliminate-dead-bindings form)]
      (is (= 1 (:bindings-removed stats))
          "z is dead, x and y are live")
      (let [[_ bindings _] form
            syms (set (take-nth 2 bindings))]
        (is (contains? syms 'x))
        (is (contains? syms 'y))
        (is (not (contains? syms 'z)))))))

;; ================================================================
;; Effectful binding preservation
;; ================================================================

(deftest effectful-binding-preservation-test
  (testing "binding marked effectful with unknown target is kept"
    ;; When a binding has :raster.effect/effectful metadata and no
    ;; identifiable mutation targets, DCE keeps it (opaque effect).
    (let [form (list 'let*
                     [(with-meta 'x {:raster.effect/effectful true})
                      '(println "side effect")
                      'y 2]
                     'y)
          {:keys [stats]} (dce/eliminate-dead-bindings form)]
      ;; The println binding should be kept as an opaque effect
      (is (= 0 (:bindings-removed stats))
          "effectful binding should be preserved"))))

(deftest unused-checked-cast-binding-is-preserved
  (let [form '(let* [checked (clojure.core/int x) result 0] result)
        {lowered :form stats :stats} (dce/eliminate-dead-bindings form)]
    (is (= 0 (:bindings-removed stats)))
    (is (= '[checked (clojure.core/int x) result 0] (second lowered))
        "DCE must retain the possible exceptional control transfer")))

(deftest checked-conversion-survives-dead-local-writes-and-allocation
  (doseq [source
          ['(let* [dead (clojure.core/double-array 1)
                   effect (do (clojure.core/int x) (clojure.core/aset dead 0 1.0)) result 0] result)
           '(let* [dead (clojure.core/double-array (clojure.core/int x)) result 0] result)]]
    (let [lowered (:form (dce/eliminate-dead-bindings source))
          run (fn [body x]
                (eval (list 'let ['x x] body)))]
      (is (= 0 (run source 1)))
      (is (= 0 (run lowered 1)))
      (is (thrown? ArithmeticException (run source 2147483648)))
      (is (thrown? ArithmeticException (run lowered 2147483648))))))

(deftest nonlocal-effects-survive-dead-local-writes-and-pure-callee-arguments
  (doseq [source
          ['(let* [dead (clojure.core/double-array 1)
                   effect (do (clojure.core/println "opaque")
                              (clojure.core/aset dead 0 1.0)) result 0] result)
           '(let* [dead (clojure.core/double-array 1)
                   effect (if positive?
                            (do (clojure.core/println "branch") (clojure.core/aset dead 0 1.0))
                            nil) result 0] result)
           '(let* [dead (clojure.core/double-array 1)
                   effect (loop [i 0]
                            (if (< i 2)
                              (do (clojure.core/println i) (clojure.core/aset dead 0 1.0)
                                  (recur (inc i))) nil)) result 0] result)
           '(let* [dead (clojure.core/double-array (do (clojure.core/println "argument") 1))
                   result 0] result)]]
    (let [lowered (:form (dce/eliminate-dead-bindings source))
          run (fn [body positive?]
                (let [writer (java.io.StringWriter.)
                      value (binding [*out* writer]
                              (eval (list 'let ['positive? positive?] body)))]
                  {:value value :output (str writer)}))]
      (doseq [condition [true false]]
        (is (= (run source condition) (run lowered condition)))))))

(deftest invk-pure-callee-transport-does-not-hide-argument-effects
  ;; A compiler call marker, not an executable fake method: this is a denial control for
  ;; retained original-op transport through the shared semantic-call/argument authority.
  (let [call (with-meta
               '(.invk zeros-like-implementation input (do (clojure.core/println "argument") 1))
               {:raster.op/original 'raster.arrays/zeros-like})
        source (list 'let* ['dead call 'result 0] 'result)
        lowered (:form (dce/eliminate-dead-bindings source))]
    (is (= :io (:effect (effects/descriptor call))))
    (is (= source lowered))
    (is (= (meta call) (meta (second (second lowered)))))))

(deftest arraycopy-loop-mutation-preservation-test
  (let [emt @#'dce/extract-mutation-targets]
    (testing "System.arraycopy identifies only its destination as mutated"
      (is (= '#{dst}
             (emt '(java.lang.System/arraycopy src 0 dst 0 n)))))

    (testing "an unused loop result survives when arraycopy updates a live destination"
      (let [effect-sym (with-meta '_effect {:raster.effect/effectful true})
            form (list 'let*
                       ['u '(clojure.core/aclone input)
                        effect-sym '(dotimes [step nsteps]
                                      (java.lang.System/arraycopy tmp 0 u 0 n))
                        'result '(clojure.core/aget u 0)]
                       'result)
            {:keys [form stats]} (dce/eliminate-dead-bindings form)
            kept (set (take-nth 2 (second form)))]
        (is (zero? (:bindings-removed stats)))
        (is (contains? kept effect-sym))))))

;; ================================================================
;; Nested let* with partially dead bindings
;; ================================================================

(deftest partially-dead-bindings-test
  (testing "multiple dead bindings among live ones"
    (let [form '(let* [a 1 b 2 c 3 d (+ a c)] d)
          {:keys [form stats]} (dce/eliminate-dead-bindings form)]
      (is (= 1 (:bindings-removed stats))
          "b is dead, a and c are needed by d")
      (let [[_ bindings _] form
            syms (set (take-nth 2 bindings))]
        (is (not (contains? syms 'b)))
        (is (contains? syms 'a))
        (is (contains? syms 'c))
        (is (contains? syms 'd)))))

  (testing "all bindings dead except body reference"
    (let [form '(let* [a 1 b 2 c 3] 42)
          {:keys [stats]} (dce/eliminate-dead-bindings form)]
      (is (= 3 (:bindings-removed stats))
          "all bindings are dead when body is a constant"))))

;; ================================================================
;; Pure par form elimination when unused
;; ================================================================

(deftest pure-par-unused-test
  (testing "external purity does not discharge reduction arithmetic"
    (doseq [[reduction removed]
            [['(raster.par/reduce acc 0 i 3 (+ acc i)) 0]
             ['(raster.par/reduce acc 0 i 0 acc) 1]]]
      (let [{:keys [stats]} (dce/eliminate-dead-bindings (list 'let* ['x reduction 'y 42] 'y))]
        (is (= removed (:bindings-removed stats)))))))

;; ================================================================
;; Root-pred option
;; ================================================================

(deftest root-pred-test
  (testing "root-pred keeps otherwise-dead bindings alive"
    (let [form '(let* [x 1 y 2 z 3] z)
          {:keys [stats]} (dce/eliminate-dead-bindings form
                                                       :root-pred #(= % 'y))]
      ;; x is dead, y is kept by root-pred, z is live from body
      (is (= 1 (:bindings-removed stats))
          "only x should be removed; y is a root"))))

;; ================================================================
;; Declared output buffer of a mutating op (the frozen-weight dW leak)
;; ================================================================

(deftest declared-output-arg-narrows-mutation-targets-test
  ;; A mutating op that DECLARES its output buffer writes exactly that argument; the
  ;; other operands are read-only. Treating them as mutation targets kept DEAD calls
  ;; alive, because DCE keeps a mutating binding whose target is live — and a GEMM's
  ;; A/B operands are of course live elsewhere. That is how a LoRA backward computed
  ;; full weight gradients for FROZEN weights (7 :tn GEMMs per gemma layer) whose
  ;; value+grad slots nobody ever reads.
  (require 'raster.linalg.blas)                ;; registers dgemm!'s :in-place-arg 2
  (let [emt @#'dce/extract-mutation-targets]
    (testing "a cblas gemm mutates ONLY its C argument"
      (is (= '#{dW} (emt '(raster.linalg.blas/dgemm-tn! dy x dW out-f batch in-f 1.0 0.0)))
          "dy and x are read-only operands, not mutation targets")
      (is (= '#{dx} (emt '(raster.linalg.blas/dgemm! dy W dx batch out-f in-f 1.0 0.0)))))

    (testing "an undeclared mutating op keeps the conservative any-arg approximation"
      (is (contains? (emt '(some.ns/unknown-mutator! a b c)) 'a))
      (is (contains? (emt '(some.ns/unknown-mutator! a b c)) 'b)))))

(deftest unknown-gemm-effects-are-not-authorized-by-dead-output
  ;; End-to-end on the flat IR the AD transform produces for a frozen-weight dW:
  ;;   dW_buf = (alloc)              ; fresh
  ;;   _      = (dgemm-tn! dy x dW_buf …)
  ;; with dy/x live (they feed the gradients we DO read) but dW_buf read by nobody.
  (require 'raster.linalg.blas)
  (let [form '(let* [dx_buf (clojure.core/double-array 8)
                     _1 (raster.linalg.blas/dgemm! dy W dx_buf 2 2 2 1.0 0.0)
                     dW_buf (clojure.core/double-array 8)
                     _2 (raster.linalg.blas/dgemm-tn! dy x dW_buf 2 2 2 1.0 0.0)]
                    dx_buf)
        {:keys [form stats]} (dce/eliminate-dead-bindings form)
        kept (set (map first (partition 2 (second form))))]
    (testing "the foreign GEMM lacks a complete local-only effect proof"
      (is (= :io (:effect (effects/descriptor
                           '(raster.linalg.blas/dgemm-tn! dy x dW_buf 2 2 2 1.0 0.0)))))
      (is (contains? kept 'dW_buf))
      (is (contains? kept '_2))
      (is (= 0 (:bindings-removed stats))))
    (testing "the dx GEMM (whose buffer IS read) survives"
      (is (contains? kept 'dx_buf))
      (is (contains? kept '_1)))))

(deftest unread-local-write-retains-arithmetic-completion
  (let [write '(clojure.core/aset dead 0 (+ (clojure.core/aget dy 0) (clojure.core/aget x 0)))
        source (list 'let* ['live '(clojure.core/double-array 1)
                           'write-live '(clojure.core/aset live 0 (clojure.core/aget dy 0))
                           'dead '(clojure.core/double-array 1) 'write-dead write] 'live)
        {lowered :form stats :stats} (dce/eliminate-dead-bindings source)
        kept (set (take-nth 2 (second lowered)))]
    (is (= {:effect :local :flags #{:unclosed-source-arithmetic}} (effects/descriptor write)))
    (is (= '#{live write-live dead write-dead} kept))
    (is (zero? (:bindings-removed stats)))))

(deftest verified-local-literal-write-into-unread-buffer-is-eliminated
  (let [source '(let* [live (clojure.core/double-array 1)
                      write-live (clojure.core/aset live 0 1.0)
                      dead (clojure.core/double-array 1)
                      write-dead (clojure.core/aset dead 0 2.0)] live)
        {lowered :form stats :stats} (dce/eliminate-dead-bindings source)]
    (is (= '#{live write-live} (set (take-nth 2 (second lowered)))))
    (is (= 2 (:bindings-removed stats)))
    (is (= [1.0] (vec (eval lowered))))))

(deftest dead-arithmetic-retains-exception-and-source-order
  (doseq [[expression bad-value] [['(clojure.core/* x 2) Long/MAX_VALUE]
                                  ['(clojure.core// 1 x) 0]]]
    (let [source (list 'let* ['before '(swap! events conj :before)
                              'unused expression
                              'after '(swap! events conj :after)] 7)
          lowered (:form (dce/eliminate-dead-bindings source))
          run (fn [body value]
                (let [events (atom [])
                      f (eval (list 'fn '[x events] body))
                      outcome (try [:value (f value events)]
                                   (catch ArithmeticException _ [:error ArithmeticException]))]
                  [outcome @events]))]
      (is (= [[:error ArithmeticException] [:before]] (run source bad-value)))
      (is (= (run source bad-value) (run lowered bad-value)))
      (is (= [[:value 7] [:before :after]] (run lowered 1))))))
