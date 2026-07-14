(ns raster.compiler.passes.scalar.dce-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.dce :as dce]))

;; ================================================================
;; Dead binding removal
;; ================================================================

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
  (testing "pure par/reduce binding is removed when unused"
    ;; par/reduce is pure (no buffer mutation), so if its result is unused
    ;; it should be eliminated.
    (let [form '(let* [x (raster.par/reduce out init n (fn [acc i] (+ acc i)))
                       y 42]
                      y)
          {:keys [stats]} (dce/eliminate-dead-bindings form)]
      (is (= 1 (:bindings-removed stats))
          "unused par/reduce is pure and should be removed"))))

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

(deftest dead-gemm-into-unread-buffer-is-eliminated-test
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
    (testing "the dW GEMM and its buffer are dead"
      (is (not (contains? kept 'dW_buf)))
      (is (not (contains? kept '_2)))
      (is (= 2 (:bindings-removed stats))))
    (testing "the dx GEMM (whose buffer IS read) survives"
      (is (contains? kept 'dx_buf))
      (is (contains? kept '_1)))))
