(ns raster.compiler.passes.scalar.dce-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.dce :as dce]))

;; ================================================================
;; Dead binding removal
;; ================================================================

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
