(ns raster.compiler.passes.scalar.resolve-alength-test
  "resolve-alength rewrites (alength buf) to a static size expression before
  mem-merge resizes buffers. The call-through alias step must identify an
  into-variant's OUTPUT buffer authoritatively from the op-descriptor
  (:in-place-arg), never by guessing 'the last hoistable argument' — the latter
  mis-aliased (alength result) to a DY *input*, overrunning later loop bounds
  (a runtime OOB when the input is larger, a silent wrong-count when smaller)."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.resolve-alength :as ra]
            ;; load the ops whose buffer-semantics :in-place-arg we rely on
            [raster.dl.nn]
            [raster.nn]))

(defn- bindings-of [form] (second form))

(defn- init-of
  "Return the init expr bound to `sym` in the pass output form."
  [form sym]
  (->> (partition 2 (bindings-of form))
       (some (fn [[s init]] (when (= s sym) init)))))

(deftest into-variant-output-identified-by-in-place-arg
  (testing "UNtracked output + tracked input: (alength result) is NOT aliased to the input"
    ;; result = conv2d-backward-db-into! [dy=big(3456), db=out-buf(untracked)] -> db (arg 1).
    ;; The real output out-buf is out-of-scope (untracked); the DY input `big` IS a
    ;; tracked hoistable alloc. The OLD 'last hoistable arg' heuristic aliased
    ;; (alength result) -> (alength big) = 3456, overrunning the length-6 db. The
    ;; fix leaves (alength result) intact (true runtime length).
    (let [dybuf (vary-meta 'big assoc :raster.buffer/hoistable true)
          form  (list 'let*
                      (vector dybuf   '(clojure.core/double-array 3456)
                              'result '(raster.dl.nn/conv2d-backward-db-into! big out-buf 1 6 24 24)
                              'reduced '(clojure.core/alength result))
                      'reduced)
          out   (:form (ra/resolve-alength-pass form))]
      (is (= '(clojure.core/alength result) (init-of out 'reduced))
          "(alength result) must NOT be rewritten to (alength big)/3456")))

  (testing "TRACKED output buffer: (alength result) IS aliased to the output's size"
    ;; here the output db buffer is a locally-tracked hoistable alloc of size 6;
    ;; the pass should size-alias result -> 6 (so mem-merge resize stays correct).
    (let [dybuf  (vary-meta 'big assoc :raster.buffer/hoistable true)
          dbbuf  (vary-meta 'db  assoc :raster.buffer/hoistable true)
          form   (list 'let*
                       (vector dybuf   '(clojure.core/double-array 3456)
                               dbbuf   '(clojure.core/double-array 6)
                               'result '(raster.dl.nn/conv2d-backward-db-into! big db 1 6 24 24)
                               'reduced '(clojure.core/alength result))
                       'reduced)
          out    (:form (ra/resolve-alength-pass form))]
      (is (= 6 (init-of out 'reduced))
          "(alength result) aliases to the OUTPUT db buffer's size (6), not the input"))))

(deftest opaque-call-leaves-alength-intact
  (testing "an unregistered call with tracked args does NOT alias (fail-safe)"
    (let [buf  (vary-meta 'buf assoc :raster.buffer/hoistable true)
          form (list 'let*
                     (vector buf     '(clojure.core/double-array 100)
                             'result '(some.unknown/helper buf 7)
                             'reduced '(clojure.core/alength result))
                     'reduced)
          out  (:form (ra/resolve-alength-pass form))]
      (is (= '(clojure.core/alength result) (init-of out 'reduced))
          "no buffer-semantics => leave (alength result) intact"))))

(deftest direct-alloc-alength-still-resolves
  (testing "plain (alength local-alloc) still resolves to its static size (no regression)"
    (let [form (list 'let*
                     (vector 'a '(clojure.core/double-array 42)
                             'n '(clojure.core/alength a))
                     'n)
          out  (:form (ra/resolve-alength-pass form))]
      (is (= 42 (init-of out 'n))))))
