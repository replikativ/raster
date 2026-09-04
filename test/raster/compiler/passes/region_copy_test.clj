(ns raster.compiler.passes.region-copy-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.region-copy :as region-copy]))

(defn- acopy-call
  [& arguments]
  (with-meta (apply list '.invk 'raster.arrays/acopy!_m_floats_long_floats_long_long-impl arguments)
    {:raster.op/original 'raster.arrays/acopy! :raster.type/tag 'floats :tag 'floats}))

(deftest a-copy-per-row-is-a-nested-store-loop
  ;; linear's bias prefill: one acopy! per batch row, in the body of a dotimes
  (let [form (list 'let* ['_ (list 'dotimes '[i batch]
                                   (acopy-call 'b 0 'y '(clojure.core/* i out-f) 'out-f))]
                   'y)
        {:keys [form stats]} (region-copy/expand-region-copies form)
        [_ [_ loop] _] form
        [_ _ inner] loop
        [_ [index len] store] inner
        [_ dst dst-index read] store]
    (is (= {:region-copies-expanded 1} stats))
    (is (= 'dotimes (first inner)))
    (is (= 'out-f len))
    (is (= 'y dst))
    (is (= (list 'clojure.core/+ '(clojure.core/* i out-f) index) dst-index)
        "the destination index is the copy offset plus the element ordinal")
    (is (= (list 'clojure.core/aget 'b index) read) "a zero source offset is the bare ordinal")
    (is (= 'float (:raster.type/tag (meta read)))
        "the read carries the copied element type the call states")))

(deftest an-effect-binding-copy-becomes-its-loop
  ;; the inlined JVM primitive returns nothing, so its binding value is never read; the walker
  ;; spells literal offsets as `(int k)`, and a zero offset is the bare ordinal
  (let [form '(let* [out (clojure.core/float-array n)
                     effect_1 (java.lang.System/arraycopy src (int 2) out (int 3) n)
                     effect_2 (java.lang.System/arraycopy src (int 0) out (int 0) n)]
                out)
        {:keys [form stats]} (region-copy/expand-region-copies form)
        [_ [_ _ _ loop _ zero-loop] _] form]
    (is (= {:region-copies-expanded 2} stats))
    (is (= 'dotimes (first loop)))
    (is (= '(clojure.core/aset out (clojure.core/+ 3 rstr_copy_1)
                               (clojure.core/aget src (clojure.core/+ 2 rstr_copy_1)))
           (nth loop 2)))
    (is (= '(clojure.core/aset out rstr_copy_2 (clojure.core/aget src rstr_copy_2))
           (nth zero-loop 2)))))

(deftest the-read-is-typed-by-a-binder-or-parameter-tag
  (testing "the destination binder's tag"
    (let [out (with-meta 'out {:raster.type/tag 'doubles})
          form (list 'let* [out '(clojure.core/double-array n)
                            'effect_1 '(java.lang.System/arraycopy src (int 0) out (int 0) n)]
                     'out)
          {:keys [form]} (region-copy/expand-region-copies form)
          [_ [_ _ _ loop] _] form]
      (is (= 'double (:raster.type/tag (meta (nth (nth loop 2) 3)))))))
  (testing "a deftm parameter's tag"
    (let [form '(let* [effect_1 (java.lang.System/arraycopy src (int 0) out (int 0) n)] out)
          {:keys [form]} (region-copy/expand-region-copies form :param-env '{src floats out floats})
          [_ [_ loop] _] form]
      (is (= 'float (:raster.type/tag (meta (nth (nth loop 2) 3)))))))
  (testing "no fact: the read stays untagged for downstream typing to decide"
    (let [form '(let* [effect_1 (java.lang.System/arraycopy src (int 0) out (int 0) n)] out)
          {:keys [form]} (region-copy/expand-region-copies form)
          [_ [_ loop] _] form]
      (is (nil? (meta (nth (nth loop 2) 3)))))))

(deftest a-copy-whose-value-is-read-keeps-its-call
  (let [call (acopy-call 'src 0 'dst 0 'n)
        form (list 'let* ['copied call] '(clojure.core/aget copied 0))
        {:keys [form stats]} (region-copy/expand-region-copies form)]
    (is (= {:region-copies-expanded 0} stats))
    (is (= call (second (second form))))))

(deftest a-copy-within-one-array-keeps-its-call
  ;; System/arraycopy moves overlapping regions correctly; an elementwise loop would not
  (let [form '(let* [effect_1 (java.lang.System/arraycopy buf 0 buf 1 n)] buf)
        {:keys [form stats]} (region-copy/expand-region-copies form)]
    (is (= {:region-copies-expanded 0} stats))
    (is (= '(java.lang.System/arraycopy buf 0 buf 1 n) (second (second form))))))
