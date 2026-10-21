(ns raster.compiler.passes.region-copy-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.region-copy :as region-copy]))

(defn- acopy-call
  [& arguments]
  (with-meta (apply list '.invk 'raster.arrays/acopy!_m_floats_long_floats_long_long-impl arguments)
    {:raster.op/original 'raster.arrays/acopy! :raster.type/tag 'floats :tag 'floats}))

(deftest no-copy-traversal-retains-exact-source-occurrences
  (let [source (with-meta '(let* [size (helper input)]
                            (do (if flag (left size) (right size))
                                (dotimes [i size] (touch i))
                                (raster.par/map-void! row size (touch row))
                                size))
                 {:line 19 :raster.type/tag 'long})
        result (region-copy/expand-region-copies source)]
    (is (= {:region-copies-expanded 0} (:stats result)))
    (is (identical? source (:form result)))
    (is (identical? (second source) (second (:form result))))))

(deftest changed-copy-preserves-unrelated-call-occurrences-and-source-metadata
  (let [kept (with-meta '(identity n) {:tag 'long})
        source (with-meta (list 'let* ['kept kept
                                       'effect '(java.lang.System/arraycopy src 0 dst 0 n)]
                                'dst)
                 {:line 29})
        result (region-copy/expand-region-copies source)
        lowered (:form result)
        execute (eval (list 'fn '[src dst n] lowered))]
    (is (= {:region-copies-expanded 1} (:stats result)))
    (is (not (identical? source lowered)))
    (is (identical? kept (second (second lowered))))
    (is (identical? (meta source) (meta lowered)))
    (doseq [n [0 1 3]]
      (let [src (double-array [2.0 4.0 8.0])
            expected (double-array [-1.0 -1.0 -1.0])
            actual (aclone expected)]
        (System/arraycopy src 0 expected 0 n)
        (is (identical? actual (execute src actual n)))
        (is (= (vec expected) (vec actual)))))))

(deftest quoted-copies-are-data-and-empty-do-is-still-empty
  (doseq [head ['quote 'clojure.core/quote]]
    (let [quoted (list head '(do (java.lang.System/arraycopy src 0 dst 0 n) dst))]
      (doseq [source [quoted (list 'do quoted nil)]]
        (let [result (region-copy/expand-region-copies source)]
          (is (zero? (get-in result [:stats :region-copies-expanded])))
          (is (identical? source (:form result)))))))
  (let [source '(do)
        result (:form (region-copy/expand-region-copies source))]
    (is (identical? source result))
    (is (nil? (eval result)))))

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
    (is (= 'long (:raster.type/tag (meta dst-index)))
        "synthesized address arithmetic retains the ordinal's long width")
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
    (is (= 'long (:raster.type/tag (meta (nth (nth loop 2) 2)))))
    (is (= 'long (:raster.type/tag (meta (nth (nth (nth loop 2) 3) 2)))))
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

(deftest a-copy-in-a-discarded-conditional-arm-is-a-statement
  (let [form '(let* [effect_1 (if flag
                                (java.lang.System/arraycopy src (int 0) out (int 0) n)
                                nil)]
                out)
        {:keys [form stats]} (region-copy/expand-region-copies form)
        [_ [_ conditional] _] form]
    (is (= {:region-copies-expanded 1} stats))
    (is (= 'dotimes (first (nth conditional 2))))))

(deftest a-copy-in-an-effect-map-body-is-a-statement
  (let [form '(let* [effect (raster.par/map-void!
                             block blocks
                             (let* [src-off (* block width)
                                    dst-off (* block width)]
                               (java.lang.System/arraycopy src src-off dst dst-off width)))]
                dst)
        {:keys [form stats]}
        (region-copy/expand-region-copies form :param-env '{src doubles dst doubles})]
    (is (= {:region-copies-expanded 1} stats))
    (is (some #(and (seq? %) (= 'dotimes (first %)))
              (tree-seq coll? seq form)))))

(deftest disagreeing-element-tags-keep-the-call
  ;; float[] → double[] is not one copy (the JVM rejects the call); nothing to spell
  (let [call '(java.lang.System/arraycopy src (int 0) out (int 0) n)
        form (list 'let* ['effect_1 call] 'out)
        {:keys [form stats]} (region-copy/expand-region-copies
                              form :param-env '{src floats out doubles})]
    (is (= {:region-copies-expanded 0} stats))
    (is (= call (second (second form))))))

(deftest the-read-is-typed-from-the-source-first
  (let [form '(let* [effect_1 (java.lang.System/arraycopy src (int 0) out (int 0) n)] out)
        {:keys [form]} (region-copy/expand-region-copies form :param-env '{src floats})
        [_ [_ loop] _] form]
    (is (= 'float (:raster.type/tag (meta (nth (nth loop 2) 3)))))))
