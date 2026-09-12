(ns raster.compiler.backend.gpu.c-emit-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.c-emit :as c-emit]))

(deftest qualified-primitive-casts-emit-like-their-bare-spelling
  (doseq [cast '[double float long int byte]]
    (is (= (c-emit/emit-expr (list cast 'x) 'i #{} "i")
           (c-emit/emit-expr (list (symbol "clojure.core" (name cast)) 'x) 'i #{} "i")))))

(deftest explicit-unchecked-integer-casts-require-integral-evidence
  (doseq [value [4294967295 (with-meta 'word {:raster.type/tag 'long})]]
    (is (.startsWith (c-emit/emit-expr (list 'clojure.core/unchecked-int value) nil #{} "idx")
                     "(int)(")))
  (is (= "(int)(word)"
         (binding [c-emit/*scalar-var-types* {'word "long"}]
           (c-emit/emit-expr '(clojure.core/unchecked-int word) nil #{} "idx"))))
  (doseq [value [1.5 'unknown (with-meta 'floating {:raster.type/tag 'double})]]
    (is (= :unchecked-cast-source-type
           (try (c-emit/emit-expr (list 'clojure.core/unchecked-int value) nil #{} "idx")
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest portable-identifiers-cover-the-cuda-and-hip-cpp-language
  (is (not (c-emit/c-identifier? "class")))
  (is (not (c-emit/c-identifier? "default")))
  (is (not (c-emit/c-identifier? "template")))
  (is (not (c-emit/c-identifier? "char8_t")))
  (is (not (c-emit/c-identifier? "_Atomic")))
  (is (not (c-emit/c-identifier? "read_only")))
  (is (not (c-emit/c-identifier? "__device__")))
  (is (not (c-emit/c-identifier? "_Reserved")))
  (is (= "class_" (c-emit/c-symbol 'class)))
  (is (= "default_" (c-emit/c-symbol 'default)))
  (is (= "rstr_Atomic" (c-emit/c-symbol '_Atomic)))
  (is (= "read_only_" (c-emit/c-symbol 'read-only)))
  (is (= "rstr__device__" (c-emit/c-symbol '__device__)))
  (is (c-emit/c-identifier? "ordinary_kernel")))

(deftest helper-naming-is-independent-of-helper-source-emission
  (is (= "gpufn_wi8_dot_q4_x8"
         (c-emit/gpu-helper-c-name 'raster.quant.kernels/wi8-dot-q4-x8))))

(deftest retained-scalar-dtypes-are-not-collapsed-to-the-kernel-element-type
  (is (= :long (c-emit/scalar-parameter-dtype 'iteration {'iteration :int64} :float)))
  (is (= :byte (c-emit/scalar-parameter-dtype 'quantized {'quantized :int8} :float)))
  (is (= :float (c-emit/scalar-parameter-dtype 'length {'length :float32} :double)))
  (is (= :long (c-emit/scalar-parameter-dtype
                (with-meta 'renamed {:raster.type/tag 'long}) {} :float)))
  (is (= :byte (c-emit/scalar-parameter-dtype
                (with-meta 'packed {:raster.type/tag 'byte}) {} :float)))
  ;; A floating tag specializes to the kernel dtype, as a declared floating param does, so host
  ;; encoding and kernel declaration agree on width.
  (is (= :float (c-emit/scalar-parameter-dtype
                 (with-meta 'scale {:raster.type/tag 'double}) {} :float)))
  ;; No declaration and no tag: refused with a structured reason, never guessed from the name
  ;; or from the kernel dtype.
  (is (= :kernel-scalar-dtype-unknown
         (try (c-emit/scalar-parameter-dtype 'row-count {} :float)
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (is (= :kernel-scalar-dtype-unknown
         (try (c-emit/scalar-parameter-dtype 'scale {} :double)
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (is (= "long"
         (binding [c-emit/*scalar-var-types* {'iteration "long"}
                   c-emit/*int-vars* #{'iteration}]
           (c-emit/infer-c-type 'iteration)))))

(deftest local-declarations-prefer-the-retained-binding-result-type
  (let [binding (with-meta 'dot-product {:raster.type/tag 'long})
        ;; Model an inlined helper whose outer collection lost its result metadata.
        untyped-rhs '(let* [k 0] k)]
    (is (= "long"
           (binding [c-emit/*scalar-type* "double"]
             (c-emit/decl-type binding untyped-rhs)))))
  (is (= :unsupported-retained-scalar-type
         (try
           (c-emit/decl-type (with-meta 'value {:raster.type/tag 'not-a-scalar}) 0)
           (catch clojure.lang.ExceptionInfo e
             (:reason (ex-data e)))))))
