(ns raster.compiler.backend.gpu.c-emit-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.c-emit :as c-emit]))

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
