(ns raster.compiler.backend.gpu.c-emit-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.c-emit :as c-emit]))

(deftest retained-scalar-dtypes-are-not-collapsed-to-the-kernel-element-type
  (is (= :long (c-emit/scalar-parameter-dtype 'iteration {'iteration :int64} :float)))
  (is (= :byte (c-emit/scalar-parameter-dtype 'quantized {'quantized :int8} :float)))
  (is (= :float (c-emit/scalar-parameter-dtype 'length {'length :float32} :double)))
  (is (= :long (c-emit/scalar-parameter-dtype
                (with-meta 'renamed {:raster.type/tag 'long}) {} :float)))
  (is (= :byte (c-emit/scalar-parameter-dtype
                (with-meta 'packed {:raster.type/tag 'byte}) {} :float)))
  (is (= :int (c-emit/scalar-parameter-dtype 'row-count {} :float)))
  (is (= :double (c-emit/scalar-parameter-dtype 'scale {} :double)))
  (is (= "long"
         (binding [c-emit/*scalar-var-types* {'iteration "long"}
                   c-emit/*int-vars* #{'iteration}]
           (c-emit/infer-c-type 'iteration)))))
