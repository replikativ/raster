(ns raster.compiler.backend.jvm.literal-carrier-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(defn- compile-value
  ([expression] (compile-value expression 'Object))
  ([expression return-tag]
  (let [{:keys [methods]}
        (bytecode/compile-specialized-class!
         (str "raster.compiled.LiteralCarrier_" (gensym))
         [{:name 'run :params [] :return-tag return-tag
           :walked-body [expression] :source-ns *ns*}])
        method (get methods 'run)]
    #(.invoke ^java.lang.reflect.Method method nil (object-array 0)))))

(deftest primitive-literal-carriers-survive-result-and-lexical-contexts
  (doseq [value [(long 1000) (long 5000000000)
                 (float 1.25) (double 1.25)]]
    (doseq [expression [value
                        (list 'let* ['value value] 'value)
                        (list 'do nil value)
                        (list 'if true value nil)]]
      (let [expected (eval expression) actual ((compile-value expression))]
        (is (= (class expected) (class actual)) (pr-str expression))
        (is (= expected actual))))))

(deftest overload-selection-and-emission-agree-on-the-literal-carrier
  (doseq [value [(long -1000) (long -5000000000)
                 (float -1.25) (double -1.25)]]
    (let [expression (list 'java.lang.Math/abs value)
          expected (eval expression) actual ((compile-value expression))]
      (is (= (class expected) (class actual)))
      (is (= expected actual)))))

(deftest retained-backend-int-literals-still-use-their-explicit-carrier
  ;; Backend-owned Integer constants are intentionally narrower than ordinary
  ;; source Long literals. Full source Object materialization is a separate
  ;; obligation; this asserts the retained backend dialect, not a Clojure hint.
  (doseq [value [(int 1000) (short 3) (byte 2)]]
    (is (= :int (#'bytecode/infer-arg-stack-type value {})))
    (is (= (int value) ((compile-value value 'int))))))

(deftest float-overloads-retain-their-own-rounding
  (let [expression (list 'java.lang.Math/nextUp (float 16777216.0))
        expected (eval expression) actual ((compile-value expression))]
    (is (instance? Float actual))
    (is (= (float 16777218.0) expected actual))))

(deftest unsupported-numeric-literals-never-truncate-through-int
  (doseq [value [(bigint "1208925819614629174706176") 1/3 (bigdec "1.25")]]
    (is (= :unsupported-jvm-numeric-literal
           (try (compile-value value 'double) :incorrectly-accepted
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
