(ns raster.compiler.backend.jvm.capture-carrier-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(defn- factory [params body]
  (let [artifact (bytecode/compile-specialized-class!
                  "raster.compiled.CaptureCarrierControl"
                  [{:name 'factory :params params :return-tag 'Object
                    :walked-body [body] :source-ns *ns*}])
        method ^java.lang.reflect.Method (get-in artifact [:methods 'factory])]
    (fn [& arguments]
      (try (.invoke method nil (object-array arguments))
           (catch java.lang.reflect.InvocationTargetException e (throw (.getCause e)))))))

(deftest boolean-local-captures-use-one-physical-descriptor
  (doseq [body ['(let* [flag (clojure.core/< x 0)] (fn [] flag))
                '(let* [flag (clojure.core/< x 0)] (reify clojure.lang.IFn (invoke [_] flag)))]]
    (let [make-fn (factory [(with-meta 'x {:tag 'long})] body)]
      (is (true? ((make-fn -1))))
      (is (false? ((make-fn 1)))))))

(deftest primitive-boolean-interface-and-with-meta-retain-capture-carriers
  (let [make-supplier (factory [(with-meta 'flag {:tag 'boolean})]
                              '(reify java.util.function.BooleanSupplier (getAsBoolean [_] flag)))]
    (doseq [value [true false]]
      (let [supplier (make-supplier value)
            copied (with-meta supplier {:checked true})]
        (is (= value (.getAsBoolean ^java.util.function.BooleanSupplier supplier)))
        (is (= value (.getAsBoolean ^java.util.function.BooleanSupplier copied)))
        (is (= {:checked true} (meta copied)))))))

(deftest mixed-width-captures-preserve-values-and-reference-identity
  (doseq [body ['(fn [] [flag n d f i payload])
                '(reify clojure.lang.IFn (invoke [_] [flag n d f i payload]))]]
   (let [params (mapv #(with-meta %1 {:tag %2}) '[flag n d f i payload]
                     '[boolean long double float int Object])
        make-fn (factory params body)]
    (doseq [flag [true false] payload [nil (double-array [0.2])]]
      (let [f (make-fn flag 17 0.4 (float 0.5) (int 6) payload)
            result (f)]
        (is (= [flag 17 0.4 (float 0.5) 6] (subvec result 0 5)))
        (is (identical? payload (nth result 5)))
        (when (instance? clojure.lang.IObj f)
          (let [copied (with-meta f {:copied true})
                result (copied)]
            (is (= [flag 17 0.4 (float 0.5) 6] (subvec result 0 5)))
            (is (identical? payload (nth result 5))))))))))
