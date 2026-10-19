(ns raster.compiler.allocation-frontend-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]))

(defmacro allocate-body [n] `(float-array ~n))
(core/deftm allocate-helper [ref :- (Array float) n :- Long] :- (Array float)
  (allocate-body n))
(core/deftm scalar-helper [x :- Double] :- Double (+ x 1.0))

(deftest refused-helper-analysis-does-not-become-a-source-import
  (let [v (core/resolve-deftm-var #'scalar-helper {:dtype :double})
        _ (core/prepare-deftm-frontend! v)
        implementation (symbol (str (:ns (meta v))) (str (:name (meta v)) "-impl"))
        call (with-meta (list '.invk implementation 1.0)
               {:raster.op/original 'raster.compiler.allocation-frontend-source-test/scalar-helper})]
    (entry/call-with-definition-inputs
     (fn []
       (let [before @entry/*emission-inputs*]
         (is (nil? (descriptor/direct-allocation-projection call {})))
         (is (= before @entry/*emission-inputs*)))))))

(deftest direct-allocation-uses-the-original-frontend-not-an-earlier-walk-cache
  (let [v (core/resolve-deftm-var #'allocate-helper {:dtype :float})
        impl (ns-resolve (:ns (meta v)) (symbol (str (:name (meta v)) "-impl")))
        cache (core/ensure-walked-body! v)
        original @#'allocate-body
        operation 'raster.compiler.allocation-frontend-source-test/allocate-helper
        implementation (symbol (str (:ns (meta impl))) (str (:name (meta impl))))
        call (fn [argument]
               (with-meta (list '.invk implementation 'x argument)
                 {:raster.op/original operation}))]
    (try
      (alter-var-root #'allocate-body
                      (constantly (fn [_ _ n] (list 'clojure.core/float-array
                                                   (list 'clojure.core/inc n)))))
      (core/ensure-bytecode-implementation! impl)
      (is (= 2 (alength ^floats (allocate-helper (float-array 0) 1))))
      (is (= cache (:raster.core/deftm-walked-body (meta v))))
      (let [projected (descriptor/direct-allocation-projection
                       (call 1) {'x {:dtype :float :shape [0]}})]
        (is (= '(clojure.core/float-array (clojure.core/inc 1)) (:constructor projected)))
        (is (= ['x 1] (get-in projected [:helper :arguments]))))
      (doseq [argument [1.0 1N '(clojure.core/inc 0) 'unknown]]
        (is (nil? (descriptor/direct-allocation-projection
                   (call argument) {'x {:dtype :float :shape [0]}}))))
      (finally (alter-var-root #'allocate-body (constantly original))))))
