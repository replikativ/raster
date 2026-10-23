(ns raster.compiler.allocation-frontend-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defmacro allocate-body [n] `(float-array ~n))
(core/deftm allocate-helper [ref :- (Array float) n :- Long] :- (Array float)
  (allocate-body n))
(core/deftm scalar-helper [x :- Double] :- Double (+ x 1.0))
(core/deftm storage-helper [ref :- (Array float) n :- Long] :- (Array float)
  (float-array n))

(deftest ordered-storage-facts-reach-the-actual-selected-allocation-helper
  (let [v (core/resolve-deftm-var #'storage-helper {:dtype :float})
        implementation (symbol (str (:ns (meta v))) (str (:name (meta v)) "-impl"))
        _ (core/ensure-bytecode-implementation!
           (ns-resolve (:ns (meta v)) (symbol (name implementation))))
        call (with-meta (list '.invk implementation 'alias 'size)
               {:raster.op/original 'raster.compiler.allocation-frontend-source-test/storage-helper})
        descriptions [{:kind :scalar :id 0 :sym 'alias :expr 'x}
                      {:kind :scalar :id 1 :sym 'size :expr '(clojure.core/* m n)}
                      {:kind :scalar :id 2 :sym 'out :expr call}]
        x (av/tensor {:dtype :float :shape ['m 'n]})
        analyze (fn [descriptions values]
                  (#'frontend/source-storage-values descriptions
                   {'x :float 'alias :float 'out :float}
                   {'m :long 'n :long}
                   {'m :long 'n :long 'size :long} values {}))
        result (analyze descriptions {'x x})
        allocation (first (:allocations result))]
    (is (= 1 (count (:allocations result))))
    (is (= 'out (:destination allocation)))
    (is (= implementation (get-in allocation [:source-projection :helper :implementation])))
    (is (= ['alias 'size] (get-in allocation [:source-projection :helper :arguments])))
    (is (identical? call (:source-expression allocation)))
    (is (identical? x (get-in result [:values 'alias])))
    (is (= :float (get-in result [:values 'out :dtype])))
    (is (empty? (:allocations (analyze descriptions {})))
        "array element declarations do not invent exemplar capacity")
    (is (empty? (:allocations (analyze (subvec (vec descriptions) 2) {'x x 'alias x})))
        "a future or missing scalar binding cannot be supplied by helper shape inference")))

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
