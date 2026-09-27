(ns raster.compiler.passes.parallel.typed-soac-ownership-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-ownership :as ownership]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- guarded-source [read-index]
  (list 'let*
        ['effect (list 'raster.par/map-void! 'i 'n
                       (list 'if (list 'clojure.core/> (list 'clojure.core/aget 'out read-index) 0.0)
                             '(dotimes [k 1] (clojure.core/aset out i 1.0))))]
        'effect))

(deftest guarded-region-ownership-includes-entry-predicate-reads
  (doseq [[read-index independent?] [['i true] [0 false] ['(clojure.core/inc i) false]]]
    (let [source (guarded-source read-index)
          options {:dtype :double :array-types {'out :double} :scalar-types {'n :long}}
          original (frontend/form->program source options)
          [proved stats] (ownership/prove original)
          attributes (comp :attributes dialect/operation-parts first dialect/equations)
          attempted (route/attempt source :double {'out :double} options)]
      (is (= :sequential (:iteration-order (attributes original))))
      (is (= (if independent? 1 0) (:effect-row-ownership-proofs stats)))
      (is (= (if independent? :independent :sequential)
             (:iteration-order (attributes proved))))
      (if independent?
        (is (some? (:program attempted)))
        (is (= :sequential-effect-continuation (get-in attempted [:declined :reason])))))))
