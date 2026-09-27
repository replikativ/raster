(ns raster.compiler.passes.parallel.typed-soac-ownership-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-ownership :as ownership]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.dl.attention :as attention]))

(deftest unrelated-remainder-local-does-not-defeat-row-ownership
  (let [access {:index '(+ (* i width) j)
                :locals '[{:id unused :init (rem i width)}]
                :loops [{:index 'j :extent 'width :lower 0 :kind :fold}]}]
    (is (some? (#'ownership/ownership-signature access 'i 'rows #{} {}))
        "a remainder used only in a value predicate does not alias row-major addresses")
    (is (nil? (#'ownership/ownership-signature
               (assoc access :index '(+ (* unused width) j)) 'i 'rows #{} {}))
        "the same derived digit in the address still requires a separate proof")))

(deftest windowed-prefill-softmax-has-a-certified-parallel-row-kernel
  (let [compiled (pipeline/show-pipeline
                  #'attention/attn-prefill-softmax-windowed-head-major!
                  :target-device :ocl:0 :dtype :float)]
    (is (= :typed-soac (get-in compiled [:soac-fused-stats :route])))
    (is (= [:kernel-body]
           (mapv #(get-in % [:attributes :emission-route]) (:kernels compiled))))))

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

(deftest branch-ownership-includes-predicate-local-and-yield-reads
  (doseq [location [:predicate :local :yield]
          [read-index independent?] [['i true] [0 false] ['(clojure.core/inc i) false]]]
    (let [base (frontend/form->program
                (guarded-source 'i)
                {:dtype :double :array-types {'out :double} :scalar-types {'n :long}})
          [eq id results [op attributes inputs captures destinations [lam parameters _]]]
          (first (dialect/equations base))
          destination (first parameters)
          read (list 'clojure.core/aget destination read-index)
          branch (dialect/effect-branch
                  [{:result 'selected :dtype :double}]
                  (if (= location :predicate) (list 'clojure.core/> read 0.0) true)
                  (dialect/result-effect-region
                   (if (= location :local) [(dialect/local-value 'loaded :double read)] [])
                   [] [(case location :local 'loaded :yield read 1.0)])
                  (dialect/result-effect-region [] [] [0.0]))
          equation (list eq id results
                         (list op attributes inputs captures destinations
                               (list lam parameters
                                     (list 'effect-region []
                                           [branch (list 'effect destination :ordered 'i true 'selected)]))))
          original (dialect/make (dialect/facts base) [equation] (dialect/outputs base))
          [proved stats] (ownership/prove original)]
      (is (= (if independent? 1 0) (:effect-row-ownership-proofs stats))
          (str location " " read-index))
      (is (contains? (:inputs (first (lower/lower-typed-effect-map original :ze:0))) 'out)
          "predicate, local and yield reads keep the destination in the read ABI")
      (is (= (if independent? :independent :sequential)
             (get-in (dialect/operation-parts (first (dialect/equations proved)))
                     [:attributes :iteration-order]))))))
