(ns raster.compiler.ir.kernel-graph-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]))

(defn- symbolic-scan []
  (soac/par-form->soac
   'scan-result
   '(raster.par/scan out acc 0.0 i n double (+ acc (aget values i)))
   41))

(deftest three-stage-scan-has-explicit-scheduled-dataflow
  (let [node (symbolic-scan)
        operations (lower/lower-scan node nil)
        scheduled (lower/scan-kernel-graph node operations)
        [intra block carry] (:nodes scheduled)
        temporary (first (:temporaries scheduled))]
    (is (graph/kernel-graph? scheduled))
    (is (= :three-stage (get-in scheduled [:attributes :strategy])))
    (is (= ['values] (mapv :id (:inputs scheduled))))
    (is (= ['out] (mapv :id (:outputs scheduled))))
    (is (= 1 (count (:temporaries scheduled))))
    (is (= :temporary (:role temporary)))
    (is (not (re-find #"min" (pr-str (get-in operations [0 :grid :num-blocks]))))
        "scan coverage is an uncapped ceil-div grid, never map/reduce grid-stride virtualization")
    (is (= (:id temporary)
           (first (disj (:outputs (:operation intra)) 'out)))
        "stage 1 produces the same stable block-totals buffer consumed by stages 2 and 3")
    (is (= [(:id intra)] (:dependencies block)))
    (is (= (mapv :id [intra block]) (:dependencies carry))
        "the explicit schedule contains every RAW/WAW hazard; a later pass may transitively reduce it")
    (is (= [:intra-block :block-scan nil]
           (mapv #(get-in % [:operation :phase]) (:nodes scheduled))))))

(deftest graph-validation-rejects-the-old-disconnected-scan
  (let [node (symbolic-scan)
        operations (lower/lower-scan node nil)
        totals (first (disj (:outputs (first operations)) 'out))
        disconnected (assoc-in operations [0 :outputs] #{'out})]
    (testing "ledger #41: stage 2 cannot read block totals that stage 1 never wrote"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"reads a buffer before any graph node writes it"
           (graph/from-segops
            disconnected
            {:inputs #{'values}
             :outputs #{'out}
             :temporaries {totals {:dtype :double :elements 'num-blocks}}
             :dtype :double}))))))

(deftest graph-validation-rejects-an-unscheduled-memory-hazard
  (let [node (symbolic-scan)
        operations (lower/lower-scan node nil)
        scheduled (lower/scan-kernel-graph node operations)
        missing-edge (assoc-in scheduled [:nodes 1 :dependencies] [])]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"omits a memory-hazard dependency"
                          (graph/validate! missing-edge)))))

(deftest single-stage-scan-is-the-degenerate-kernel-graph
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i 64 double (+ acc (aget values i)))
              5)
        operations (lower/lower-scan node nil)
        scheduled (lower/scan-kernel-graph node operations)]
    (is (= :single (get-in scheduled [:attributes :strategy])))
    (is (= 1 (count (:nodes scheduled))))
    (is (empty? (:temporaries scheduled)))
    (is (empty? (:dependencies (first (:nodes scheduled)))))))

(deftest in-place-dataflow-retains-one-buffer-identity
  (let [operation (segop/->SegMap
                   9 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(inc (aget state i)) nil #{'state} #{'state} #{}
                   (segop/->KernelGrid 1 32 0) :float 'state nil)
        scheduled (graph/from-segops [operation]
                                     {:inputs #{'state} :outputs #{'state} :dtype :float})]
    (is (= :inout (get-in scheduled [:inputs 0 :role])))
    (is (identical? (first (:inputs scheduled)) (first (:outputs scheduled))))
    (is (= :read-write (get-in scheduled [:nodes 0 :uses 0 :access])))))

(deftest in-place-graph-interface-retains-the-read-write-abi-direction
  (let [operation (segop/->SegMap
                   9 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(inc (aget state i)) nil #{'state} #{'state} #{}
                   (segop/->KernelGrid 1 32 0) :float 'state nil)
        scheduled (graph/from-segops [operation]
                                     {:inputs #{'state} :outputs #{'state} :dtype :float})
        interfaced (assoc scheduled
                          :abi [(kabi/slot 'state :inout :float)
                                (kabi/slot 'n :scalar :long)]
                          :arguments '[state n])]
    (is (identical? interfaced (graph/validate! interfaced)))
    (is (= :inout (get-in interfaced [:abi 0 :kind])))))

(deftest graph-scalars-are-an-ordered-target-neutral-interface
  (let [operation (segop/->SegMap
                   10 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(+ alpha (aget values i)) nil #{'values} #{'out} #{'alpha}
                   (segop/->KernelGrid 1 32 0) :float 'out nil)
        scalars [(graph/scalar 'n :long) (graph/scalar 'alpha :float)]
        scheduled (graph/from-segops
                   [operation]
                   {:inputs #{'values} :outputs #{'out} :dtype :float :scalars scalars})]
    (is (= scalars (:scalars scheduled)))
    (is (= #{'n 'alpha} (get-in scheduled [:nodes 0 :scalar-uses])))
    (is (= :kernel-graph-node-scalar-use
           (try
             (graph/validate! (assoc-in scheduled [:nodes 0 :scalar-uses] #{'n}))
             nil
             (catch clojure.lang.ExceptionInfo exception
               (:reason (ex-data exception))))))
    (is (not (graph/dataflow-equivalent?
              scheduled (assoc scheduled :scalars (vec (reverse scalars))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar identities must be unique"
                          (graph/validate!
                           (assoc scheduled :scalars [(graph/scalar 'n :long)
                                                      (graph/scalar 'n :long)]))))
    (is (= :kernel-graph-value-identity
           (try
             (graph/validate!
              (assoc scheduled :scalars [(graph/scalar 'values :long)]))
             nil
             (catch clojure.lang.ExceptionInfo exception
               (:reason (ex-data exception))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"symbol or keyword"
                          (graph/scalar 42 :int)))))

(deftest ordered-output-vectors-retain-write-access
  (let [operation (segop/->SegFoldMap
                   10 (segop/make-seg-space 'segment 'nsegments)
                   'index 'width [] [0.0] #{'values} ['out] #{}
                   (segop/->KernelGrid 1 32 0) [:float] :no-write-alias)
        scheduled (graph/from-segops [operation]
                                     {:inputs #{'values} :outputs #{'out} :dtype :float})]
    (is (= :write (->> scheduled :nodes first :uses
                       (filter #(= 'out (:buffer %))) first :access))
        "ordered tuple outputs are value collections, not associative-set indices")))

(deftest target-emission-cannot-change-the-scheduled-dataflow-contract
  (let [operation (segop/->SegMap
                   9 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(inc (aget values i)) nil #{'values} #{'out} #{}
                   (segop/->KernelGrid 1 32 0) :float 'out nil)
        scheduled (graph/from-segops [operation]
                                     {:inputs #{'values} :outputs #{'out} :dtype :float})
        emitted-shape (graph/map-operations scheduled (constantly :artifact))]
    (is (graph/dataflow-equivalent? scheduled emitted-shape))
    (is (not (graph/dataflow-equivalent?
              scheduled (assoc emitted-shape :effects {:semantic #{:io}}))))
    (is (not (graph/dataflow-equivalent?
              scheduled (assoc-in emitted-shape [:inputs 0 :elements] 'different-extent))))))

(deftest scan-graph-preserves-per-buffer-storage-dtypes
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n double
                                (+ acc (double (aget integer-values i))))
              12)
        operations (lower/lower-scan node nil)
        scheduled (lower/scan-kernel-graph
                   node operations {:array-types {'integer-values :int 'out :double}})]
    (is (= :int (get-in scheduled [:inputs 0 :dtype])))
    (is (= :double (get-in scheduled [:outputs 0 :dtype])))
    (is (= :double (get-in scheduled [:temporaries 0 :dtype])))
    (is (= 'n (get-in scheduled [:inputs 0 :elements])))
    (is (= 'n (get-in scheduled [:outputs 0 :elements])))))
