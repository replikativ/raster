(ns raster.compiler.passes.parallel.contract-descriptor-validator-test
  "A launch descriptor that under-describes its kernel fails at LAUNCH, not at compile — the kernel
   is valid C, the caller simply binds the wrong number of arguments. That bug class has bitten
   twice in this subsystem:

     1. invoke-registered-contraction! reconstructed scalar-args from a `case` with no default, so
        four of six strategies crashed when reached.
     2. an epilogue's operand arrays were declared in the emitted signature but absent from the
        descriptor, so a caller bound 6 args to a 7-arg kernel.

   Both are mechanically detectable by comparing the emitted signature with what the descriptor
   says to bind. This suite pins that every strategy stays consistent, and that the validator
   actually rejects each historical shape."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.passes.parallel.contract-route :as route]))

(defn- mm [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(defn- with-epilogue [epilogue]
  (let [contract (assoc (vec (mm 128 256 64)) 1 'out)]
    (apply list (concat contract [:epilogue epilogue]))))

(def ^:private epilogues
  {:activation {:acc 'acc :expr '(raster.math/exp acc) :dtype :float}
   :bias {:acc 'acc :expr '(raster.numeric/+ acc (clojure.core/aget bias j))
          :operands [{:sym 'bias :dtype :float :map (axis-map/of-axes '[[j 256]])}]
          :dtype :float}
   :resid {:acc 'acc :expr '(raster.numeric/+ acc
                                              (clojure.core/aget R (+ (* i 256) j)))
           :operands [{:sym 'R :dtype :float :map (axis-map/of-axes '[[i 128] [j 256]])}]
           :dtype :float}
   :rowscale {:acc 'acc :expr '(raster.numeric/* acc (clojure.core/aget rs i))
              :operands [{:sym 'rs :dtype :float :map (axis-map/of-axes '[[i 128]])}]
              :dtype :float}})

;; ── every strategy's descriptor matches the kernel it describes ───────────────────────
(deftest every-strategy-descriptor-is-consistent
  ;; route-contraction validates on the way out, so simply routing each shape is the assertion.
  (testing "the base strategies, including the explicit migration fallback"
    (is (= :dpas         (:strategy (route/route-contraction (mm 256 512 128) :dtype :half))))
    (is (= :regtiled     (:strategy (route/route-contraction (mm 96 96 64) :dtype :double))))
    (is (= :segmap       (:strategy (route/route-contraction
                                     '(raster.par/contract C [[i 4] [j 3]] [] (* (aget a i) (aget b j)))
                                     :dtype :double))))
    (is (= :portable-segred
           (:strategy (route/route-contraction
                       '(raster.par/contract C [[b 2] [i 4] [j 3]] [[l 5]]
                                             (* (aget A (+ (* (+ (* b 4) i) 5) l))
                                                (aget B (+ (* (+ (* b 5) l) 3) j))))
                       :dtype :double))))
    ;; The source template remains an explicit correctness fallback only where no physical
    ;; operand layout can be proven (these deliberately unbound gather coordinates model that).
    (is (= :naive-segred (:strategy (route/route-contraction
                                     '(raster.par/contract C [[b 2] [i 4] [j 3]] [[l 5]]
                                                           (* (aget A x) (aget B y))) :dtype :double))))
    (is (= :full-reduce  (:strategy (route/route-contraction
                                     '(raster.par/contract O [] [[i 8]] (* (aget A i) (aget B i)))
                                     :dtype :double))))
    (is (= :dp4a         (:strategy (route/route-contraction
                                     '(raster.par/contract C [[i 4] [j 4]] [[l 8]]
                                                           (* (aget A (+ (* i 8) l)) (aget B (+ (* j 8) l))))
                                     :dtype :byte))))
    (is (= :quant-naive  (:strategy (route/route-contraction
                                     '(raster.par/contract C [[i 4] [j 4]] [[l 8]]
                                                           (* (aget A (+ (* i 8) l)) (aget B (+ (* l 4) j))))
                                     :dtype :byte)))))
  (testing "fused epilogues, including the operand-carrying shapes"
    (doseq [[label epilogue] epilogues]
      (is (= :dpas (:strategy (route/route-contraction (with-epilogue epilogue) :dtype :half)))
          (str label " descriptor must validate")))))

;; ── the validator rejects each historical bug shape ──────────────────────────────────
(deftest validator-rejects-the-bugs-that-actually-happened
  (let [good (route/route-contraction (with-epilogue (:bias epilogues))
                                      :dtype :half)]
    (testing "bug 2: dropping the epilogue's operand under-describes the kernel"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"pointer params"
                            (route/validate-descriptor (assoc good :epilogue-operands [])))))
    (testing "bug 1: a scalar-arg count that disagrees with the signature"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar params"
                            (route/validate-descriptor (update good :scalar-args conj {:type :int :value 1}))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar params"
                            (route/validate-descriptor (assoc good :scalar-args [])))))
    (testing "the same counts in the wrong order are still an ABI mismatch"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"ordered ABI"
                            (route/validate-descriptor (update good :abi #(vec (reverse %)))))))
    (testing "a missing :out-elems (the invoke sizes the output with it)"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"out-elems"
                            (route/validate-descriptor (dissoc good :out-elems)))))
    (testing "malformed launch geometry"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"matching 1-3D geometry"
                            (route/validate-descriptor (assoc good :grid [4])))))))

(deftest validator-models-both-invoke-protocols
  (let [red (route/route-contraction '(raster.par/contract O [] [[i 8]] (* (aget A i) (aget B i)))
                                     :dtype :double)]
    (testing ":invoke :reduction owns its launch and carries an ordered ABI"
      (is (= :reduction (:invoke red)))
      (is (nil? (:wg red)))
      (is (nil? (:grid red)))
      (is (empty? (:scalar-args red)))
      (is (some? (:reduce-bound red)))
      (is (= '[A B O _n_bound] (mapv :name (:abi red))))
      (is (= '[A B O 8] (:arguments red))))
    (testing "…and the validator enforces each of those, so the protocol cannot drift"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must not"
                            (route/validate-descriptor (assoc red :wg [256 1]))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"ordered :abi is required"
                            (route/validate-descriptor (dissoc red :abi))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires :reduce-bound"
                            (route/validate-descriptor (dissoc red :reduce-bound)))))))

(deftest zero-free-contraction-is-a-real-flat-scalar-segred-schedule
  (let [routed (route/route-contraction
                '(raster.par/contract O [] [[i 4] [j 6]]
                                      (aget A (+ (* i 6) j)))
                :dtype :float)
        source (get-in routed [:artifact :provenance :scheduled-operation :source])]
    (is (= 24 (:reduce-bound routed))
        "the physical bound is the flattened product, not the first surface axis")
    (is (= 24 (get-in source [:space :dims 0 :bound])))
    (is (= :block-local (:phase source)))
    (is (= [:block :virtual] ((juxt :level :virt) (:level source))))
    (is (= :scalar-workgroup-tree (get-in source [:schedule :strategy])))
    (is (= (get-in source [:grid :num-blocks])
           (get-in source [:schedule :attributes :group-count])))))

(deftest scalar-contraction-affine-coordinates-require-the-exact-facts-proof
  (let [form '(raster.par/contract O [] [[i 8]] (aget A i))
        facts (contraction-facts/contraction-facts form :dtype :float)
        forgeries
        [(assoc-in facts [:operands 0 :idx] '(+ i 1024))
         ;; Repeated reads of one array are distinct proof obligations; validating only the first
         ;; occurrence would admit this unproved second coordinate.
         (update facts :operands conj {:sym 'A :idx '(+ i 1024) :map nil})]]
    (doseq [forged forgeries]
      (let [failure (try
                      (route/route-contraction nil :facts forged :dtype :float)
                      nil
                      (catch clojure.lang.ExceptionInfo exception exception))]
        (is (= :kernel-graph-target-lowering-missing (:reason (ex-data failure))))
        (is (= :contraction-coordinate-proof
               (get-in (ex-data failure) [:kernel-body-decline :missing-rule])))))))

(deftest signature-parser-handles-the-real-kernels
  (testing "the parser finds every param of a multi-line DPAS signature"
    (let [d (route/route-contraction (mm 256 512 128) :dtype :half)
          ps (route/kernel-signature-params (:source d))]
      (is (= 3 (count (filter #(clojure.string/includes? % "*") ps))) "A, B, C")
      (is (= 3 (count (remove #(clojure.string/includes? % "*") ps))) "M, N, K")))
  (testing "ABI C names use the emitter's symbol mangling"
    (let [d (route/route-contraction
             '(raster.par/contract out-buffer [[row-index 4]] []
                                   (aget input-buffer row-index))
             :dtype :double)]
      (is (= ["input_buffer" "out" "_nseg"] (mapv :c-name (:abi d)))))))
