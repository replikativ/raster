(ns raster.compiler.passes.parallel.staged-contract-route-test
  "Routing + surface for a STAGED contract axis. Device-free.

   Three things are checked, because a staged contraction has three places to go wrong:
     1. ROUTING — `:stages` reaches the multi-level leaf and the descriptor it produces survives
        validate-descriptor (the per-stage scale arrays are extra POINTER params, so a descriptor
        that forgets them is the exact launch-arity bug class the validator exists for).
     2. DISPATCH ORDER — staging is a property of the REDUCTION, not the dtype, so it must win
        over the int8 leaves rather than being shadowed by them.
     3. THE SURFACE — `par/contract` with `:stages` must compute the flat EQUIVALENT, not the bare
        body. Ignoring the option would silently drop every lift factor (i.e. every quant scale),
        which is the one failure mode that produces plausible wrong numbers instead of an error."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-route :as cr]
            [raster.par :as par]
            [raster.numeric]))

(def ^:private M 4) (def ^:private N 6)
(def ^:private NB 4) (def ^:private B 32) (def ^:private K (* NB B))

(defn- staged-form [dtype]
  (list 'raster.par/contract 'out [['i M] ['j N]] [['blk NB] ['t B]]
        (list 'raster.numeric/*
              (list 'aget 'a (list 'clojure.core/+ (list 'clojure.core/* 'i K)
                                   (list 'clojure.core/+ (list 'clojure.core/* 'blk B) 't)))
              (list 'aget 'b (list 'clojure.core/+ (list 'clojure.core/* 'j K)
                                   (list 'clojure.core/+ (list 'clojure.core/* 'blk B) 't))))
        :stages [{:axis 'blk :extent NB :dtype :float :init 0.0
                  :lift '(raster.numeric/* inner (aget da _) (aget db _))
                  :operands [{:sym 'da :map {:groups [[['i M]] [['blk NB]]]} :dtype :float}
                             {:sym 'db :map {:groups [[['j N]] [['blk NB]]]} :dtype :float}]}
                 {:axis 't :extent B :dtype :int :init 0}]))

(deftest staged-routes-to-the-multi-level-leaf
  (let [r (cr/route-contraction (staged-form :byte) :dtype :byte)]
    (testing "strategy + the two operand groups are kept distinct"
      (is (= :staged-segred (:strategy r)))
      (is (= '[a b] (:array-params r)) "the contraction's own operands")
      (is (= '[da db] (:lift-operands r)) "the per-stage scale arrays, bound separately"))
    (testing "dtypes: int8 operands, float output — the widening comes from the dtype pair"
      (is (= :byte (:dtype r)))
      (is (= :float (:out-dtype r))))
    (testing "launch geometry + count, on the default (non-reduction) invoke protocol"
      (is (= (* M N) (:out-elems r)))
      (is (= [256 1] (:wg r)))
      (is (= [{:type :int :value (* M N)}] (:scalar-args r))))
    (testing "the emitted kernel really nests one loop per stage"
      (is (= 2 (count (re-seq #"for \(int" (:source r)))))
      (is (re-find #"int acc_1 = 0;" (:source r)) "inner accumulator is int32")
      (is (re-find #"float acc_0 = 0.0;" (:source r)) "outer accumulator is float"))
    (testing "route-contraction validated it on the way out (it returned at all)"
      (is (some? (:kernel-name r))))))

(deftest staging-beats-dtype-in-the-dispatch
  (testing "an int8 staged contraction must NOT be captured by the flat int8 leaves — they have a
            single accumulator and cannot express a per-block scale at all"
    (is (= :staged-segred (:strategy (cr/route-contraction (staged-form :byte) :dtype :byte))))
    (is (= :staged-segred (:strategy (cr/route-contraction (staged-form :half) :dtype :half))))))

(deftest a-descriptor-that-forgets-the-scale-arrays-is-rejected
  (testing "dropping :lift-operands leaves the kernel with 2 unbound pointer params — the launch
            arity bug class, caught at generation instead of at launch"
    (let [r (cr/route-contraction (staged-form :byte) :dtype :byte)]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"pointer params"
           (cr/validate-descriptor (dissoc r :lift-operands))))
      (is (some? (cr/validate-descriptor r)) "…while the real descriptor validates"))))

(deftest single-stage-falls-through-to-the-flat-leaves
  (testing "one stage is the flat case; nothing about the existing routing changes"
    (let [form (list 'raster.par/contract 'out [['i M] ['j N]] [['l K]]
                     (list 'raster.numeric/*
                           (list 'aget 'a (list 'clojure.core/+ (list 'clojure.core/* 'i K) 'l))
                           (list 'aget 'b (list 'clojure.core/+ (list 'clojure.core/* 'l N) 'j)))
                     :stages [{:axis 'l :extent K :dtype :float :init 0.0}])]
      (is (not= :staged-segred (:strategy (cr/route-contraction form :dtype :double)))))))

;; ── the surface: par/contract with :stages must compute the flat EQUIVALENT ──────────
(deftest surface-contract-honours-stages
  (let [m 2 n 2 nb 2 b 4 k (* nb b)
        ;; double[] — par/contract's interpreted expansion writes its output with a double aset
        a (double-array (map #(double (mod % 5)) (range (* m k))))
        bb (double-array (map #(double (mod (* 3 %) 5)) (range (* n k))))
        ;; powers of two → the identity is exact in floating point, so it is asserted with =
        da (double-array (map #(Math/scalb 1.0 (int (- (mod % 3) 1))) (range (* m nb))))
        db (double-array (map #(Math/scalb 1.0 (int (- (mod (inc %) 3) 1))) (range (* n nb))))
        out (double-array (* m n))
        ;; reference: the block formula, scale applied once per block
        ref (vec (for [i (range m) j (range n)]
                   (double (reduce + (for [blk (range nb)]
                                      (* (double (aget da (+ (* i nb) blk)))
                                         (double (aget db (+ (* j nb) blk)))
                                         (reduce + (for [t (range b)]
                                                     (let [l (+ (* blk b) t)]
                                                       (* (double (aget a (+ (* i k) l)))
                                                          (double (aget bb (+ (* j k) l)))))))))))))]
    (par/contract out [[i 2] [j 2]] [[blk 2] [t 4]]
                  (raster.numeric/* (aget a (clojure.core/+ (clojure.core/* i 8)
                                                            (clojure.core/+ (clojure.core/* blk 4) t)))
                                    (aget bb (clojure.core/+ (clojure.core/* j 8)
                                                             (clojure.core/+ (clojure.core/* blk 4) t))))
                  :stages [{:axis blk :extent 2 :dtype :float :init 0.0
                            :lift (raster.numeric/* inner (aget da _) (aget db _))
                            :operands [{:sym da :map {:groups [[[i 2]] [[blk 2]]]}}
                                       {:sym db :map {:groups [[[j 2]] [[blk 2]]]}}]}
                           ;; NB the surface runs the FLAT equivalent, so the stage dtypes are a
                           ;; schedule annotation for the device leaf, not a cast applied here
                           {:axis t :extent 4 :dtype :double :init 0.0}])
    (testing "the surface applies the lift factors — dropping them would give plausible wrong numbers"
      (is (= ref (vec out))))
    (testing "…and the result is non-trivial"
      (is (some #(not (zero? %)) (vec out))))))

(deftest surface-rejects-illegal-stages
  (testing "an illegal stage list fails at macroexpansion, not at runtime"
    (is (thrown? Throwable
                 (eval '(let [out (float-array 4) a (float-array 32) bb (float-array 32)]
                          (raster.par/contract out [[i 2] [j 2]] [[blk 2] [t 4]]
                                               (raster.numeric/* (aget a t) (aget bb t))
                                               ;; lift discards `inner` → the partial sum is lost
                                               :stages [{:axis blk :extent 2 :dtype :float :init 0.0
                                                         :lift (raster.numeric/* 2.0 3.0)}
                                                        {:axis t :extent 4 :dtype :double :init 0.0}])))))))
