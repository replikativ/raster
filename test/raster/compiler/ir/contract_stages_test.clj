(ns raster.compiler.ir.contract-stages-test
  "Staged contract axes: legality + the flat-equivalence identity. Device-free — this is IR.

   The identity is the load-bearing claim: staging is a SCHEDULE, so a staged contraction must
   have a flat form equal to it in exact arithmetic. Everything else (the GPU leaf, the CPU path,
   the device oracle) rests on that, so it is tested here directly rather than only via a kernel."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.contract-stages :as cs]
            [raster.compiler.ir.axis-map :as am]))

;; q8_0-shaped: 2 stages, per-block scale on each operand
(def ^:private q8-stages
  '[{:axis blk :extent 4 :dtype :float :init 0.0
      :lift (raster.numeric/* inner (aget da blk) (aget db blk))
      :operands [{:sym da :map {:groups [[[blk 4]]]}}
                 {:sym db :map {:groups [[[blk 4]]]}}]}
    {:axis t :extent 32 :dtype :int :init 0}])

(def ^:private body '(raster.numeric/* (aget a idx-a) (aget b idx-b)))

(deftest single-stage-is-todays-flat-contraction
  (testing "one stage is legal, needs no lift, and its flat equivalent is the body verbatim"
    (let [s '[{:axis l :extent 128 :dtype :float :init 0.0}]]
      (is (:ok (cs/stages-legal? s '[[l 128]])))
      (is (= body (cs/flat-equivalent s body)))
      (is (= 128 (cs/contract-extent s)))
      (is (= [] (cs/lift-operands s))))))

(deftest two-stage-block-quant
  (testing "the q8_0 shape is legal"
    (is (:ok (cs/stages-legal? q8-stages '[[blk 4] [t 32]]))))
  (testing "extent is the product of the stage extents"
    (is (= 128 (cs/contract-extent q8-stages))))
  (testing "flat equivalent multiplies the lift factors into the body — this IS the semantics"
    (is (= '(raster.numeric/* (raster.numeric/* (aget a idx-a) (aget b idx-b))
                              (aget da blk) (aget db blk))
           (cs/flat-equivalent q8-stages body))
        "operand indices come from the declared maps, so the result is self-contained"))
  (testing "scale arrays are surfaced as operands so a launch descriptor binds them"
    (is (= '[da db] (mapv :sym (cs/lift-operands q8-stages)))))
  (testing "operand indices come from the DECLARED axis-map, not from inferred strides"
    (is (= '{da blk db blk} (cs/stage-index-exprs q8-stages)))))

(deftest three-stage-super-block
  (testing "k-quant nesting is just one more stage — no new concept"
    (let [s '[{:axis sb :extent 2 :dtype :float :init 0.0
               :lift (raster.numeric/* inner (aget dsuper sb))
               :operands [{:sym dsuper :map {:groups [[[sb 2]]]}}]}
              {:axis blk :extent 8 :dtype :float :init 0.0
               :lift (raster.numeric/* inner (aget dsub blk))
               :operands [{:sym dsub :map {:groups [[[blk 8]]]}}]}
              {:axis t :extent 32 :dtype :int :init 0}]]
      (is (:ok (cs/stages-legal? s '[[sb 2] [blk 8] [t 32]])))
      (is (= 512 (cs/contract-extent s)))
      (is (= '[dsuper dsub] (mapv :sym (cs/lift-operands s))))
      (is (= '(raster.numeric/* (raster.numeric/* (aget a idx-a) (aget b idx-b))
                                (aget dsuper sb) (aget dsub blk))
             (cs/flat-equivalent s body))))))

(deftest linearity-is-the-schedule-condition
  (testing "linear-in-inner returns the remaining factors"
    (is (= [] (cs/linear-in-inner 'inner 'inner)))
    (is (= '[(aget da blk)] (cs/linear-in-inner '(raster.numeric/* inner (aget da blk)) 'inner)))
    (is (= '[(aget da blk)] (cs/linear-in-inner '(raster.numeric/* (aget da blk) inner) 'inner))
        "factor order does not matter"))
  (testing "non-linear lifts have NO flat form and are rejected, not mis-scheduled"
    (is (nil? (cs/linear-in-inner '(raster.numeric/sqrt inner) 'inner)))
    (is (nil? (cs/linear-in-inner '(raster.numeric/+ inner 1.0) 'inner))
        "additive is not linear-with-a-factor: Σ(x+1) ≠ (Σx)+1 across stages")
    (is (= :lift-not-linear-in-inner
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 4 :dtype :float :init 0.0
                        :lift (raster.numeric/sqrt inner)}
                       {:axis t :extent 32 :dtype :int :init 0}]
                     '[[blk 4] [t 32]]))))))

(deftest rejections
  (testing "stages must cover the contract axes exactly, outer→inner"
    (is (= :stages-do-not-match-contract-axes
           (:reason (cs/stages-legal? q8-stages '[[t 32] [blk 4]])) ) "wrong order")
    (is (= :stages-do-not-match-contract-axes
           (:reason (cs/stages-legal? q8-stages '[[blk 4]]))) "missing axis"))
  (testing "a lift that DISCARDS the inner accumulator is silently wrong — refuse it"
    (is (= :lift-discards-inner-accumulator
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 4 :dtype :float :init 0.0
                        :lift (raster.numeric/* (aget da blk) 2.0)}
                       {:axis t :extent 32 :dtype :int :init 0}]
                     '[[blk 4] [t 32]])))))
  (testing "…and one that uses it twice duplicates the whole inner reduction"
    (is (= :inner-accumulator-used-more-than-once
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 4 :dtype :float :init 0.0
                        :lift (raster.numeric/* inner inner)}
                       {:axis t :extent 32 :dtype :int :init 0}]
                     '[[blk 4] [t 32]])))))
  (testing "the innermost stage accumulates the body, so it must not carry a lift"
    (is (= :innermost-stage-has-a-lift
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 4 :dtype :float :init 0.0
                        :lift (raster.numeric/* inner (aget da blk))}
                       {:axis t :extent 32 :dtype :int :init 0
                        :lift (raster.numeric/* inner 2.0)}]
                     '[[blk 4] [t 32]])))))
  (testing "an outer stage with no lift at all"
    (is (= :outer-stage-without-a-lift
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 4 :dtype :float :init 0.0}
                       {:axis t :extent 32 :dtype :int :init 0}]
                     '[[blk 4] [t 32]])))))
  (testing "an integral OUTER accumulator over a float inner one truncates every partial sum"
    (is (= :narrowing-stage-accumulator
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 4 :dtype :int :init 0
                        :lift (raster.numeric/* inner (aget da blk))}
                       {:axis t :extent 32 :dtype :float :init 0.0}]
                     '[[blk 4] [t 32]]))))
    (is (:ok (cs/stages-legal? q8-stages '[[blk 4] [t 32]]))
        "int inner → float outer is the quant case and must stay legal"))
  (testing "a reduction inside a lift is a re-tiling decision we do not have"
    (is (= :layout-changing-op-in-lift
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 4 :dtype :float :init 0.0
                        :lift (raster.numeric/* inner (raster.par/reduce (aget da blk)))}
                       {:axis t :extent 32 :dtype :int :init 0}]
                     '[[blk 4] [t 32]])))))
  (testing "no stages at all"
    (is (= :no-stages (:reason (cs/stages-legal? [] []))))))

(deftest symbolic-extents
  (testing "extents may be symbolic; the product stays symbolic"
    (let [s '[{:axis blk :extent nb :dtype :float :init 0.0
               :lift (raster.numeric/* inner (aget da blk))
               :operands [{:sym da :map {:groups [[[blk nb]]]}}]}
              {:axis t :extent 32 :dtype :int :init 0}]]
      (is (:ok (cs/stages-legal? s '[[blk nb] [t 32]])))
      (is (= '(clojure.core/* 32 nb) (cs/contract-extent s))))))

;; ── the oracle: a stage list must SPAN the axes the form declared ────────────────────
(deftest stages-must-span-the-declared-contract-space
  (testing "axis names matching is not enough — extents must span the declared space.
            A stage of extent 4 against a declared extent of 32 emitted a kernel summing an
            EIGHTH of the terms while the interpreted path summed all of them: two consumers of
            one form disagreeing silently."
    (is (= :stage-extents-do-not-span-the-contract-axes
           (:reason (cs/stages-legal?
                     '[{:axis blk :extent 2 :dtype :float :init 0.0 :lift inner}
                       {:axis t :extent 4 :dtype :int :init 0}]
                     '[[blk 2] [t 32]]))))
    (is (:ok (cs/stages-legal?
              '[{:axis blk :extent 2 :dtype :float :init 0.0 :lift inner}
                {:axis t :extent 32 :dtype :int :init 0}]
              '[[blk 2] [t 32]]))
        "…and the spanning list is legal"))
  (testing "the failure names both sides, so the diagnostic is actionable"
    (let [r (cs/stages-legal? '[{:axis l :extent 8 :dtype :float :init 0.0}] '[[l 64]])]
      (is (= 8 (:spans r)))
      (is (= 64 (:declared r))))))

;; hoisted OUT of a GPU-gated device test: this is the only assertion anywhere that pins
;; multi-axis map substitution, and behind a GPU gate it never ran in CI. A mutation making
;; stage-index-exprs always return the bare axis survived every CI-visible assertion.
(deftest flat-equivalent-substitutes-multi-axis-operand-maps
  (testing "an operand indexed by a FREE axis and a STAGE axis gets its declared 2-D index,
            not the bare stage axis"
    (let [stages '[{:axis blk :extent 4 :dtype :float :init 0.0
                    :lift (raster.numeric/* inner (aget da _) (aget db _))
                    :operands [{:sym da :map {:groups [[[i 4]] [[blk 4]]]}}
                               {:sym db :map {:groups [[[j 6]] [[blk 4]]]}}]}
                   {:axis t :extent 32 :dtype :int :init 0}]
          b '(raster.numeric/* (aget a ia) (aget b ib))]
      (is (= '(raster.numeric/* (raster.numeric/* (aget a ia) (aget b ib))
                                (aget da (clojure.core/+ (clojure.core/* i 4) blk))
                                (aget db (clojure.core/+ (clojure.core/* j 4) blk)))
             (cs/flat-equivalent stages b)))
      (is (= '{da (clojure.core/+ (clojure.core/* i 4) blk)
               db (clojure.core/+ (clojure.core/* j 4) blk)}
             (cs/stage-index-exprs stages))))))
