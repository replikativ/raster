(ns raster.compiler.ir.contraction-facts-test
  "FACTS: the single derivation whose only input is the form. Device-free.

   The property under test is structural, not behavioural: because there is one producer and its
   only input is the form, a checker cannot be handed a value derived from the thing it is meant
   to check. That shape is what made two silent miscompiles writable — a stage list validated
   against axes derived from itself, and a body-replacing leaf that never checked the body."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.contraction-facts :as cf]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.reduction :as reduction]))

(def ^:private ma (am/of-groups '[[[i 4]] [[blk 4] [t 32]]]))
(def ^:private mb (am/of-groups '[[[j 6]] [[blk 4] [t 32]]]))
(def ^:private body
  (list 'raster.numeric/* (list 'aget 'a (am/index-expr ma)) (list 'aget 'b (am/index-expr mb))))

(defn- form [& opts]
  (concat (list 'raster.par/contract 'out [['i 4] ['j 6]] [['blk 4] ['t 32]] body) opts))

(deftest facts-come-from-the-forms-own-slots
  (let [f (cf/contraction-facts (form) :dtype :byte)]
    (testing "tagged, so a hand-built map cannot be substituted for a derivation"
      (is (cf/facts? f))
      (is (not (cf/facts? {:contract-axes '[[t 4]]}))))
    (testing "axes come from the form's declaration slots"
      (is (= '[[i 4] [j 6]] (:free-axes f)))
      (is (= '[[blk 4] [t 32]] (:contract-axes f)))
      (is (= 2 (:n-free f)))
      (is (= 2 (:n-contract f))))
    (testing "roles let a leaf state requirements without naming the user's symbols"
      (is (= '{:free0 i :free1 j :contract0 blk :contract1 t} (:roles f)))
      (is (= {:free0 4 :free1 6 :contract0 4 :contract1 32} (:dims f))))
    (testing "operands are the body's aget reads"
      (is (= '[a b] (mapv :sym (:operands f)))))
    (testing "one canonical ProductReduction owns fold semantics before scheduling"
      (let [operator (:reduction f)]
        (is (reduction/product-reduction? operator))
        (is (= :byte (first (reduction/dtypes operator))))
        (is (= [(:index operator) 128] (:flat-contract-axis f)))
        (is (= (:element (cf/scalar-reduction-view f))
               (second (rest (first (:results (reduction/fold-region operator)))))))))))

(deftest a-stage-list-cannot-supply-the-axes-it-is-checked-against
  (testing "the form's contract axes are independent of its :stages — the two are separate slots,
            so a stage list that under-covers cannot make itself look complete"
    (let [f (cf/contraction-facts
             (form :stages [{:axis 'blk :extent 4 :dtype :float :init 0.0 :lift 'inner}
                            {:axis 't :extent 4 :dtype :int :init 0}])   ; 4, not 32
             :dtype :byte)]
      (is (= '[[blk 4] [t 32]] (:contract-axes f)) "from the form, not from the stages")
      (is (= [4 4] (mapv :extent (:stages f))) "the stage list is carried verbatim, unmerged")
      (is (not= (mapv second (:contract-axes f)) (mapv :extent (:stages f)))
          "…so the mismatch is VISIBLE to a checker instead of being defined away"))))

(deftest declarations-are-evidence-not-facts
  (testing "a declared :maps entry is admitted only after it provably generates the actual index"
    (let [f (cf/contraction-facts (form :maps {'a ma 'b mb}) :dtype :byte)]
      (is (every? some? (map :map (:operands f))))))
  (testing "a LYING declaration fails extraction — it does not quietly fall back to derivation"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"does not generate that operand's actual index"
         (cf/contraction-facts (form :maps {'a (am/of-groups '[[[i 4]] [[t 32] [blk 4]]])})
                               :dtype :byte))))
  (testing "an undeclared operand simply has no map — absence is not a lie"
    (let [f (cf/contraction-facts (form :maps {'a ma}) :dtype :byte)]
      (is (some? (:map (first (:operands f)))))
      (is (nil? (:map (second (:operands f))))))))

(deftest body-product-is-the-requirement-a-body-replacing-leaf-carries
  (testing "the exact product of the named operands"
    (is (= '[a b] (mapv :sym (cf/body-product-of body '[a b])))))
  (testing "an extra factor is NOT accounted for — this leaf would drop it silently"
    (is (nil? (cf/body-product-of (list 'raster.numeric/* body (list 'aget 'mask 't)) '[a b]))))
  (testing "…nor is a sum, nor a product of the wrong arity"
    (is (nil? (cf/body-product-of (list 'raster.numeric/+ body) '[a b])))
    (is (nil? (cf/body-product-of body '[a b c])))
    (is (nil? (cf/body-product-of body '[a]))))
  (testing "operator spelling does not matter (bare / clojure.core / raster.numeric)"
    (is (some? (cf/body-product-of (list '* (list 'aget 'a 'x) (list 'aget 'b 'y)) '[a b])))))

(deftest malformed-forms-are-refused
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not a par/contract form"
                        (cf/contraction-facts '(raster.par/map! out i 4 body))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"key/value pairs"
                        (cf/contraction-facts (concat (form) [:dangling]))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"free-axes must be a vector"
                        (cf/contraction-facts (list 'raster.par/contract 'out '(i 4) [['l 8]] body)))))

;; ── leaf layout requirements as data rows, not three hand-written predicates ─────────
(defn- mm-form [col-idx]
  (list 'raster.par/contract 'C [['i 4] ['j 6]] [['l 8]]
        (list 'raster.numeric/*
              (list 'aget 'A (list 'clojure.core/+ (list 'clojure.core/* 'i 8) 'l))
              (list 'aget 'B col-idx))))
(def ^:private nn-idx '(clojure.core/+ (clojure.core/* l 6) j))   ; B[l,j]
(def ^:private nt-idx '(clojure.core/+ (clojure.core/* j 8) l))   ; B[j,l] — K-contiguous

(deftest orientation-is-a-data-row
  (let [nn (cf/contraction-facts (mm-form nn-idx))
        nt (cf/contraction-facts (mm-form nt-idx))]
    (testing ":nn satisfies the dpas/quant row, not the dp4a row"
      (is (:ok (cf/check-layout nn (:dpas cf/leaf-layouts))))
      (is (:ok (cf/check-layout nn (:quant-naive cf/leaf-layouts))))
      (is (= :layout (:reason (cf/check-layout nn (:dp4a cf/leaf-layouts))))))
    (testing ":nt satisfies the dp4a row (both operands K-contiguous), not dpas"
      (is (:ok (cf/check-layout nt (:dp4a cf/leaf-layouts))))
      (is (= :layout (:reason (cf/check-layout nt (:dpas cf/leaf-layouts))))))
    (testing "the two rows differ ONLY in the col operand's axis order"
      (is (= [:free0 :contract0] (get-in cf/leaf-layouts [:dpas :row])
                                 (get-in cf/leaf-layouts [:dp4a :row])))
      (is (= [:contract0 :free1] (get-in cf/leaf-layouts [:dpas :col])))
      (is (= [:free1 :contract0] (get-in cf/leaf-layouts [:dp4a :col]))))))

(deftest role-assignment-is-by-verification-not-by-variance
  (testing "operands written in the OTHER order still bind correctly — an operand is the row
            operand because its index provably IS the row layout, not because of which axes it
            mentions (which is what the variance heuristic tested)"
    (let [swapped (cf/contraction-facts
                   (list 'raster.par/contract 'C [['i 4] ['j 6]] [['l 8]]
                         (list 'raster.numeric/*
                               (list 'aget 'B nn-idx)
                               (list 'aget 'A '(clojure.core/+ (clojure.core/* i 8) l)))))]
      (is (= '{:row A :col B} (:bindings (cf/check-layout swapped (:dpas cf/leaf-layouts)))))))
  (testing "a commuted index spelling binds too (one affine relation underneath)"
    (let [commuted (cf/contraction-facts (mm-form '(clojure.core/+ j (clojure.core/* l 6))))]
      (is (:ok (cf/check-layout commuted (:dpas cf/leaf-layouts))))))
  (testing "a wrong operand count is named, not guessed at"
    (let [three (cf/contraction-facts
                 (list 'raster.par/contract 'C [['i 4] ['j 6]] [['l 8]]
                       (list 'raster.numeric/* (list 'aget 'A 'l) (list 'aget 'B 'l)
                                               (list 'aget 'D 'l))))]
      (is (= :operand-count (:reason (cf/check-layout three (:dpas cf/leaf-layouts)))))))
  (testing "check-layout refuses a hand-built map — it requires real facts"
    (is (thrown? clojure.lang.ExceptionInfo
                 (cf/check-layout {:operands [] :roles {}} (:dpas cf/leaf-layouts))))))

;; ── anti-drift: the data table and the predicate it will replace must AGREE ──────────
;; `check-layout` is not yet consumed by the gates. That is deliberate — four of the six
;; orientation call sites live in the quant and dp4a generators, which the next increment deletes
;; outright, and wiring only the fifth (dpas) would create a dual path. But an unconsumed
;; capability is how this subsystem drifted in the first place, so the equivalence is PINNED here:
;; if the table and the live predicate ever disagree on a case, this fails.
(deftest table-agrees-with-the-live-orientation-gate
  (let [cases [[:nn nn-idx :dpas true] [:nn nn-idx :dp4a false]
               [:nt nt-idx :dp4a true] [:nt nt-idx :dpas false]]
        gate (requiring-resolve 'raster.compiler.backend.gpu.segop-opencl/dpas-contraction-legal?)
        lower (requiring-resolve 'raster.compiler.passes.parallel.contract-lower/contract-form->segred)]
    (doseq [[label idx leaf expected] cases]
      (testing (str label " under " leaf)
        (is (= expected (boolean (:ok (cf/check-layout (cf/contraction-facts (mm-form idx))
                                                       (get cf/leaf-layouts leaf)))))
            "table verdict")))
    (testing "and the live DPAS gate reaches the same orientation verdict the table does"
      ;; f64 so the gate's own dtype requirement does not mask the orientation decision
      (let [nn-sr (lower (mm-form nn-idx) :dtype :half)
            nt-sr (lower (mm-form nt-idx) :dtype :half)]
        (is (not= :non-canonical-orientation (:reason (gate nn-sr :half)))
            ":nn is orientation-legal for dpas (it may still fail on pitch alignment)")
        (is (= :non-canonical-orientation (:reason (gate nt-sr :half)))
            ":nt is orientation-ILLEGAL for dpas — same verdict the table gives")))))
