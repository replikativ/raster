(ns raster.compiler.ir.contraction-facts-test
  "FACTS: the single derivation whose only input is the form. Device-free.

   The property under test is structural, not behavioural: because there is one producer and its
   only input is the form, a checker cannot be handed a value derived from the thing it is meant
   to check. That shape is what made two silent miscompiles writable — a stage list validated
   against axes derived from itself, and a body-replacing leaf that never checked the body."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.contraction-facts :as cf]
            [raster.compiler.ir.axis-map :as am]))

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
      (is (= '[a b] (mapv :sym (:operands f)))))))

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
