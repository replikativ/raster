(ns raster.compiler.passes.scalar.soa-lower-test
  "Track A — value-type SoA scalar-replacement (Option 2 / A′ shared lowering).
   Validates the SROA pass both structurally (lowered IR has no value-type ops)
   and end-to-end (a CxSoA kernel compiles to wasm and runs via Chicory)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm defvalue]]
            [raster.numeric]
            [raster.arrays]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.pipeline :as pl]
            [raster.compiler.passes.scalar.soa-lower :as sl])
  (:import [com.dylibso.chicory.wasm Parser]
           [com.dylibso.chicory.runtime Instance]))

(defvalue Cplx [re :- Double, im :- Double])

(defvalue ArrayBundle [positions :- (Array float), labels :- (Array int)])

(deftest value-factories-retain-their-result-tag
  (is (= 'Cplx (:raster.core/return-tag (meta #'->Cplx))))
  (is (= 'Cplx
         (inference/infer-arg-tag
          '(raster.compiler.passes.scalar.soa-lower-test/->Cplx a b)
          {'a 'double 'b 'double}))))

(deftest local-constructor-projection-is-escape-aware
  (let [projected '(let* [a 1.0 b 2.0 p (->Cplx a b)]
                     (clojure.core/+ (.re p) (.im p)))
        escaping '(let* [a 1.0 b 2.0 p (->Cplx a b)] p)
        effectful '(let* [b 2.0 p (->Cplx (println "effect") b)] (.re p))
        pure-casts '(let* [p (->Cplx (clojure.core/double a)
                                      (clojure.core/double b))]
                      (clojure.core/+ (.re p) (.im p)))
        shadowed '(let* [a 1.0 p (->Cplx a a)]
                    (let* [p 3.0] (clojure.core/+ p 1.0)))
        quoted '(let* [a 1.0 p (->Cplx a a)]
                  (clojure.core/list (.re p) (quote p)))
        nested-fn '(let* [a 1.0 p (->Cplx a a)]
                     (fn* [p] (clojure.core/+ p 1.0)))]
    (is (= '(let* [a 1.0 b 2.0] (clojure.core/+ a b))
           (sl/lower-local-constructors projected)))
    (is (= escaping (sl/lower-local-constructors escaping)))
    (is (= effectful (sl/lower-local-constructors effectful)))
    (is (= '(let* []
              (clojure.core/+ (clojure.core/double a)
                              (clojure.core/double b)))
           (sl/lower-local-constructors pure-casts)))
    (is (= '(let* [a 1.0] (let* [p 3.0] (clojure.core/+ p 1.0)))
           (sl/lower-local-constructors shadowed)))
    (is (= '(let* [a 1.0]
              (clojure.core/list a (quote p)))
           (sl/lower-local-constructors quoted)))
    (is (= nested-fn (sl/lower-local-constructors nested-fn)))))

(deftm cadd-soa! [as :- CplxSoA, bs :- CplxSoA, os :- CplxSoA, n :- Long] :- nil
  (dotimes [i n]
    (aset-cplx! os i (->Cplx (raster.numeric/+ (.re (aget-cplx as i)) (.re (aget-cplx bs i)))
                             (raster.numeric/+ (.im (aget-cplx as i)) (.im (aget-cplx bs i)))))))

(deftest soa-lower-eliminates-value-type-ops
  (testing "expand-params turns one SoA param into per-field array params"
    (let [specs [{:sym 'as :tag 'CplxSoA} {:sym 'n :tag 'long}]
          env (sl/soa-param-env specs)
          expanded (sl/expand-params specs env)]
      (is (= '[as_re as_im n] (mapv :sym expanded)))
      (is (= '[doubles doubles long] (mapv :tag expanded)))
      (is (= {'as_re {:binding 'as :field :re}
              'as_im {:binding 'as :field :im}}
             (sl/buffer-projections env))))))

(deftest array-bundle-lowering-uses-the-same-physical-product-boundary
  (let [specs [{:sym 'state :tag 'ArrayBundle} {:sym 'n :tag 'long}]
        env (sl/soa-param-env specs)
        lowered (sl/soa-lower
                 '(let* [p (.positions state) l (.labels state)]
                    (clojure.core/aset p 0 (float (clojure.core/aget l 0))))
                 specs)]
    (is (= '[state_positions state_labels n] (mapv :sym (:params lowered))))
    (is (= '[floats ints long] (mapv :tag (:params lowered))))
    (is (= {'state_positions {:binding 'state :field :positions}
            'state_labels {:binding 'state :field :labels}}
           (sl/buffer-projections env)))
    (is (= '(let* [p state_positions l state_labels]
              (clojure.core/aset p 0 (float (clojure.core/aget l 0))))
           (:body lowered)))))

(deftm get-re-soa! [ps :- CplxSoA, out :- (Array double), n :- Long] :- nil
  (dotimes [i n]
    (let [p (aget-cplx ps i)]                ; value-type LOCAL bound to a SoA read
      (raster.arrays/aset out i (.re p)))))   ; projected → (aget ps_re i)

(defn- instantiate [^bytes wasm] (-> (Instance/builder (Parser/parse wasm)) (.build)))

(deftest cadd-soa-compiles-and-runs
  (testing "CxSoA elementwise add: SoA params → per-field arrays, value ops scalar-replaced"
    (let [m (pl/compile-wasm #'cadd-soa! :name "cadd")]
      ;; 3 SoA params × 2 fields + n  → 7 i32 params, void result
      (is (= 7 (count (:param-types m))))
      (is (= [] (:result-types m)))
      (let [inst (instantiate (:bytes m))
            mem  (.memory inst)
            n    64]
        ;; layout: as_re@0 as_im@n bs_re@2n bs_im@3n os_re@4n os_im@5n (n doubles each)
        (dotimes [i n]
          (.writeF64 mem (* 8 i) (double i))
          (.writeF64 mem (* 8 (+ n i)) (double (* 2 i)))
          (.writeF64 mem (* 8 (+ (* 2 n) i)) 10.0)
          (.writeF64 mem (* 8 (+ (* 3 n) i)) 100.0))
        (.apply (.export inst "cadd")
                (long-array [0 (* 8 n) (* 8 (* 2 n)) (* 8 (* 3 n)) (* 8 (* 4 n)) (* 8 (* 5 n)) n]))
        (let [bad (count (filter (fn [i]
                                   (or (not= (.readDouble mem (* 8 (+ (* 4 n) i))) (+ (double i) 10.0))
                                       (not= (.readDouble mem (* 8 (+ (* 5 n) i))) (+ (double (* 2 i)) 100.0))))
                                 (range n)))]
          (is (zero? bad) (str bad " complex-add mismatches")))))))

(deftest value-type-local-projection
  (testing "let-bound value-type local: (let [p (aget-cplx ps i)] … (.re p)) → (aget ps_re i)"
    (let [m (pl/compile-wasm #'get-re-soa! :name "getre")
          inst (instantiate (:bytes m))
          mem  (.memory inst)
          n    32]
      ;; layout: ps_re@0 ps_im@n out@2n
      (dotimes [i n] (.writeF64 mem (* 8 i) (double (* 3 i))) (.writeF64 mem (* 8 (+ n i)) 0.0))
      (.apply (.export inst "getre") (long-array [0 (* 8 n) (* 8 (* 2 n)) n]))
      (let [bad (count (filter (fn [i] (not= (.readDouble mem (* 8 (+ (* 2 n) i))) (double (* 3 i))))
                               (range n)))]
        (is (zero? bad) (str bad " get-re mismatches"))))))

;; value-type HELPER deftm: returns a bare (->Cplx …) constructor. The inliner
;; must inline it (inlinable-body? accepts value-type ctor tails) so soa-lower
;; can scalar-replace the result; soa-lower floats the inlined let*-of-args out
;; of the store and preserves .invk op metadata; the emitter handles the void
;; dotimes binding + let*-in-effect-position the inliner/float produce.
(deftm cadd-el [a :- Cplx, b :- Cplx] :- Cplx
  (->Cplx (raster.numeric/+ (.re a) (.re b)) (raster.numeric/+ (.im a) (.im b))))

(deftm cadd-via-helper! [as :- CplxSoA, bs :- CplxSoA, os :- CplxSoA, n :- Long] :- nil
  (dotimes [i n]
    (aset-cplx! os i (cadd-el (aget-cplx as i) (aget-cplx bs i)))))

(deftest value-type-helper-deftm-inlines
  (testing "value-type helper deftm inlines + lowers + runs (Cplx add via cadd-el)"
    (let [m (pl/compile-wasm #'cadd-via-helper! :name "caddh")]
      (is (= 7 (count (:param-types m))))   ; 3 SoA × 2 fields + n
      (is (= [] (:result-types m)))
      (let [inst (instantiate (:bytes m))
            mem  (.memory inst)
            n    64]
        ;; layout: as_re@0 as_im@n bs_re@2n bs_im@3n os_re@4n os_im@5n
        (dotimes [i n]
          (.writeF64 mem (* 8 i) (double i))
          (.writeF64 mem (* 8 (+ n i)) (double (* 2 i)))
          (.writeF64 mem (* 8 (+ (* 2 n) i)) 10.0)
          (.writeF64 mem (* 8 (+ (* 3 n) i)) 100.0))
        (.apply (.export inst "caddh")
                (long-array [0 (* 8 n) (* 8 (* 2 n)) (* 8 (* 3 n)) (* 8 (* 4 n)) (* 8 (* 5 n)) n]))
        (let [bad (count (filter (fn [i]
                                   (or (not= (.readDouble mem (* 8 (+ (* 4 n) i))) (+ (double i) 10.0))
                                       (not= (.readDouble mem (* 8 (+ (* 5 n) i))) (+ (double (* 2 i)) 100.0))))
                                 (range n)))]
          (is (zero? bad) (str bad " helper-add mismatches")))))))
