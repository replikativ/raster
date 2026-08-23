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
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.passes.parallel.par-fusion :as pf]))

(defn- mm [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(defn- fused [mbody]
  (let [b ['C (mm 128 256 64) 'out (list 'raster.par/map! 'out 't (* 128 256) nil mbody)]]
    (second (:bindings (pf/fuse-contract-map b [])))))

;; ── every strategy's descriptor matches the kernel it describes ───────────────────────
(deftest every-strategy-descriptor-is-consistent
  ;; route-contraction validates on the way out, so simply routing each shape is the assertion.
  (testing "the six base strategies"
    (is (= :dpas         (:strategy (route/route-contraction (mm 256 512 128) :dtype :half))))
    (is (= :regtiled     (:strategy (route/route-contraction (mm 96 96 64) :dtype :double))))
    (is (= :segmap       (:strategy (route/route-contraction
                                     '(raster.par/contract C [[i 4] [j 3]] [] (* (aget a i) (aget b j)))
                                     :dtype :double))))
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
    (doseq [[label body] [[:activation (list 'silu_f (list 'aget 'C 't))]
                          [:bias  (list 'raster.numeric/+ (list 'aget 'C 't)
                                        (list 'aget 'bias (list 'mod 't 256)))]
                          [:resid (list 'raster.numeric/+ (list 'aget 'C 't) (list 'aget 'R 't))]
                          [:rowscale (list 'raster.numeric/* (list 'aget 'C 't)
                                           (list 'aget 'rs (list 'quot 't 256)))]]]
      (is (= :dpas (:strategy (route/route-contraction (fused body) :dtype :half)))
          (str label " descriptor must validate")))))

;; ── the validator rejects each historical bug shape ──────────────────────────────────
(deftest validator-rejects-the-bugs-that-actually-happened
  (let [good (route/route-contraction (fused (list 'raster.numeric/+ (list 'aget 'C 't)
                                                   (list 'aget 'bias (list 'mod 't 256))))
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
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"2-element vectors"
            (route/validate-descriptor (assoc good :grid [4])))))))

(deftest validator-models-both-invoke-protocols
  (let [red (route/route-contraction '(raster.par/contract O [] [[i 8]] (* (aget A i) (aget B i)))
                                     :dtype :double)]
    (testing ":invoke :reduction owns its launch — no :wg/:grid, empty :scalar-args, needs :reduce-bound"
      (is (= :reduction (:invoke red)))
      (is (nil? (:wg red)))
      (is (nil? (:grid red)))
      (is (empty? (:scalar-args red)))
      (is (some? (:reduce-bound red))))
    (testing "…and the validator enforces each of those, so the protocol cannot drift"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must not carry"
            (route/validate-descriptor (assoc red :wg [256 1]))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must leave :scalar-args empty"
            (route/validate-descriptor (assoc red :scalar-args [{:type :int :value 8}]))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires :reduce-bound"
            (route/validate-descriptor (dissoc red :reduce-bound)))))))

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
