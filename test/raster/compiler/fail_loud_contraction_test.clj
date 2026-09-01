(ns raster.compiler.fail-loud-contraction-test
  "Silently-wrong paths, converted to loud ones. Device-free.

   Each case here produced a plausible wrong ANSWER rather than an error. That is the worst failure
   mode a compiler has, and every one of these was invisible to the suite — three were found by
   audit, not by a test."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-route :as cr]
            [raster.compiler.backend.gpu.segop-opencl :as sco]
            [raster.compiler.ir.axis-map :as am]))

(defn- staged-form [& {:keys [combine extent] :or {extent 32}}]
  (concat (list 'raster.par/contract 'out [['i 2] ['j 2]] [['blk 2] ['t extent]]
                (list 'raster.numeric/* (list 'aget 'a 't) (list 'aget 'b 't))
                :stages [{:axis 'blk :extent 2 :dtype :float :init 0.0 :lift 'inner}
                         {:axis 't :extent extent :dtype :int :init 0}])
          (when combine [:combine combine :init Byte/MIN_VALUE])))

(deftest staged-refuses-a-combine-it-would-silently-turn-into-a-sum
  (testing "every accumulator level uses `+=` and a lift's linearity argument assumes addition, so
            a form carrying :combine max was SILENTLY SUMMED — contraction-facts surfaced :combine
            and nothing read it"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"only `\+` combine"
                          (cr/route-contraction (staged-form :combine 'max) :dtype :byte)))
    (is (some? (cr/route-contraction (staged-form) :dtype :byte)) "…and `+` still routes")))

(deftest staged-refuses-symbolic-bounds-it-cannot-declare
  (testing "this emitter interpolates extents but declares no scalar params for them, so a symbolic
            bound emitted an UNDECLARED identifier — and validate-descriptor passed it (the counts
            agree), so it failed only at device build, i.e. never in CI"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"symbolic bounds are not supported"
         (sco/generate-staged-contraction-kernel
          {:free-axes '[[i m] [j 2]] :contract-axes '[[l 8]]
           :stages [{:axis 'l :extent 8 :dtype :float :init 0.0}]
           :body '(raster.numeric/* (aget a l) (aget b l))
           :inputs '[a b] :dtype :float :out-dtype :float}
          'out)))))

(deftest staged-emits-the-extension-pragma-its-dtype-needs
  (testing "all three sibling emitters emit this; the staged one did not, so a :double staged
            contraction — reachable from opencl_pass, whose default dtype IS :double — emitted
            `double` with no #pragma"
    (let [src (:source (sco/generate-staged-contraction-kernel
                        {:free-axes '[[i 2] [j 2]] :contract-axes '[[l 8]]
                         :stages [{:axis 'l :extent 8 :dtype :double :init 0.0}]
                         :body '(raster.numeric/* (aget a l) (aget b l))
                         :inputs '[a b] :dtype :double :out-dtype :double}
                        'out))]
      (is (re-find #"cl_khr_fp64" src)))))

(deftest the-portable-leaf-keeps-a-multidimensional-result-transform
  (testing "a three-free-axis contraction carries its result transform through typed scalar SSA"
    (let [ep {:acc 'acc :expr '(raster.numeric/* acc 2.0)}
          form (concat (list 'raster.par/contract 'out [['b 2] ['i 2] ['j 2]] [['l 4]]
                             (list 'raster.numeric/* (list 'aget 'a 'l) (list 'aget 'c 'l)))
                       [:epilogue ep])
          routed (cr/route-contraction form :dtype :double)]
      (is (= :portable-segred (:strategy routed)))
      (is (true? (:fused-epilogue routed)))
      (is (re-find #"\* 2\.0" (:source routed))))))

(deftest the-decompose-is-row-major-and-distinguishably-so
  ;; MUTATION-PROVEN INADEQUATE, and this is the replacement. The previous version asserted
  ;;     (re-find #"\(seg / \(2\)\) % 2" src)     with free axes [[i 2] [j 2]]
  ;; Both extents were 2, so row-major and column-major emit the SAME two lines with the axes
  ;; swapped — a one-line regex over a symmetric shape cannot tell them apart. Flipping
  ;; flat-decompose-c to column-major left 101 CI-visible contraction assertions passing.
  ;;
  ;; Fix: ASYMMETRIC extents, and assert each axis's own line, so the two orderings differ.
  (let [src (:source (sco/generate-staged-contraction-kernel
                      {:free-axes '[[i 3] [j 5]] :contract-axes '[[l 8]]
                       :stages [{:axis 'l :extent 8 :dtype :float :init 0.0}]
                       :body '(raster.numeric/* (aget a l) (aget b l))
                       :inputs '[a b] :dtype :float :out-dtype :float}
                      'out))]
    (testing "outer axis divides by the product of the extents INSIDE it, then mods by its own"
      (is (re-find #"int i = \(seg / \(5\)\) % 3;" src)))
    (testing "innermost axis is a bare mod — no division"
      (is (re-find #"int j = seg % 5;" src)))
    (testing "and the column-major spelling is ABSENT (this is what the old test could not say)"
      (is (not (re-find #"int i = seg % 3;" src)))
      (is (not (re-find #"int j = \(seg / \(3\)\) % 5;" src))))))

(deftest symbolic-bounds-are-refused-not-mangled
  ;; The bug the old test NAMED was a symbolic bound (`n-cols` emitted raw as `(seg / (n-cols))`,
  ;; valid C computing `n - cols`) — but its fixture passed only literals, so it exercised nothing.
  ;; Symbolic bounds are now refused outright by this emitter, so assert THAT.
  (testing "a symbolic free-axis bound is refused rather than emitted undeclared"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"symbolic bounds are not supported"
         (sco/generate-staged-contraction-kernel
          {:free-axes '[[i n-cols] [j 5]] :contract-axes '[[l 8]]
           :stages [{:axis 'l :extent 8 :dtype :float :init 0.0}]
           :body '(raster.numeric/* (aget a l) (aget b l))
           :inputs '[a b] :dtype :float :out-dtype :float}
          'out)))))

;; ── producer-side assertions: a widened channel is only a fix if something fills it ──────
(deftest full-reduce-descriptor-carries-its-combine
  ;; A previous commit "fixed" the silent-sum by widening the REGISTRATION to pass the whole
  ;; descriptor through. That was a no-op: the descriptor did not contain :c-op/:identity-val —
  ;; generate-segred-kernel returned them into a local that route-contraction discarded. The
  ;; consumer-side assertion could not catch it; only this producer-side one can.
  (testing ":combine max reaches the descriptor as fmax + its identity, not defaulted to +/0.0"
    (let [r (cr/route-contraction
             (list 'raster.par/contract 'O [] [['l 8]]
                   (list 'raster.numeric/* (list 'aget 'a 'l) (list 'aget 'b 'l))
                   :combine 'max :init 'Double/NEGATIVE_INFINITY)
             :dtype :double)]
      (is (= :full-reduce (:strategy r)))
      (is (= "fmax" (:c-op r)))
      (is (= Double/NEGATIVE_INFINITY (:identity-val r))
          "the identity must be the op's, not 0.0 — summing max-partials from 0.0 is wrong for
           all-negative data"))))

(deftest fp64-result-transform-uses-the-typed-register-tiled-store
  (testing "an f64 two-free/one-contract result transform executes on the register-tiled leaf"
    (let [form (list 'raster.par/contract 'C [['i 4] ['j 4]] [['l 8]]
                     (list 'raster.numeric/*
                           (list 'aget 'A (list 'clojure.core/+ (list 'clojure.core/* 'i 8) 'l))
                           (list 'aget 'B (list 'clojure.core/+ (list 'clojure.core/* 'l 4) 'j)))
                     :epilogue {:acc 'acc :expr '(raster.numeric/* acc 2.0)})
          routed (cr/route-contraction form :dtype :double)]
      (is (= :regtiled (:strategy routed)))
      (is (true? (:fused-epilogue routed)))
      (is (re-find #"\* 2\.0" (:source routed))))))

(deftest dp4a-refuses-a-decode-it-would-discard
  ;; The dp4a leaf replaces the whole summand with one hardware op, so a per-operand :decode — the
  ;; load-lambda where a zero-point lives — would be silently dropped: Σ a·b instead of
  ;; Σ(a−za)(b−zb). Load-bearing, since q4_0/q8_0 zero-points are 8 and 128.
  (let [am (requiring-resolve 'raster.compiler.ir.axis-map/of-groups)
        idx (requiring-resolve 'raster.compiler.ir.axis-map/index-expr)
        gate (requiring-resolve 'raster.compiler.backend.gpu.segop-opencl/staged-inner-dp4a-legal?)
        ma (am [['[i 4]] ['[blk 4] '[t 32]]])
        mb (am [['[j 4]] ['[blk 4] '[t 32]]])
        base {:stages [{:axis 'blk :extent 4 :dtype :float :init 0.0 :lift 'inner}
                       {:axis 't :extent 32 :dtype :int :init 0}]
              :dtype :byte
              :body (list 'raster.numeric/* (list 'aget 'a (idx ma)) (list 'aget 'b (idx mb)))}]
    (testing "no decode → the peak leaf is legal"
      (is (:ok (gate (assoc base :operands [{:sym 'a :map ma} {:sym 'b :map mb}])))))
    (testing "a zero-point on either operand → refused by name, not silently dropped"
      (is (= :decode-on-a-body-replacing-leaf
             (:reason (gate (assoc base :operands [{:sym 'a :map ma :decode '(clojure.core/- x 8)}
                                                   {:sym 'b :map mb}])))))
      (is (= :decode-on-a-body-replacing-leaf
             (:reason (gate (assoc base :operands [{:sym 'a :map ma}
                                                   {:sym 'b :map mb :decode '(clojure.core/- x 128)}]))))))))
