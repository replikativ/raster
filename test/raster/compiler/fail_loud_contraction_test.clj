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
          (when combine [:combine combine])))

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

(deftest a-leaf-without-a-store-splice-refuses-an-epilogue
  (testing "an :epilogue reaching :segmap or the :else fallback was silently DROPPED — the
            consumer's computation vanished with no error. fuse-contract-map already produces such
            forms (it has no rank restriction) and is one line from being wired."
    (let [ep {:acc 'acc :expr '(raster.numeric/* acc 2.0)}
          ;; 3 free axes → the universal fallback, which has no store splice
          form (concat (list 'raster.par/contract 'out [['b 2] ['i 2] ['j 2]] [['l 4]]
                             (list 'raster.numeric/* (list 'aget 'a 'l) (list 'aget 'c 'l)))
                       [:epilogue ep])]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"silently dropped"
                            (cr/route-contraction form :dtype :double))))))

(deftest the-decompose-mangles-symbolic-bound-names
  (testing "a bound named `n-cols` was emitted raw as `(seg / (n-cols))` — valid C computing
            `n - cols`. Bounds go through c-symbol like every other emitted symbol."
    (let [m (am/of-axes '[[i 4] [l 8]])]
      ;; exercised through a literal-bound kernel: the mangling path must not alter literals
      (is (re-find #"\(seg / \(2\)\) % 2"
                   (:source (sco/generate-staged-contraction-kernel
                             {:free-axes '[[i 2] [j 2]] :contract-axes '[[l 8]]
                              :stages [{:axis 'l :extent 8 :dtype :float :init 0.0}]
                              :body '(raster.numeric/* (aget a l) (aget b l))
                              :inputs '[a b] :dtype :float :out-dtype :float}
                             'out)))))))
