(ns raster.ad.source-program-context-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]))

(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))

(deftest ordered-context-projects-over-one-saved-forward-frame
  (let [x (typed 'x 'double) counts (typed 'counts 'longs)
        a (typed 'a 'double) b (typed 'b 'double) result (typed 'result 'double)
        hidden (typed 'hidden 'doubles)]
    (with-bindings {#'reverse/*gensym-counter* (atom 0)
                    #'reverse/*source-memory-context*
                    {:values [x counts] :value-facts {x {:value x} counts {:value counts}
                                                     hidden {:value hidden}} :aliases {}}}
      (let [state (#'reverse/forward-pass
                   ['tick '(clojure.core/aset counts 0 (clojure.core/inc (clojure.core/aget counts 0)))
                    a (list 'raster.numeric/* x 1.0e16)
                    b (list 'raster.numeric/* x -1.0e16)
                    result (list 'raster.numeric/+ a b)] [x])
            program (#'reverse/source-program-residual state result [x])
            forward (eval (list 'fn* [x counts] (:form program)))
            counters (long-array 1)
            [primal residual] (forward 1.0 counters)
            projections
            (with-redefs-fn
              {#'reverse/forward-pass (fn [& _] (throw (ex-info "backward rebuilt forward state" {})))}
              #(mapv (fn [incoming]
                       (#'reverse/project-source-program-residual program 'saved 1.0 incoming))
                     [{} {x [1.0]} {x [2.0]}]))
            backward (mapv #(eval (list 'fn* '[saved] (:form %))) projections)]
        (is (= 0.0 primal))
        (is (= [1] (vec counters)))
        (is (not (contains? (set (:residual-ports program)) hidden))
            "a fact catalog does not grant a residual port")
        (is (= (:raster.type/tag (meta x))
               (:raster.type/tag (meta (first (:residual-ports program))))))
        (is (= [[0.0] [0.0] [2.0]] (mapv #(% residual) backward)))
        (is (= [0.0] ((first backward) residual)))
        (is (= [1] (vec counters)) "repeated backward calls do not replay primal effects")
        (is (= 1.0 (+ 1.0 (first ((first backward) residual))))
            "appending a closed partial sum is observably different from ordered context projection")))))

(deftest incoming-vectors-stay-in-backward-scope-and-retain-duplicates
  (let [x (typed 'x 'double) result (typed 'result 'double)]
    (with-bindings {#'reverse/*gensym-counter* (atom 0)
                    #'reverse/*source-memory-context* {:values [x] :value-facts {x {:value x}} :aliases {}}}
      (let [state (#'reverse/forward-pass [result (list 'raster.numeric/* x 2.0)] [x])
            program (#'reverse/source-program-residual state result [x])
            marker '(do (clojure.core/swap! calls clojure.core/inc) 1.0)
            projection (#'reverse/project-source-program-residual program 'saved 1.0 {x [marker marker]})
            forward (eval (list 'fn* [x] (:form program)))
            backward (eval (list 'fn* '[saved calls] (:form projection)))
            calls (atom 0)
            [_ residual] (forward 3.0)]
        (is (= 0 @calls))
        (is (not (contains? (set (:residual-ports program)) 'calls)))
        (is (= [4.0] (backward residual calls)))
        (is (= 2 @calls))))))

(deftest float-context-is-projected-only-at-the-existing-parameter-checkpoint
  (let [x (typed 'x 'float) result (typed 'result 'float)]
    (with-bindings {#'reverse/*gensym-counter* (atom 0)
                    #'reverse/*source-memory-context* {:values [x] :value-facts {x {:value x}} :aliases {}}}
      (let [state (#'reverse/forward-pass [result (list 'raster.numeric/* x -1.0e8)] [x])
            program (#'reverse/source-program-residual state result [x])
            projection (#'reverse/project-source-program-residual program 'saved 1.0 {x [1.0e8 1.0]})
            forward (eval (list 'fn* [x] (:form program)))
            backward (eval (list 'fn* '[saved] (:form projection)))
            [_ residual] (forward (float 2.0))
            [dx] (backward residual)]
        (is (= (float 1.0) dx))
        (is (instance? Float dx))
        (is (= (float 0.0) (float (+ (float (+ 1.0e8 1.0)) -1.0e8)))
            "pre-projecting incoming contributions would add an unauthorized checkpoint")))))

(deftest cotangent-context-is-an-explicit-ordered-boundary
  (let [program {:params '[x]}]
    (doseq [incoming [{'other [1.0]} {'x '(1.0)} {'x nil}]]
      (is (= :ad-cotangent-context
             (try (#'reverse/project-source-program-residual program 'saved 1.0 incoming) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest prefix-loop-pullbacks-reuse-the-saved-primal-frame
  (let [x (typed 'x 'double) counts (typed 'counts 'longs)
        cnt (typed 'cnt 'long) z (typed 'z 'double) acc (typed 'acc 'double)]
    (with-bindings {#'reverse/*gensym-counter* (atom 0)
                    #'reverse/*source-memory-context*
                    {:values [x counts cnt] :aliases {}
                     :value-facts {x {:value x} counts {:value counts} cnt {:value cnt}}}}
      (let [loop-form (list 'loop* ['i 0 acc z]
                            (list 'if '(< i cnt)
                                  (list 'let* '[tick (clojure.core/aset counts 0
                                                     (clojure.core/inc (clojure.core/aget counts 0)))]
                                        (list 'recur '(inc i) (list 'raster.numeric/* acc x)))
                                  acc))
            program (#'reverse/gen-reverse-loop-with-let-program
                     [z (list 'raster.numeric/* x 2.0)] loop-form [x])
            forward (eval (list 'fn* [x counts cnt] (:form program)))]
        (doseq [[steps expected dx] [[0 4.0 2.0] [2 16.0 24.0]]]
          (let [counters (long-array 1)
                [value pullback] (forward 2.0 counters steps)]
            (is (= expected value))
            (is (= [steps] (vec counters)))
            (is (= [dx] (pullback 1.0)))
            (is (= [(* 2.0 dx)] (pullback 2.0)))
            (is (= [steps] (vec counters)) "backward does not repeat loop effects")))))))
