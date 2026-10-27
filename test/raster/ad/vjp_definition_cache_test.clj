(ns raster.ad.vjp-definition-cache-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.core :as core]))

(defn- with-model [f]
  (let [namespace (create-ns (gensym "raster.vjp-definition-model."))]
    (try
      (binding [*ns* namespace]
        (refer 'clojure.core)
        (alias 'n 'raster.numeric)
        ;; Keep this lifecycle fixture on the genuine retained-helper route
        ;; under both the committed and whole-call inliners. :no-inline is the
        ;; public helper opacity policy, not fabricated compiler evidence.
        (eval '(raster.core/deftm ^:no-inline child [x :- Double] :- Double (n/* x x)))
        (eval '(raster.core/deftm parent [x :- Double] :- Double
                 (n/+ (child x) 1.0)))
        (f namespace (ns-resolve namespace 'parent)))
      (finally (remove-ns (ns-name namespace))))))

(defn- replace-child! []
  (eval '(raster.core/deftm ^:no-inline child [x :- Double] :- Double (n/* 3.0 x))))

(deftest nested-definition-change-invalidates-a-fresh-vjp-acquisition
  (with-model
    (fn [_ parent]
      (let [[before old-pullback] (reverse/vjp parent 2.0)
            resolved (core/resolve-deftm-var parent)
            walked (core/ensure-walked-body! resolved)
            revision (dispatch/compiler-definition-revision)]
        (is (= 5.0 before))
        (is (= [4.0] (old-pullback 1.0)))
        (is (= [2.0] (old-pullback 0.5))
            "the owned paired fallback retains the public arbitrary seed port")
        (replace-child!)
        (is (< revision (dispatch/compiler-definition-revision)))
        (is (= walked (core/ensure-walked-body! resolved)))
        (let [[after new-pullback] (reverse/vjp parent 2.0)]
          (is (= 7.0 after))
          (is (= [3.0] (new-pullback 1.0)))
          (is (= [6.0] (new-pullback 2.0)))
          (is (= [4.0] (old-pullback 1.0))
              "an already acquired pullback retains its captured residual"))))))

(deftest unchanged-definitions-still-reuse-the-vjp
  (with-model
    (fn [_ parent]
      (let [transform reverse/transform-body
            calls (atom 0)]
        (with-redefs [reverse/transform-body
                      (fn [& args] (swap! calls inc) (apply transform args))]
          (reverse/vjp parent 2.0)
          (let [initial @calls]
            (is (pos? initial))
            (let [[value pullback] (reverse/vjp parent 3.0)]
              (is (= 10.0 value))
              (is (= [6.0] (pullback 1.0)))
              (is (= initial @calls)))))))))

(deftest mutation-after-paired-construction-discards-the-old-definition-snapshot
  (with-model
    (fn [namespace parent]
      (let [construct @#'reverse/get-selected-paired-helper-program
            replaced? (atom false)
            constructions (atom [])
            reason (fn [guard]
                     (try (guard) nil
                          (catch clojure.lang.ExceptionInfo failure
                            (:reason (ex-data failure)))))]
        (with-redefs-fn
          {#'reverse/get-selected-paired-helper-program
           (fn [& arguments]
             (let [artifact (apply construct arguments)]
               (swap! constructions conj artifact)
               ;; Mutate only after the old parent artifact is complete. The
               ;; enclosing revision fence must discard it and retry the whole
               ;; acquisition rather than publishing mixed source snapshots.
               (when (compare-and-set! replaced? false true)
                 (binding [*ns* namespace] (replace-child!)))
               artifact))}
          (fn []
            (let [[value pullback] (reverse/vjp parent 2.0)
                  construction-count (count @constructions)]
              (is @replaced?)
              (is (<= 2 construction-count)
                  "the stale completed artifact forces a fresh root construction")
              (is (some? (reason (:guard (first @constructions))))
                  "the discarded artifact retains its stale-source refusal")
              (is (nil? (reason (:guard (last @constructions)))))
              (is (= 7.0 value))
              (is (= [3.0] (pullback 1.0)))
              (let [[cached-value cached-pullback] (reverse/vjp parent 2.0)]
                (is (= construction-count (count @constructions))
                    "the unchanged retried artifact is reused without reconstruction")
                (is (= 7.0 cached-value))
                (is (= [3.0] (cached-pullback 1.0)))))))))))

(deftest mutation-during-walk-does-not-return-an-old-cache-hit
  (with-model
    (fn [namespace parent]
      (reverse/vjp parent 2.0)
      (let [walk core/ensure-walked-body!
            resolved (core/resolve-deftm-var parent)
            replaced? (atom false)]
        (with-redefs [core/ensure-walked-body!
                      (fn [v]
                        (let [result (walk v)]
                          (when (and (identical? v resolved)
                                     (compare-and-set! replaced? false true))
                            (binding [*ns* namespace] (replace-child!)))
                          result))]
          (let [[value pullback] (reverse/vjp parent 2.0)]
            (is @replaced?)
            (is (= 7.0 value))
            (is (= [3.0] (pullback 1.0)))))))))
