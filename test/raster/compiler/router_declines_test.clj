(ns raster.compiler.router-declines-test
  "THE ROUTER MUST SAY WHY IT CHOSE A SLOWER LEAF. Device-free.

   `route-2free-1contract` ended with `(catch clojure.lang.ExceptionInfo _ nil)`. The tensorize
   emitters throw messages as specific as `tensorize: operand must be an aget` and
   `tensorize: needs literal dims`; every one was discarded, and the caller fell through to
   `:naive-segred` with nothing recorded. Discovering why a canonical matmul had been demoted meant
   reading emitter source — which is how an 8x slowdown lived in the compiler unnoticed.

   Two things are true at once, and the old shape could express neither:

     • a decline is usually LEGITIMATE — symbolic dims, a non-`+` combine, a non-product body are
       all perfectly good contractions that merely are not tiled-leaf shaped;
     • it is always worth REPORTING.

   So declines become data, and the distinction that must NOT be lost is decline-vs-fatal: a
   violated invariant recorded as a fallback would convert a loud error into a slow silent path,
   which is the exact inversion this work exists to prevent."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-route :as cr]
            [raster.compiler.core.hardware :as chw]))

(defn- mm
  [m n k & {:keys [sym-dims combine]}]
  (concat (list 'raster.par/contract 'C
                [['i (if sym-dims 'M m)] ['j n]] [['l k]]
                (list 'raster.numeric/*
                      (list 'clojure.core/aget 'A
                            (list 'clojure.core/+ (list 'clojure.core/* 'i k) 'l))
                      (list 'clojure.core/aget 'B
                            (list 'clojure.core/+ (list 'clojure.core/* 'l n) 'j))))
          (when combine [:combine combine])))

(defn- route [form dtype] (cr/route-contraction form :dtype dtype))

(deftest a-peak-route-declines-nothing
  (testing "when the fastest leaf is taken there is nothing to explain — an empty/absent :declines
            is the honest answer, not a fabricated 'not eligible'"
    (let [r (route (mm 128 128 128) :half)]
      (is (= :dpas (:strategy r)))
      (is (empty? (:declines r))))))

(deftest a-demoted-route-names-every-leaf-that-refused
  (testing "f64: DPAS declines on dtype and regtiled takes the work. The existing
            :fallback-reason is preserved for its consumers, AND the attempt is now itemized"
    (let [r (route (mm 128 128 128) :double)]
      (is (= :regtiled (:strategy r)))
      (is (= :dtype-not-dpas (:fallback-reason r)) "existing key kept")
      (is (= [[:dpas :dtype-not-dpas]] (mapv (juxt :leaf :reason) (:declines r))))))

  (testing "symbolic dims: BOTH tensorize leaves refuse, so we land on the general leaf — which
            previously recorded NOTHING AT ALL, not even a :fallback-reason key"
    (let [r (route (mm 128 128 128 :sym-dims true) :half)
          by-leaf (into {} (map (juxt :leaf identity)) (:declines r))]
      (is (= :naive-segred (:strategy r)))
      (is (= #{:dpas :regtiled} (set (keys by-leaf))) "both attempts itemized")
      (is (= :symbolic-dims (:reason (get by-leaf :regtiled))))
      (is (re-find #"literal dims" (str (:message (get by-leaf :regtiled))))
          "the emitter's own sentence survives — it is the thing that was being thrown away")
      (is (= :symbolic-dims (:fallback-reason r))
          "the headline reason is the DECISIVE (last) decline — the leaf that would otherwise have
           taken the work — not DPAS's generic :not-a-contraction")))

  (testing "a non-+ combine is a legitimate contraction that simply is not tiled-leaf shaped"
    (let [r (route (mm 128 128 128 :combine 'max) :half)
          by-leaf (into {} (map (juxt :leaf identity)) (:declines r))]
      (is (= :naive-segred (:strategy r)))
      (is (= :non-plus-combine (:reason (get by-leaf :regtiled))))
      (is (re-find #"combine must be \+" (str (:message (get by-leaf :regtiled))))))))

(deftest a-shape-with-no-faster-leaf-claims-no-declines
  (testing "a 0-free full reduction never attempts a tensorize leaf, so reporting one as having
            'declined' would be a fabrication"
    (let [r (route '(raster.par/contract O [] [[l 256]]
                      (raster.numeric/* (clojure.core/aget A l) (clojure.core/aget B l)))
                   :double)]
      (is (empty? (:declines r))))))

;; ── the guard that matters most ─────────────────────────────────────────────────────
(deftest only-whitelisted-legality-reasons-become-declines
  (testing "A DENYLIST was written here first and was wrong the usual way — it assumed every
            unknown reason is safe to treat as a decline. `emit-gemm-tiled` throws validation
            errors with NO :reason at all (a non-divisible tile, split-k with beta≠0), so an
            emitter BUG would have been filed as 'not tiled-leaf shaped' and silently demoted.
            Only whitelisted legality reasons are declines; anything else propagates."
    (let [decline-of (var-get (requiring-resolve
                              'raster.compiler.passes.parallel.contract-route/decline-of))]
      (testing "a documented leaf-legality reason becomes a record"
        (let [d (decline-of :regtiled (ex-info "tensorize: needs literal dims"
                                               {:reason :symbolic-dims :dims '[M 128 128]}))]
          (is (= :regtiled (:leaf d)))
          (is (= :symbolic-dims (:reason d)))
          (is (= "tensorize: needs literal dims" (:message d)))
          (is (= '{:dims [M 128 128]} (:data d)) ":reason is lifted, not duplicated into :data")))
      (testing "an invariant violation escapes"
        (doseq [fatal [:raster/fatal :raster/bug]]
          (is (thrown? clojure.lang.ExceptionInfo
                       (decline-of :dpas (ex-info "invariant violated" {:reason fatal})))
              (str fatal " must escape, not become a decline"))))
      (testing "…and so does an exception with NO reason, or an unrecognized one"
        (is (thrown? clojure.lang.ExceptionInfo
                     (decline-of :dpas (ex-info "emit-gemm-tiled: not divisible" {:tile [129 128]}))))
        (is (thrown? clojure.lang.ExceptionInfo
                     (decline-of :dpas (ex-info "something new" {:reason :a-reason-nobody-declared}))))))))

(deftest an-emitter-failure-propagates-through-the-WHOLE-router
  (testing "the end-to-end version of the above, injected through `route-contraction` rather than
            at the private classifier: a tile whose block-m is not divisible by sg-m makes
            emit-gemm-tiled throw a REASONLESS validation error. That must surface, not become a
            quietly slower kernel"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not divisible"
         (cr/route-contraction (mm 128 128 128) :dtype :half
                               :tile {:block-m 129 :block-n 128 :sg-m 32 :sg-n 32 :block-k 32
                                      :matrix {:m 8 :n 16 :k 16 :subgroup 16}})))))

;; ── the schedule axis that was declared, costed, gated — and never emitted ───────────
(deftest pipeline-depth-actually-reaches-the-emitted-kernel
  (testing "`:num-stages` is a schedule key with a schema, an SLM-capped feasibility filter and an
            autotune axis. The contraction door never passed it to the emitter, so every routed
            kernel used emit-gemm-tiled's `:or prefetch 3` and tiles differing ONLY in :num-stages
            emitted IDENTICAL source — a tuner 'measuring' that axis was measuring noise.

            This is north-star §10's 'no knob without emission', and the resident door had always
            passed it (ze_runtime `:prefetch (:num-stages tile 3)`)."
    (let [desc (try (chw/descriptor-for :ze:0) (catch Throwable _ nil))
          base (chw/gemm-tile-for desc)
          norm #(clojure.string/replace (str %) #"_\d{3,}" "_N")
          src-for (fn [ns] (norm (:source (cr/route-contraction (mm 256 256 256) :dtype :half
                                                                :tile (assoc base :num-stages ns)
                                                                :desc desc))))
          sources (mapv src-for [2 3 4 8])]
      (is (= 4 (count (distinct sources)))
          "four pipeline depths must produce four different kernels")
      (is (= :dpas (:strategy (cr/route-contraction (mm 256 256 256) :dtype :half :desc desc)))
          "and threading it must not change which leaf is chosen"))))
