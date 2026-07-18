(ns raster.dl.gpu-grad-parity
  "GPU-vs-CPU gradient PARITY HARNESS (Phase-1 QLoRA backward-kernel gate).

   `grad-parity` takes a loss deftm var + explicit arg-specs (name/type/value per
   param), computes the CPU-interpreted reference gradients via
   raster.ad.reverse/value+grad, then for each requested gradient builds a
   value+grad WRAPPER deftm — (let [vg ((value+grad #'loss) args…) g (nth vg k)]
   (copy-into! g out n)) → the k-th grad array copied into a resident output buffer
   — compiles THAT through the resident GPU path (compile-gpu-program at :dtype, the
   same route the resident AD-GEMM test uses) and runs it on ze:0. NB: value+grad is
   bound to its own symbol BEFORE nth — the AD transform only fires on a directly-
   bound call; nesting it inside nth silently yields the raw inputs (see .internal
   phase1_composition_findings F1), and a bare (nth vg k) terminal extracts to no
   kernels (F2), hence the copy-into! bridge.

   It asserts, per gradient:
     • the wrapper extracts FULLY RESIDENT — compile-gpu-program returns a non-nil
       descriptor whose every step is a resident kernel convention (no CPU-fallback
       staging-fn). A nil descriptor (a kernel silently staying on the host) FAILS
       loudly (this is the whole point of the gate).
     • the GPU grad matches the CPU grad within rtol (default 1e-3 for f32).

   Returns a report {:grads {argname {:step-kinds [...] :rel-err d :resident? bool}}}
   so callers can additionally assert the exact resident step-kind lists.

   Study/mirror: raster.dl.gpu-ad-gemm-test (run-resident + residency assertions)."
  (:require [clojure.test :refer [is]]
            [raster.core :refer [deftm]]
            [raster.compiler.pipeline :as pl]
            [raster.par]
            [raster.arrays]
            [raster.ad.reverse :as rev]))

;; Resident bridge kernel: copy the (untyped Object) grad `src` — pulled out of the
;; value+grad result vector via nth, so it carries no array tag — into a caller-
;; provided output buffer `out`. As an (All [T]) primitive its (Array T) param
;; RE-TYPES src on inline (monomorphized to the compile dtype), so aget src
;; devirtualizes and the copy lowers to a resident :map-void kernel. This makes the
;; grad a genuine KERNEL OUTPUT (a bare (nth vg k) terminal extracts to no kernels),
;; giving the EXACT gradient with no subtractive cancellation.
(deftm copy-into! (All [T] [src :- (Array T) out :- (Array T) n :- Long] :- (Array T)
                       (raster.par/map-void! i n
                                             (raster.arrays/aset out i (raster.arrays/aget src i)))
                       out))

;; ── GPU availability probe (HONEST: distinguishes "no device" from "broken load") ──
;; The old form was `(try … (catch Throwable _ false))`, which swallowed a BROKEN
;; ze-runtime load (a compile error in the ns, a missing native symbol, an FFM bind
;; failure) as if it meant "no GPU here". Every GPU-gated deftest then took its skip
;; branch and the ENTIRE GPU suite went green with zero assertions — a suite that
;; reported success while testing nothing. `gpu-status` separates the cases so a
;; runtime breakage surfaces LOUDLY instead of masquerading as an absent device.
(def gpu-status
  "Delay yielding one of:
     {:status :available   :n-devices k}  Level-Zero device(s) present.
     {:status :no-device}                 runtime loaded cleanly, zero devices — a
                                          LEGITIMATE, visible skip.
     {:status :probe-error :error e}      query-devices threw (e.g. no L0 loader on a
                                          GPU-less box) — a loud WARNING + skip, not a
                                          hard failure (won't redden CI with no GPU).
     {:status :load-failed :error e}      `require 'raster.gpu.ze-runtime` THREW — the
                                          runtime failed to LOAD. This is a breakage,
                                          NOT 'no device', and must fail loud."
  (delay
    (let [loaded (try (require 'raster.gpu.ze-runtime) {:ok true}
                      (catch Throwable e {:ok false :error e}))]
      (if-not (:ok loaded)
        {:status :load-failed :error (:error loaded)}
        (try
          (let [devs ((resolve 'raster.gpu.ze-runtime/query-devices))]
            (if (seq devs)
              {:status :available :n-devices (count devs)}
              {:status :no-device}))
          (catch Throwable e {:status :probe-error :error e}))))))

;; Boolean convenience for `(if-not @gp/gpu-available? …)` call sites: TRUE only when a
;; device is actually usable. :load-failed is FALSE here so the skip branch runs — but
;; that branch MUST route through `gpu-skip!`, which converts :load-failed into a
;; failing assertion (below), so the breakage is never silently skipped.
(def gpu-available?
  (delay (= :available (:status @gpu-status))))

;; Accumulates skip reasons across the whole JVM run so ONE authoritative summary line
;; can be printed at shutdown, instead of the count silently wandering ±N run-to-run.
(defonce ^:private gpu-skip-log (atom {}))

(defonce ^:private gpu-summary-hook
  (delay
    (.addShutdownHook
     (Runtime/getRuntime)
     (Thread.
      (fn []
        (let [{:keys [status n-devices]} @gpu-status
              skips @gpu-skip-log
              total (reduce + 0 (vals skips))]
          (println (format "  [GPU SUITE] probe=%s%s | %d test(s) took the skip path%s"
                           (name status)
                           (if n-devices (str " (" n-devices " device)") "")
                           total
                           (if (pos? total) (str " " (pr-str skips)) "")))))))
    true))

(defn gpu-skip!
  "Call in the SKIP branch of a GPU-gated deftest instead of a bare `println`. It:
     • registers exactly ONE marker assertion so the suite's assertion count is
       DETERMINISTIC whether or not a GPU is present (skips no longer drop the body's
       assertions silently → no more ±N count wander between runs);
     • emits a visible, attributed skip line and records the reason for the shutdown
       summary;
     • on :load-failed registers a FAILING assertion — a broken runtime load must
       never masquerade as a clean skip."
  [test-label]
  @gpu-summary-hook
  (let [{:keys [status error]} @gpu-status]
    (swap! gpu-skip-log update status (fnil inc 0))
    (case status
      :load-failed
      (is false (str "[GPU LOAD FAILED] " test-label
                     " — raster.gpu.ze-runtime failed to load: "
                     (some-> error .getMessage)
                     ". This is a RUNTIME BREAKAGE, not 'no GPU device' — the whole "
                     "GPU suite would otherwise go green having tested nothing."))
      :probe-error
      (do (println (str "  [GPU SKIP/WARN] " test-label
                        " — query-devices threw (" (some-> error .getMessage)
                        "); treating as no usable device."))
          (is true "gpu-skip-marker"))
      :no-device
      (do (println (str "  [GPU SKIP] " test-label " — no Level-Zero device"))
          (is true "gpu-skip-marker"))
      :available
      (throw (IllegalStateException.
              (str "gpu-skip! called for " test-label " while a GPU IS available"))))))

;; ── GPU session plumbing (mirrors gpu-ad-gemm-test/run-resident) ────────────────

(defn- run-resident
  "Bind + replay f-var's resident descriptor on ze:0 for `args`; returns the result array.
   gemm-precision is compile-gpu-program's :gemm-precision (:f16-xmx | :f32-scalar)."
  [f-var args dtype gemm-precision]
  (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
        make-session (ns-resolve gpu 'make-session)
        bind-program! (ns-resolve gpu 'bind-program!)
        run-program! (ns-resolve gpu 'run-program!)
        close-session! (ns-resolve gpu 'close-session!)
        p (pl/compile-gpu-program f-var :ze:0 :dtype dtype :gemm-precision gemm-precision)
        s (make-session :ze:0)]
    (try
      (bind-program! s p args {})
      (let [r (run-program! s p args)]
        {:descriptor p :out (or (get r (:result-sym p)) (first (vals r)))})
      (finally (close-session! s)))))

(defn- rel-err [gpu cpu]
  (let [err (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) (seq gpu) (seq cpu)))
        mag (reduce max 1e-9 (map #(Math/abs (double %)) (seq cpu)))]
    (/ err mag)))

(defn- array-type? [t]
  (and (sequential? t) (= 'Array (first t))))

(defn- build-wrapper
  "Eval a value+grad wrapper deftm that extracts the k-th component (a grad array)
   of (value+grad loss) and COPIES it into a fresh output buffer via copy-into!, so
   the grad is a resident kernel output. value+grad MUST bind to its own symbol
   before nth — the AD transform only fires on a directly-bound call, not one nested
   inside another form. Returns the wrapper var. Extra params: __gout__ (zeros of the
   grad's array type, the result buffer) and __gn__ (its length, the copy grid bound)."
  [wname names types loss-fqsym k ret-type]
  (let [param-vec (vec (concat (mapcat (fn [n t] [n :- t]) names types)
                               ['__gout__ :- ret-type '__gn__ :- 'Long]))]
    (eval
     ;; return the bare __gout__ SYMBOL (not the copy-into! call) so the resident
     ;; descriptor's :result-sym is a plain array-param symbol — run-program!'s
     ;; functional-result path calls (name result-sym) and chokes on a call list.
     `(raster.core/deftm ~wname ~param-vec :- ~ret-type
        (let [vg# ((rev/value+grad (var ~loss-fqsym)) ~@names)
              g#  (clojure.core/nth vg# ~k)
              _#  (copy-into! g# ~'__gout__ ~'__gn__)]
          ~'__gout__)))))

(defn grad-parity
  "GPU-vs-CPU gradient parity for a loss deftm var.

   arg-specs: vector (in loss param order) of {:name sym :type type-form :val value}.
     scalar params (Long/Double) carry their value; array params carry the input array.
   opts:
     :dtype       :float (default) | :double   — resident compile dtype
     :rtol        max relative error (default 1e-3 for f32)
     :grad-args   coll of arg :name symbols to check (default: all Array-typed args)
     :gemm-precision  :f16-xmx (default) | :f32-scalar — compile-gpu-program's resident
                  :gemm binding policy (:f32-scalar = exact-grad scalar f32 GEMM)

   Runs on ze:0. Fails loudly (via clojure.test/is) if any checked gradient's wrapper
   does not extract fully resident, or if the GPU grad diverges beyond rtol.
   Returns {:grads {argname {:resident? :step-kinds :rel-err}}}."
  [loss-var arg-specs & {:keys [dtype rtol grad-args gemm-precision]
                         :or {dtype :float rtol 1.0e-3 gemm-precision :f16-xmx}}]
  (let [names (mapv :name arg-specs)
        types (mapv :type arg-specs)
        vals  (mapv :val arg-specs)
        vmeta (meta loss-var)
        loss-fqsym (symbol (str (:ns vmeta)) (str (:name vmeta)))
        ;; CPU reference: [loss grad0 grad1 …]; grad for arg j is at index (j+1).
        cpu-vg (apply (rev/value+grad loss-var) vals)
        targets (or grad-args
                    (keep-indexed (fn [j s] (when (array-type? (:type s)) (:name s))) arg-specs))
        report (reduce
                (fn [acc gname]
                  (let [j (long (first (keep-indexed (fn [i n] (when (= n gname) i)) names)))
                        k (inc j)
                        cpu-grad (nth cpu-vg k)
                        glen (count (nth vals j))
                        gout (if (= dtype :double) (double-array glen) (float-array glen))
                        ;; wrapper takes the loss args ++ [__gout__ (zeros) __gn__ (len)]
                        wargs (conj (vec vals) gout glen)
                        wname (symbol (str "grad-parity-" (name (:name vmeta)) "-" (name gname)
                                           "-" (Math/abs (hash [loss-fqsym gname dtype]))))
                        wvar (build-wrapper wname names types loss-fqsym k (nth types j))
                        ;; residency probe (no throw): nil ⇒ a step fell to the host.
                        desc (pl/compile-gpu-program wvar :ze:0 :dtype dtype :on-non-resident :nil
                                                     :gemm-precision gemm-precision)
                        step-kinds (mapv :convention (:steps desc))
                        _ (is (some? desc)
                              (str "grad(" gname ") of " loss-fqsym
                                   " must extract FULLY RESIDENT (compile-gpu-program returned nil ⇒ "
                                   "a kernel fell back to the host)"))
                        re (when desc
                             (let [{:keys [out]} (run-resident wvar wargs dtype gemm-precision)
                                   e (rel-err out cpu-grad)]
                               (is (< e rtol)
                                   (str "grad(" gname ") GPU-vs-CPU rel-err " e " (rtol " rtol ")"))
                               e))]
                    (assoc acc gname {:resident? (some? desc)
                                      :step-kinds step-kinds
                                      :rel-err re})))
                {} targets)]
    {:grads report}))
