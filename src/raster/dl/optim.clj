(ns raster.dl.optim
  "Optimizers for the Raster deep learning framework.

  All optimizers are deftm functions that update parameters in-place.
  They compile to single fused CUDA kernels per parameter.

  Optimizers:
    sgd-step!       - vanilla SGD
    adam-step!       - Adam (adaptive moments)
    adamw-step!      - AdamW (decoupled weight decay)

  Gradient clipping:
    clip-grad-norm!  - clip by global norm

  EMA:
    ema-update!      - exponential moving average

  Learning rate schedulers:
    cosine-lr, warmup-cosine-lr, step-lr, linear-warmup-lr"
  (:refer-clojure :exclude [aget aset alength aclone + - * / < > <= >= == zero? pos? neg? max min abs mod rem])
  (:require [raster.core :refer [deftm ftm broadcast reduce!]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.numeric :refer [+ - * / < > <= >= == zero? pos? neg? max min abs mod rem]]
            [raster.math :as m]
            [raster.numeric :as n]))

;; ================================================================
;; SGD
;; ================================================================

;; SGD: in-place weight update (mutating — param is shared state)
(deftm sgd-step! (All [T] [param :- (Array T), grad :- (Array T),
                           n :- Long, lr :- Double] :- (Array T)
                      (dotimes [i (alength param)]
                        (aset param i (- (aget param i) (* lr (aget grad i)))))
                      param))

;; Compat: some compiled paths pass n as Double
(deftm sgd-step! [param :- (Array double), grad :- (Array double),
                  n :- Double, lr :- Double] :- (Array double)
  (dotimes [i (alength param)]
    (aset param i (- (aget param i) (* lr (aget grad i)))))
  param)

;; ================================================================
;; Adam
;; ================================================================

(deftm adam-step! (All [T] [param :- (Array T) grad :- (Array T)
                            m :- (Array T) v :- (Array T)
                            n :- Long lr :- Double beta1 :- Double beta2 :- Double
                            eps :- Double t :- Long] :- (Array T)
                       (let [bc1 (- 1.0 (n/pow beta1 t))
                             bc2 (- 1.0 (n/pow beta2 t))]
                         (dotimes [i n]
                           (let [g (aget grad i)
                                 m-new (+ (* beta1 (aget m i)) (* (- 1.0 beta1) g))
                                 v-new (+ (* beta2 (aget v i)) (* (- 1.0 beta2) (* g g)))
                                 m-hat (/ m-new bc1)
                                 v-hat (/ v-new bc2)]
                             (aset m i m-new)
                             (aset v i v-new)
                             (aset param i
                                   (- (aget param i)
                                      (* lr (/ m-hat (+ (n/sqrt v-hat) eps)))))))
                         param)))

;; ================================================================
;; AdamW (decoupled weight decay)
;; ================================================================

(deftm adamw-step! (All [T] [param :- (Array T) grad :- (Array T)
                             m :- (Array T) v :- (Array T)
                             n :- Long lr :- Double beta1 :- Double beta2 :- Double
                             eps :- Double weight-decay :- Double t :- Long]
                        :- (Array T)
                        (let [bc1 (- 1.0 (n/pow beta1 t))
                              bc2 (- 1.0 (n/pow beta2 t))]
                          (dotimes [i n]
                            (let [g (aget grad i)
                                  m-new (+ (* beta1 (aget m i)) (* (- 1.0 beta1) g))
                                  v-new (+ (* beta2 (aget v i)) (* (- 1.0 beta2) (* g g)))
                                  m-hat (/ m-new bc1)
                                  v-hat (/ v-new bc2)
            ;; weight decay applied to param directly
                                  p-decayed (* (aget param i) (- 1.0 (* lr weight-decay)))]
                              (aset m i m-new)
                              (aset v i v-new)
                              (aset param i
                                    (- p-decayed (* lr (/ m-hat (+ (n/sqrt v-hat) eps)))))))
                          param)))

;; ================================================================
;; Gradient clipping
;; ================================================================

(deftm clip-grad-norm! (All [T] [grads :- (Array T), n :- Long,
                                 max-norm :- Double] :- (Array T)
                            (let [norm (n/sqrt (reduce! [s 0.0] [grads] (+ s (* grads grads))))]
                              (when (> norm max-norm)
                                (let [scale (/ max-norm norm)]
                                  (dotimes [i (alength grads)]
                                    (aset grads i (* (aget grads i) scale)))))
                              grads)))

;; ================================================================
;; EMA (Exponential Moving Average)
;; ================================================================

;; EMA: in-place state update (mutating — shadow is shared state)
(deftm ema-update! (All [T] [shadow :- (Array T), param :- (Array T),
                             n :- Long, mu :- Double] :- (Array T)
                        (dotimes [i (alength shadow)]
                          (aset shadow i (+ (* mu (aget shadow i)) (* (- 1.0 mu) (aget param i)))))
                        shadow))

;; ================================================================
;; Learning rate schedulers (pure functions, not deftm)
;; ================================================================

(deftm cosine-lr [base-lr :- Double step :- Long total-steps :- Long] :- Double
  (* 0.5 base-lr
     (+ 1.0 (m/cos (* n/pi (/ step (* 1.0 total-steps)))))))

(deftm warmup-cosine-lr [base-lr :- Double step :- Long total-steps :- Long warmup-steps :- Long] :- Double
  (if (< step warmup-steps)
    (* base-lr (/ step (* 1.0 warmup-steps)))
    (cosine-lr base-lr (- step warmup-steps)
               (- total-steps warmup-steps))))

(deftm step-lr [base-lr :- Double step :- Long step-size :- Long gamma :- Double] :- Double
  (* base-lr
     (n/pow gamma (quot step step-size))))

(deftm linear-warmup-lr [base-lr :- Double step :- Long warmup-steps :- Long] :- Double
  (if (< step warmup-steps)
    (* base-lr (/ step (* 1.0 warmup-steps)))
    base-lr))

;; ================================================================
;; Optimizer state management (convenience, not deftm)
;; ================================================================

(defn- alloc-like-arr
  "Allocate a zeroed array of the same type and length."
  [x]
  (let [n (alength x)]
    (if (instance? (Class/forName "[F") x)
      (float-array n)
      (double-array n))))

(defn make-adam-state
  "Create Adam optimizer state for a set of parameters.
  params: map of name->array (double[] or float[])
  Returns map of name->{:m array, :v array, :t (atom 0)}"
  [params]
  (reduce-kv
   (fn [acc name param]
     (assoc acc name {:m (alloc-like-arr param)
                      :v (alloc-like-arr param)
                      :t (atom 0)}))
   {} params))

(defn adam-update!
  "Apply Adam update to all parameters.
  params: map of name->array (double[] or float[])
  grads: map of name->array
  state: from make-adam-state
  Returns params (mutated in place)."
  [params grads state lr & {:keys [beta1 beta2 eps]
                            :or {beta1 0.9 beta2 0.999 eps 1e-8}}]
  (doseq [[name param] params]
    (when-let [grad (get grads name)]
      (let [{:keys [m v t]} (get state name)
            step (swap! t inc)
            n (alength param)]
        (adam-step! param grad m v n (double lr)
                    (double beta1) (double beta2)
                    (double eps) (long step)))))
  params)

;; ================================================================
;; Integrated optimizer (schedule-aware state management)
;; ================================================================

(defn- compute-lr
  "Compute learning rate for given optimizer state and step."
  ^double [state ^long step]
  (if-let [sched (:schedule state)]
    (case (:type sched)
      :cosine (cosine-lr (:base-lr state) step (:total-steps sched))
      :warmup-cosine (warmup-cosine-lr (:base-lr state) step
                                       (:total-steps sched) (:warmup-steps sched))
      :step (step-lr (:base-lr state) step (:step-size sched) (:gamma sched))
      :linear-warmup (linear-warmup-lr (:base-lr state) step (:warmup-steps sched))
      ;; default: constant
      (double (:base-lr state)))
    (double (:base-lr state))))

(defn make-optimizer
  "Create an optimizer with integrated schedule.

  params: map of name (string/keyword) → array
  opts: {:type      :sgd/:adam/:adamw
         :lr        base learning rate (double)
         :schedule  {:type :cosine/:warmup-cosine/:step/:linear-warmup
                     :total-steps N
                     :warmup-steps N (for warmup variants)
                     :step-size N :gamma G (for step schedule)}
         :beta1 0.9, :beta2 0.999, :eps 1e-8    (Adam/AdamW)
         :weight-decay 0.0                        (AdamW)
         :max-grad-norm nil}                      (optional gradient clipping)

  Returns optimizer state map."
  [params opts]
  (let [opt-type (or (:type opts) :adam)
        base-lr (or (:lr opts) 0.001)
        adam-state (when (#{:adam :adamw} opt-type)
                     (make-adam-state params))]
    {:step (atom 0)
     :type opt-type
     :base-lr base-lr
     :schedule (:schedule opts)
     :params-state adam-state
     :hyperparams {:beta1 (or (:beta1 opts) 0.9)
                   :beta2 (or (:beta2 opts) 0.999)
                   :eps (or (:eps opts) 1e-8)
                   :weight-decay (or (:weight-decay opts) 0.0)
                   :max-grad-norm (:max-grad-norm opts)}}))

(defn optimizer-step!
  "Apply one optimizer step with scheduled learning rate.

  params: map of name → array (must match keys from make-optimizer)
  grads: map of name → array
  state: optimizer state from make-optimizer
  Returns params (mutated in place)."
  [params grads state]
  (let [step (swap! (:step state) inc)
        lr (compute-lr state step)
        {:keys [beta1 beta2 eps weight-decay max-grad-norm]} (:hyperparams state)]
    ;; Optional gradient clipping
    (when max-grad-norm
      (doseq [[_name grad] grads]
        (clip-grad-norm! grad (alength grad) max-grad-norm)))
    ;; Apply optimizer step per parameter
    (doseq [[name param] params]
      (when-let [grad (get grads name)]
        (let [n (alength param)]
          (case (:type state)
            :sgd
            (sgd-step! param grad n lr)

            :adam
            (let [{:keys [m v t]} (get (:params-state state) name)
                  t-val (swap! t inc)]
              (adam-step! param grad m v n lr beta1 beta2 eps t-val))

            :adamw
            (let [{:keys [m v t]} (get (:params-state state) name)
                  t-val (swap! t inc)]
              (adamw-step! param grad m v n lr beta1 beta2 eps weight-decay t-val))))))
    params))

(defn get-lr
  "Get the current learning rate for the optimizer."
  ^double [state]
  (compute-lr state @(:step state)))
