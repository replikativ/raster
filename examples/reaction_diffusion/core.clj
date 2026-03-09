(ns reaction-diffusion.core
  "GPU-accelerated Gray-Scott reaction-diffusion with interactive cljfx UI.

  Two chemicals U and V diffuse and react on a 2D grid:
    dU/dt = Du*∇²U - U*V² + F*(1-U)
    dV/dt = Dv*∇²V + U*V² - (F+k)*V

  The Laplacian ∇² uses a 5-point stencil with toroidal boundary conditions.
  Different (F, k) parameter regimes produce spots, stripes, spirals, coral, etc.

  Uses raster.gpu.core for kernel compilation and buffer management. The simulation
  kernels are written as deftm functions using par/map-void!, compiled to
  OpenCL/SPIR-V automatically. Zero-copy display on integrated GPUs.

  Usage:
    clj -M:demo -m demo.reaction-diffusion

  Or from REPL (with :demo alias):
    (load-file \"demo/reaction_diffusion.clj\")"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.par :as par]
            [raster.gpu.core :as gpu]
            [cljfx.api :as fx])
  (:import [java.nio ByteOrder]
           [javafx.scene.image WritableImage PixelFormat]
           [javafx.scene.input MouseEvent]
           [javafx.animation AnimationTimer]
           [javafx.application Platform]))

;; ================================================================
;; Gray-Scott kernels as deftm + par primitives
;; ================================================================

(deftm gray-scott-step!
  "One Gray-Scott diffusion step. Reads from U/V, writes to U2/V2.
   5-point Laplacian with toroidal boundary conditions."
  [U :- (Array float), V :- (Array float),
   U2 :- (Array float), V2 :- (Array float),
   Du :- Float, Dv :- Float, F :- Float, k :- Float,
   W :- Long, H :- Long, n :- Long] :- Void
  (par/map-void! idx n
    (let [x (int (rem idx W))
          y (int (quot idx W))
          ;; Toroidal wrap
          xm (int (rem (+ x (- W 1)) W))
          xp (int (rem (+ x 1) W))
          ym (int (rem (+ y (- H 1)) H))
          yp (int (rem (+ y 1) H))
          ;; Current values
          u (float (aget U idx))
          v (float (aget V idx))
          ;; 5-point Laplacian
          lap-u (float (- (+ (aget U (+ (* ym W) x))
                             (aget U (+ (* yp W) x))
                             (aget U (+ (* y W) xm))
                             (aget U (+ (* y W) xp)))
                          (* (float 4.0) u)))
          lap-v (float (- (+ (aget V (+ (* ym W) x))
                             (aget V (+ (* yp W) x))
                             (aget V (+ (* y W) xm))
                             (aget V (+ (* y W) xp)))
                          (* (float 4.0) v)))
          ;; Reaction terms
          uvv (float (* u v v))]
      (aset U2 idx (float (+ u (* Du lap-u) (- uvv) (* F (- (float 1.0) u)))))
      (aset V2 idx (float (+ v (* Dv lap-v) uvv (- (* (+ F k) v))))))))

(deftm clamp01 [x :- Float] :- Float
  (if (< x (float 0.0)) (float 0.0)
    (if (> x (float 1.0)) (float 1.0) x)))

(deftm colorize!
  "Convert U/V concentrations to ARGB int pixels."
  [U :- (Array float), V :- (Array float),
   pixels :- (Array int), n :- Long] :- Void
  (par/map-void! idx n
    (let [u (float (aget U idx))
          v (float (aget V idx))
          r (int (* (clamp01 (* v (float 5.0))) (float 255.0)))
          g (int (* (clamp01 (- (* u (float 1.5)) (float 0.3))) (float 255.0)))
          b (int (* (clamp01 (- (* (- (float 1.0) u) (float 2.0)) (float 0.5))) (float 255.0)))]
      ;; ARGB packed int (alpha=255)
      (aset pixels idx (int (bit-or (bit-shift-left (int 255) (int 24))
                                     (bit-shift-left r (int 16))
                                     (bit-shift-left g (int 8))
                                     b))))))

;; ================================================================
;; Simulation parameters
;; ================================================================

(def W 512)
(def H 512)
(def N (* W H))

(def presets
  {"Mitosis"      {:F 0.0367 :k 0.0649}
   "Coral"        {:F 0.0545 :k 0.062}
   "Spirals"      {:F 0.014  :k 0.045}
   "Spots"        {:F 0.035  :k 0.065}
   "Worms"        {:F 0.078  :k 0.061}
   "Maze"         {:F 0.029  :k 0.057}
   "Fingerprint"  {:F 0.055  :k 0.062}
   "Bubbles"      {:F 0.012  :k 0.050}
   "Waves"        {:F 0.025  :k 0.060}})

(def *state
  (atom {:F 0.055
         :k 0.062
         :Du 0.21
         :Dv 0.105
         :steps-per-frame 8
         :running true
         :fps 0.0
         :frame-count 0
         :preset "Fingerprint"}))

;; ================================================================
;; GPU buffer management via raster.gpu.core
;; ================================================================

(defonce gpu-state (atom nil))

(defn init-gpu!
  "Compile kernels and allocate GPU buffers."
  []
  (let [sess (gpu/make-session :ze:0)]
    (gpu/compile! sess :step #'gray-scott-step!)
    (gpu/compile! sess :colorize #'colorize!)
    (gpu/alloc! sess {:u1     [:float N nil]
                      :v1     [:float N nil]
                      :u2     [:float N nil]
                      :v2     [:float N nil]
                      :pixels [:int   N nil]})
    (reset! gpu-state sess)))

(defn shutdown-gpu! []
  (when-let [sess @gpu-state]
    (gpu/close-session! sess)
    (reset! gpu-state nil)))

;; ================================================================
;; Grid initialization
;; ================================================================

(defn init-grid!
  "Initialize U=1 everywhere, V=0 everywhere, with a seed region in the center."
  []
  (let [sess @gpu-state
        u-arr (float-array N (float 1.0))
        v-arr (float-array N (float 0.0))
        rng (java.util.Random.)
        cx (/ W 2) cy (/ H 2) r 10]
    ;; Seed center region
    (doseq [dy (range (- r) (inc r))
            dx (range (- r) (inc r))]
      (let [x (+ cx dx) y (+ cy dy)]
        (when (and (>= x 0) (< x W) (>= y 0) (< y H))
          (let [i (+ (* y W) x)]
            (clojure.core/aset u-arr i (float (+ 0.5 (* 0.1 (.nextFloat rng)))))
            (clojure.core/aset v-arr i (float (+ 0.25 (* 0.1 (.nextFloat rng)))))))))
    ;; Upload to GPU
    (gpu/upload! sess :u1 u-arr)
    (gpu/upload! sess :v1 v-arr)
    (gpu/upload! sess :u2 u-arr)
    (gpu/upload! sess :v2 v-arr)))

(defn seed-at!
  "Seed a perturbation at pixel coordinates (px, py)."
  [px py]
  (let [sess @gpu-state
        ;; Download current U/V, modify, re-upload
        u-arr (gpu/download sess :u1)
        v-arr (gpu/download sess :v1)
        rng (java.util.Random.)
        r 5]
    (doseq [dy (range (- r) (inc r))
            dx (range (- r) (inc r))]
      (let [x (+ px dx) y (+ py dy)]
        (when (and (>= x 0) (< x W) (>= y 0) (< y H))
          (let [i (+ (* y W) x)]
            (clojure.core/aset ^floats u-arr i (float (+ 0.5 (* 0.1 (.nextFloat rng)))))
            (clojure.core/aset ^floats v-arr i (float (+ 0.25 (* 0.1 (.nextFloat rng)))))))))
    (gpu/upload! sess :u1 u-arr)
    (gpu/upload! sess :v1 v-arr)))

;; ================================================================
;; GPU simulation step
;; ================================================================

(def ^:private step-sym->key
  {"U" :u1 "V" :v1 "U2" :u2 "V2" :v2})

(def ^:private step-sym->key-swap
  {"U" :u2 "V" :v2 "U2" :u1 "V2" :v1})

(def ^:private color-sym->key
  {"U" :u1 "V" :v1 "pixels" :pixels})

(def ^:private color-sym->key-swap
  {"U" :u2 "V" :v2 "pixels" :pixels})

(defn- build-scalar-args
  "Build scalar args vector in kernel parameter order from a name→value map.
   Infers :type from value class."
  [kernel-info scalar-map]
  (mapv (fn [sym]
          (let [v (get scalar-map (name sym))]
            (when (nil? v)
              (throw (ex-info (str "Missing scalar arg: " sym)
                              {:sym sym :available (keys scalar-map)})))
            {:type (cond (float? v) :float
                         (integer? v) :int
                         (instance? Double v) :double
                         :else :float)
             :value v}))
        (:scalar-params kernel-info)))

(defn simulate-frame!
  "Run multiple simulation steps and colorize."
  []
  (let [sess @gpu-state
        step-kernel (first (gpu/kernel sess :step))
        {:keys [steps-per-frame F k Du Dv]} @*state
        step-scalars (build-scalar-args step-kernel
                       {"Du" (float Du) "Dv" (float Dv) "F" (float F)
                        "k" (float k) "W" (int W) "H" (int H)})]
    ;; Ping-pong between u1/v1 and u2/v2
    (dotimes [i steps-per-frame]
      (let [sym->key (if (even? i) step-sym->key step-sym->key-swap)]
        (gpu/invoke! sess :step sym->key step-scalars N)))
    ;; Colorize from the last-written buffer
    (let [sym->key (if (even? steps-per-frame) color-sym->key color-sym->key-swap)]
      (gpu/invoke! sess :colorize sym->key [] N))))

;; ================================================================
;; JavaFX display — zero-copy via DeviceBuffer segment
;; ================================================================

(def byte-bgra-fmt (PixelFormat/getByteBgraInstance))
(defonce img (WritableImage. W H))
(defonce pw (.getPixelWriter img))

(defn copy-pixels-to-image!
  "Download pixel buffer and write to JavaFX WritableImage.
   On unified memory GPUs, the download is a no-op (shared memory)."
  []
  (let [sess @gpu-state
        pixel-buf (gpu/buffer sess :pixels)
        ;; Get the underlying MemorySegment for zero-copy on unified memory
        seg (:segment pixel-buf)
        bb (doto (.asByteBuffer seg)
             (.order (ByteOrder/nativeOrder))
             (.rewind))]
    (.setPixels pw 0 0 W H byte-bgra-fmt bb (* W 4))))

;; ================================================================
;; cljfx UI
;; ================================================================

(defn root-view [{:keys [F k Du Dv steps-per-frame fps preset running]}]
  {:fx/type :stage
   :showing true
   :title (format "Gray-Scott Reaction-Diffusion — %s — %.0f fps" preset fps)
   :on-close-request (fn [_] (swap! *state assoc :running false))
   :scene
   {:fx/type :scene
    :root
    {:fx/type :v-box
     :children
     [{:fx/type :image-view
       :image img
       :fit-width (double W)
       :fit-height (double H)
       :on-mouse-clicked {:event/type :canvas-click}}
      {:fx/type :h-box
       :padding 8 :spacing 10
       :children
       [{:fx/type :label :text "Feed F:"}
        {:fx/type :slider :min 0.01 :max 0.10 :value F
         :pref-width 150.0
         :on-value-changed {:event/type :F-changed}}
        {:fx/type :label :text (format "%.4f" F) :min-width 50.0}
        {:fx/type :label :text "  Kill k:"}
        {:fx/type :slider :min 0.03 :max 0.07 :value k
         :pref-width 150.0
         :on-value-changed {:event/type :k-changed}}
        {:fx/type :label :text (format "%.4f" k) :min-width 50.0}]}
      {:fx/type :h-box
       :padding 8 :spacing 10
       :children
       [{:fx/type :label :text "Du:"}
        {:fx/type :slider :min 0.05 :max 0.40 :value Du
         :pref-width 120.0
         :on-value-changed {:event/type :Du-changed}}
        {:fx/type :label :text (format "%.3f" Du) :min-width 45.0}
        {:fx/type :label :text "  Dv:"}
        {:fx/type :slider :min 0.01 :max 0.20 :value Dv
         :pref-width 120.0
         :on-value-changed {:event/type :Dv-changed}}
        {:fx/type :label :text (format "%.3f" Dv) :min-width 45.0}
        {:fx/type :label :text "  Steps/frame:"}
        {:fx/type :slider :min 1 :max 32 :value steps-per-frame
         :pref-width 100.0
         :block-increment 1.0
         :on-value-changed {:event/type :steps-changed}}
        {:fx/type :label :text (str (int steps-per-frame))}]}
      {:fx/type :h-box
       :padding 8 :spacing 10
       :children
       (into
         [{:fx/type :button :text "Reset"
           :on-action {:event/type :reset}}
          {:fx/type :button :text (if running "Pause" "Resume")
           :on-action {:event/type :toggle-pause}}
          {:fx/type :label :text "  Presets:"}]
         (mapv (fn [name]
                 {:fx/type :button :text name
                  :on-action {:event/type :preset :name name}})
               (sort (keys presets))))}]}}})

(defn event-handler [event]
  (case (:event/type event)
    :F-changed (swap! *state assoc :F (:fx/event event))
    :k-changed (swap! *state assoc :k (:fx/event event))
    :Du-changed (swap! *state assoc :Du (:fx/event event))
    :Dv-changed (swap! *state assoc :Dv (:fx/event event))
    :steps-changed (swap! *state assoc :steps-per-frame (int (:fx/event event)))
    :canvas-click (let [^MouseEvent e (:fx/event event)]
                    (seed-at! (int (.getX e)) (int (.getY e))))
    :reset (do (init-grid!) (swap! *state assoc :frame-count 0))
    :toggle-pause (swap! *state update :running not)
    :preset (let [{:keys [F k]} (get presets (:name event))]
              (init-grid!)
              (swap! *state assoc :F F :k k :preset (:name event) :frame-count 0))
    nil))

;; ================================================================
;; Main
;; ================================================================

(defn start! []
  (println "Compiling Gray-Scott kernels via raster.gpu.core...")
  (init-gpu!)
  (init-grid!)
  (println "Kernels compiled. Starting UI...")

  ;; Warm up: one step + colorize
  (simulate-frame!)

  (def renderer
    (fx/create-renderer
      :middleware (fx/wrap-map-desc root-view)
      :opts {:fx.opt/map-event-handler event-handler}))
  (fx/mount-renderer *state renderer)

  (def frame-counter (atom 0))
  (def last-fps-time (atom (System/nanoTime)))

  (def timer
    (proxy [AnimationTimer] []
      (handle [now]
        (when (:running @*state)
          (simulate-frame!)
          (copy-pixels-to-image!)
          ;; FPS counter
          (let [fc (swap! frame-counter inc)]
            (when (zero? (mod fc 30))
              (let [elapsed (/ (- now @last-fps-time) 1e9)
                    fps (/ 30.0 elapsed)]
                (reset! last-fps-time now)
                (swap! *state assoc :fps fps :frame-count fc))))))))

  (.start timer)
  (println "Gray-Scott reaction-diffusion running on GPU via raster.gpu.core.")
  (println "Click canvas to seed. Use sliders or presets to explore.")
  (println "To stop: (.stop demo.reaction-diffusion/timer)"))

(defn -main [& _args]
  (try
    (Platform/startup (fn []))
    (catch IllegalStateException _
      ;; Already initialized (e.g. from REPL)
      nil))
  (Platform/runLater #(start!)))
