(ns raster.runtime.display
  "CPU display and explicitly owned GPU pixel storage for Raster.

  Java AWT wraps the CPU int[] without copying. GPU storage is a separate owned
  shared-memory DeviceBuffer; sync-from-gpu! copies completed pixels into that int[].
  GPU callers must establish completion before synchronization or close, and call
  close-render-buffer! when the allocation is no longer borrowed by a kernel.

  Usage:
    (def screen (render-buffer 800 600))

    ;; GPU or CPU fills the pixel buffer:
    ;; (par/map-void! i (* 800 600)
    ;;   (aset screen i (pack-rgba r g b 1.0)))

    (def frame (show-frame! 800 600 \"My Window\"))
    (display! screen frame)     ;; zero-copy paint

  For ~60fps animation:
    (loop []
      (render-something! screen)
      (display! screen frame)
      (Thread/sleep 16)
      (recur))"
  (:require [raster.gpu.resource-cleanup :as cleanup])
  (:import [java.awt Canvas Frame Graphics]
           [java.awt.image BufferedImage DataBufferInt]
           [java.lang.foreign MemorySegment ValueLayout]
           [java.awt.event WindowAdapter WindowEvent]))

;; ================================================================
;; RenderBuffer — shared GPU/CPU pixel buffer (RGBA8, packed as ARGB int)
;; ================================================================

(defrecord RenderBuffer
           [^long   width
            ^long   height
            ^ints   pixels      ;; int[] — writable by CPU; wrap for AWT
            ^MemorySegment seg  ;; shared MemorySegment over the same backing int[]
            ])

(defn render-buffer
  "Allocate a shared CPU/GPU pixel buffer for width×height RGBA pixels.
  Returns a RenderBuffer. The backing int[] is directly usable by:
  - CPU code via (:pixels buf)
  - Host transfer staging via MemorySegment/ofArray
  - AWT via DataBufferInt (zero-copy)"
  [^long width ^long height]
  (let [n      (* width height)
        pixels (int-array n)
        seg    (MemorySegment/ofArray pixels)]
    (->RenderBuffer width height pixels seg)))

(defn render-buffer-gpu
  "Allocate a GPU-shared pixel buffer for width×height RGBA pixels.
  Allocates shared memory via Level Zero so GPU kernels can write directly.
  Returns a RenderBuffer with :device-buffer holding the canonical allocation and
  :seg a borrowed pointer into it. Close with close-render-buffer! after GPU completion.
  Display uses a separate CPU pixel array, not a zero-copy GPU-to-AWT view."
  [^long width ^long height]
  (when (or (neg? width) (neg? height))
    (throw (ex-info "Render dimensions must be non-negative"
                    {:reason :invalid-render-dimensions :width width :height height})))
  (let [n (Math/multiplyExact width height)
        pixels (int-array (Math/toIntExact n))
        make-buffer (requiring-resolve 'raster.gpu.ze-runtime/make-buffer)
        child (volatile! nil)
        owner (cleanup/owner [{:id :pixel-allocation
                               :release #(when-let [buffer-owner @child]
                                           (cleanup/release! buffer-owner))}])
        retain! #(vreset! child %)]
    (cleanup/build!
     owner
     (fn []
       (let [buffer (make-buffer n :int {:retain-owner! retain! :adopt-cleanup! retain!})
             segment (:segment buffer)]
         (when-not (and (::cleanup/owner buffer)
                        (identical? @child (::cleanup/owner buffer)))
           (throw (ex-info "Render allocation has no retained canonical owner"
                           {:reason :missing-cleanup-owner})))
         ;; Shared memory is host-visible. Initialize it deterministically before publication.
         (MemorySegment/copy (MemorySegment/ofArray pixels) 0 segment 0 (* n 4))
         (assoc (->RenderBuffer width height pixels segment) :device-buffer buffer)))
     nil)))

(defn close-render-buffer!
  "Retire an owned GPU RenderBuffer exactly once, after callers establish GPU completion.
  CPU render buffers own no native resource; closing one is a no-op. Unknown native teardown
  remains retained by the canonical cleanup owner; this function does not cancel GPU work."
  [^RenderBuffer buf]
  (when-let [owner (::cleanup/owner buf)] (cleanup/release! owner))
  nil)

;; ================================================================
;; Pixel packing helpers
;; ================================================================

(defn pack-rgba
  "Pack r,g,b,a floats (0.0–1.0) into a packed ARGB int.
  Returns int suitable for aset into a RenderBuffer's pixels array."
  ^long [^double r ^double g ^double b ^double a]
  (bit-or (bit-shift-left (long (* 255.0 (max 0.0 (min 1.0 a)))) 24)
          (bit-shift-left (long (* 255.0 (max 0.0 (min 1.0 r)))) 16)
          (bit-shift-left (long (* 255.0 (max 0.0 (min 1.0 g)))) 8)
          (long (* 255.0 (max 0.0 (min 1.0 b))))))

(defn pack-rgb
  "Pack r,g,b floats (0.0–1.0) into a packed ARGB int with full alpha."
  ^long [^double r ^double g ^double b]
  (pack-rgba r g b 1.0))

;; ================================================================
;; AWT display
;; ================================================================

(defn show-frame!
  "Create and show an AWT Frame with the given dimensions.
  Returns the Frame. Call display! to paint pixels into it."
  ([^long width ^long height]
   (show-frame! width height "Raster"))
  ([^long width ^long height ^String title]
   (let [frame (Frame. title)
         canvas (Canvas.)]
     (.setSize canvas (int width) (int height))
     (.add frame canvas)
     (.pack frame)
     (.addWindowListener frame
                         (proxy [WindowAdapter] []
                           (windowClosing [^WindowEvent e]
                             (.setVisible frame false))))
     (.setVisible frame true)
     frame)))

(defn display!
  "Paint a RenderBuffer into a Frame. Zero-copy on the CPU path:
  the int[] backing the RenderBuffer is wrapped as a DataBufferInt
  and used directly as the BufferedImage's raster data.

  For GPU-written pixels in a render-buffer-gpu allocation: call sync-from-gpu!
  first to copy from MemorySegment into the pixels int[].

  Returns nil."
  [^RenderBuffer buf ^Frame frame]
  (let [w      (int (:width buf))
        h      (int (:height buf))
        pixels (:pixels buf)
        ;; ARGB bitmasks (alpha mask 0xFF000000 overflows int, need unchecked)
        masks  (let [a (int-array 4)]
                 (aset a 0 (unchecked-int 0x00FF0000))
                 (aset a 1 (unchecked-int 0x0000FF00))
                 (aset a 2 (unchecked-int 0x000000FF))
                 (aset a 3 (unchecked-int 0xFF000000))
                 a)
        ;; Wrap int[] as BufferedImage — zero copy
        db     (DataBufferInt. pixels (* w h))
        img    (BufferedImage.
                (java.awt.image.DirectColorModel.
                 32
                 (unchecked-int 0x00FF0000)
                 (unchecked-int 0x0000FF00)
                 (unchecked-int 0x000000FF)
                 (unchecked-int 0xFF000000))
                (java.awt.image.WritableRaster/createPackedRaster
                 db w h w masks nil)
                false nil)
        ^Canvas canvas (first (seq (.getComponents frame)))
        ^Graphics g    (.getGraphics canvas)]
    (when g
      (.drawImage g img 0 0 nil)
      (.dispose g))
    nil))

(defn sync-from-gpu!
  "Copy GPU-written pixels from RenderBuffer's MemorySegment into its int[] array.
  Call after GPU completion and before display!. This function does not wait for a kernel."
  [^RenderBuffer buf]
  (let [copy! #(let [n-bytes (* (:width buf) (:height buf) 4)
                     dst-seg (MemorySegment/ofArray (:pixels buf))]
                 (MemorySegment/copy (:seg buf) 0 dst-seg 0 n-bytes)
                 buf)]
    (if-let [owner (::cleanup/owner buf)]
      (locking (:state owner)
        (cleanup/assert-live! owner)
        (cleanup/assert-live! (::cleanup/owner (:device-buffer buf)))
        (copy!))
      (copy!))))

;; ================================================================
;; Convenience: animate loop
;; ================================================================

(defn animate!
  "Run an animation loop calling render-fn each frame.
  render-fn: (fn [^RenderBuffer buf frame-count]) — fills buf's pixels
  Returns nil when the loop is stopped (frame closed or interrupt)."
  [^RenderBuffer buf ^Frame frame render-fn & {:keys [fps] :or {fps 60}}]
  (let [delay-ms (long (/ 1000 fps))]
    (try
      (loop [i 0]
        (when (.isVisible frame)
          (render-fn buf i)
          (display! buf frame)
          (Thread/sleep delay-ms)
          (recur (inc i))))
      (catch InterruptedException _
        nil))))
