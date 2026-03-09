(ns raster.runtime.display
  "Zero-copy GPU → display path for Raster.

  Uses Java AWT to render GPU compute output directly to screen without
  intermediate copies. GPU writes to shared MemorySegment; AWT reads from
  the same memory via a DataBufferInt wrapping an int[].

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
  - GPU via ze/alloc-shared wrapping or MemorySegment/ofArray
  - AWT via DataBufferInt (zero-copy)"
  [^long width ^long height]
  (let [n      (* width height)
        pixels (int-array n)
        seg    (MemorySegment/ofArray pixels)]
    (->RenderBuffer width height pixels seg)))

(defn render-buffer-gpu
  "Allocate a GPU-shared pixel buffer for width×height RGBA pixels.
  Allocates shared memory via Level Zero so GPU kernels can write directly.
  Returns a RenderBuffer backed by ze shared memory."
  [^long width ^long height]
  (let [n       (* width height)
        n-bytes (* n 4)
        ze-alloc (requiring-resolve 'raster.gpu.ze-runtime/alloc-shared)
        seg     (ze-alloc n-bytes)
        ;; Wrap shared MemorySegment as int[] for AWT access
        pixels  (int-array n)
        _       (.copyInto seg (MemorySegment/ofArray pixels) 0 0 n-bytes)]
    ;; We store the seg; pixels is a view copy for initial use.
    ;; For zero-copy from GPU, call display-from-seg! instead of display!
    (->RenderBuffer width height pixels seg)))

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

  For GPU-written pixels (via ze alloc-shared): call sync-from-gpu!
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
  Call before display! when the GPU has written to (:seg buf) via ze shared memory."
  [^RenderBuffer buf]
  (let [n-bytes (* (:width buf) (:height buf) 4)
        dst-seg (MemorySegment/ofArray (:pixels buf))]
    (MemorySegment/copy (:seg buf) 0 dst-seg 0 n-bytes)
    buf))

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
