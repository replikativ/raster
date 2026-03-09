(ns raster.vk.atlas
  "Texture atlas packer using shelf-packing algorithm.
  Composes multiple images into a single GPU texture with UV sub-rects."
  (:require
   [raster.vk.texture :as texture])
  (:import
   [java.nio ByteBuffer]
   [org.lwjgl.system MemoryUtil]))

(set! *warn-on-reflection* true)

(defn create-atlas-builder
  "Create a new atlas builder. width/height are atlas dimensions in pixels."
  [^long width ^long height]
  (atom {:width width
         :height height
         :shelves [] ;; [{:y :height :x}] — current shelves
         :entries {} ;; name → {:u0 :v0 :u1 :v1 :x :y :w :h}
         :pixels (byte-array (* width height 4))}))

(defn atlas-add!
  "Add an image to the atlas. Returns {:u0 :v0 :u1 :v1} UV rect.
  rgba-pixels is byte[] of RGBA data, w/h are image dimensions."
  [builder name rgba-pixels w h]
  (let [^bytes rgba-pixels rgba-pixels
        w (long w) h (long h)
        state @builder
        aw (long (:width state))
        ah (long (:height state))
        ^bytes atlas-pixels (:pixels state)
        shelves (:shelves state)
        ;; Find best shelf (shortest that fits width, with room)
        best-shelf-idx (reduce
                        (fn [best-idx i]
                          (let [shelf (nth shelves i)]
                            (if (and (<= (+ (long (:x shelf)) w) aw)
                                     (>= (long (:height shelf)) h)
                                     (or (nil? best-idx)
                                         (< (long (:height shelf))
                                            (long (:height (nth shelves (int best-idx)))))))
                              i
                              best-idx)))
                        nil
                        (range (count shelves)))
        ;; Place on existing shelf or create new one
        [px py new-shelves]
        (if best-shelf-idx
          (let [shelf (nth shelves (int best-shelf-idx))
                px (:x shelf)
                py (:y shelf)]
            [px py (update shelves (int best-shelf-idx) assoc :x (+ (long px) w))])
          ;; New shelf
          (let [next-y (if (empty? shelves)
                         0
                         (let [last-shelf (last shelves)]
                           (+ (long (:y last-shelf)) (long (:height last-shelf)))))
                next-y (long next-y)]
            (when (> (+ next-y h) ah)
              (throw (ex-info "Atlas full" {:name name :w w :h h
                                            :next-y next-y :atlas-height ah})))
            [0 next-y (conj shelves {:y next-y :height h :x w})]))
        px (long px)
        py (long py)]
    ;; Blit pixels into atlas
    (dotimes [row h]
      (let [src-offset (* row w 4)
            dst-offset (* (+ (* (+ py row) aw) px) 4)]
        (System/arraycopy rgba-pixels src-offset atlas-pixels dst-offset (* w 4))))
    ;; Record UV coordinates
    (let [u0 (/ (double px) (double aw))
          v0 (/ (double py) (double ah))
          u1 (/ (double (+ px w)) (double aw))
          v1 (/ (double (+ py h)) (double ah))
          entry {:u0 u0 :v0 v0 :u1 u1 :v1 v1 :x px :y py :w w :h h}]
      (swap! builder assoc
             :shelves new-shelves
             :entries (assoc (:entries @builder) name entry))
      entry)))

(defn atlas-build!
  "Finalize the atlas and upload to GPU.
  Returns {:texture Texture :entries {name → {:u0 :v0 :u1 :v1}} :desc-set long}."
  [ctx builder ^long tex-pool ^long tex-layout]
  (let [state @builder
        aw (long (:width state))
        ah (long (:height state))
        ^bytes pixels (:pixels state)
        ;; Create ByteBuffer for texture upload
        pixel-buf (MemoryUtil/memAlloc (int (* aw ah 4)))]
    (.put pixel-buf pixels)
    (.flip pixel-buf)
    (let [tex (texture/create-texture ctx
                                      {:pixels pixel-buf :width aw :height ah :channels 4}
                                      :filter-mode :nearest
                                      :address-mode :clamp)
          desc-set (texture/allocate-texture-descriptor-set
                    (:device ctx) tex-pool tex-layout tex)]
      (MemoryUtil/memFree pixel-buf)
      {:texture tex
       :entries (:entries state)
       :desc-set desc-set})))

(defn atlas-uv
  "Look up UV rect for a named texture in the atlas."
  [atlas ^String name]
  (get (:entries atlas) name))

(defn destroy-atlas!
  "Destroy the atlas GPU texture."
  [ctx atlas]
  (texture/destroy-texture! (:device ctx) (:allocator ctx) (:texture atlas)))
