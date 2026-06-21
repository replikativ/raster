(ns valley.ui
  "Cross-platform 2D HUD overlay (crosshair, health hearts, hotbar) drawn after the world
   with depth off + alpha blend. Geometry is authored directly in clip space in Vulkan's
   y-down convention; the WGSL emitter's y-flip makes it render identically in the browser.
   Icons live in the shared valley.tex atlas (layers 33..37) + block top-faces for hotbar
   items, so the HUD reuses the same texture-array binding as the terrain/mobs. One shader
   → GLSL + WGSL."
  (:require [valley.tex :as tex]))

(def ^:const STRIDE 20)   ; pos2(8) + uv2(8) + layer(4)

;; one textured quad (two triangles) centred at clip (cx,cy), half-extents (hw,hh)
(defn- quad! [out cx cy hw hh layer]
  (let [x0 (- cx hw) x1 (+ cx hw) y0 (- cy hh) y1 (+ cy hh) L (double layer)
        push (fn [t px py u v] (-> t (conj! px) (conj! py) (conj! u) (conj! v) (conj! L)))]
    (vswap! out (fn [t] (-> t (push x0 y0 0.0 0.0) (push x1 y0 1.0 0.0) (push x1 y1 1.0 1.0)
                            (push x0 y0 0.0 0.0) (push x1 y1 1.0 1.0) (push x0 y1 0.0 1.0))))))

(defn verts
  "Flat float seq (triangle list, pos2 uv2 layer) for the HUD given player state + aspect.
   state = {:health 0..max :max-health :hotbar [block-ids] :sel idx}. Squares stay square
   by scaling x by 1/aspect (clip x spans the wider screen dimension)."
  [{:keys [health max-health hotbar sel] :or {health 20 max-health 20 hotbar [] sel 0}} aspect]
  (let [out (volatile! (transient []))
        ax  (/ 1.0 (double aspect))
        hh  0.5]                                   ; unused; per-element sizes below
    ;; crosshair (centre)
    (quad! out 0.0 0.0 (* 0.022 ax) 0.022 tex/CROSSHAIR)
    ;; hearts row (bottom; clip y-down so +y is lower on screen)
    (let [n (quot (long max-health) 2) s 0.026 sx (* 0.058 ax) y 0.80
          x0 (* (- (/ (dec n) 2.0)) sx)]
      (dotimes [i n]
        (let [lvl (- (long health) (* 2 i))            ; HP into heart i: >=2 full, 1 half, else empty
              icon (cond (>= lvl 2) tex/HEART-FULL (= lvl 1) tex/HEART-HALF :else tex/HEART-EMPTY)
              cx (+ x0 (* i sx))]
          (quad! out cx y (* s ax) s icon))))
    ;; hotbar (below hearts)
    (let [n (count hotbar) s 0.05 sx (* 0.112 ax) y 0.90
          x0 (* (- (/ (dec (max 1 n)) 2.0)) sx)]
      (dotimes [i n]
        (let [cx (+ x0 (* i sx))]
          (quad! out cx y (* s ax) s tex/SLOT)                       ; slot bg
          (quad! out cx y (* 0.038 ax) 0.038 (tex/face-layer (long (nth hotbar i)) 0))  ; block top-face icon
          (when (= i (long sel)) (quad! out cx y (* s ax) s tex/SELECTOR))))) ; highlight
    (persistent! @out)))

(def shader
  '{:uniform        {:name "H" :fields [[:tint :vec4]]}
    :attributes     [[inPos vec2 0] [inUv vec2 1] [inLayer float 2]]
    :varyings       [[vUv vec2 0] [vLayer float 1]]
    :texture-arrays [[atlas 0]]
    :vertex         [(set-position (vec4 inPos 0.0 1.0)) (out vUv inUv) (out vLayer inLayer)]
    :fragment       [(color (* (sample-layer atlas vUv vLayer) tint))]})

(def pipeline-spec
  {:shader shader :topology :triangles :depth? false :cull :none :blend? true
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float2 :offset 0}
                                        {:location 1 :format :float2 :offset 8}
                                        {:location 2 :format :float  :offset 16}]}})
