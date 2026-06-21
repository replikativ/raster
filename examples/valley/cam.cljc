(ns valley.cam
  "Camera / projection helpers shared by the valley shells: a Vulkan-style perspective
   (Y-down, depth [0,1]), look-at, and `fly-mvp` (view-projection from a {:pos :yaw :pitch}
   camera). One source, both backends — the WGSL emitter flips Y so the same matrices render
   identically on Vulkan and WebGPU."
  (:require [clojure.string]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(defn- tan [x] (#?(:clj Math/tan :cljs js/Math.tan) x))
(defn- sqrt [x] (#?(:clj Math/sqrt :cljs js/Math.sqrt) x))

;; --- column-major 4x4 (m[col*4+row]) -----------------------------------------
(defn- mat-mul [a b]
  (vec (for [c (range 4) r (range 4)]
         (+ (* (nth a (+ r 0)) (nth b (+ (* c 4) 0)))
            (* (nth a (+ r 4)) (nth b (+ (* c 4) 1)))
            (* (nth a (+ r 8)) (nth b (+ (* c 4) 2)))
            (* (nth a (+ r 12)) (nth b (+ (* c 4) 3)))))))

(defn- perspective [fovy aspect zn zf]
  (let [f (/ 1.0 (tan (* 0.5 fovy)))]            ; Vulkan-style: Y-down, depth [0,1]
    [(/ f aspect) 0.0 0.0 0.0
     0.0 (- f) 0.0 0.0
     0.0 0.0 (/ zf (- zn zf)) -1.0
     0.0 0.0 (/ (* zn zf) (- zn zf)) 0.0]))

(defn- v- [a b] [(- (a 0) (b 0)) (- (a 1) (b 1)) (- (a 2) (b 2))])
(defn- cross [a b] [(- (* (a 1) (b 2)) (* (a 2) (b 1)))
                    (- (* (a 2) (b 0)) (* (a 0) (b 2)))
                    (- (* (a 0) (b 1)) (* (a 1) (b 0)))])
(defn- dot [a b] (+ (* (a 0) (b 0)) (* (a 1) (b 1)) (* (a 2) (b 2))))
(defn- norm [a] (let [l (sqrt (dot a a))] [(/ (a 0) l) (/ (a 1) l) (/ (a 2) l)]))

(defn- look-at [eye center up]
  (let [f (norm (v- center eye)) s (norm (cross f up)) u (cross s f)]
    [(s 0) (u 0) (- (f 0)) 0.0
     (s 1) (u 1) (- (f 1)) 0.0
     (s 2) (u 2) (- (f 2)) 0.0
     (- (dot s eye)) (- (dot u eye)) (dot f eye) 1.0]))

(defn- forward-vec [yaw pitch]
  (let [cp (cos pitch)] [(* (sin yaw) cp) (sin pitch) (* (- (cos yaw)) cp)]))

(defn fly-mvp
  "View-projection matrix (flat 16-float column-major) for a {:pos [x y z] :yaw :pitch} camera."
  [cam aspect]
  (let [{:keys [pos yaw pitch]} cam
        f (forward-vec yaw pitch)
        center [(+ (pos 0) (f 0)) (+ (pos 1) (f 1)) (+ (pos 2) (f 2))]]
    (mat-mul (perspective 0.9 aspect 0.1 400.0)
             (look-at pos center [0.0 1.0 0.0]))))
