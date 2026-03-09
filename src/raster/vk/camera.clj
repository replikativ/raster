(ns raster.vk.camera
  "FPS and orbital camera abstractions. Pure data — camera state is a map."
  (:require
   [raster.vk.math :as math]
   [raster.vk.input :as input])
  (:import
   [org.lwjgl.glfw GLFW]))

(set! *warn-on-reflection* true)

(defn create-fps-camera
  "Create an FPS camera at position [x y z] with yaw/pitch in radians."
  [[x y z] ^double yaw ^double pitch]
  {:pos [(double x) (double y) (double z)]
   :yaw yaw :pitch pitch
   :fov (/ Math/PI 2.0)
   :near 0.05 :far 200.0
   :aspect (/ 1024.0 768.0)})

(defn create-orbit-camera
  "Create an orbital camera looking at target."
  [[tx ty tz] ^double distance ^double angle ^double elevation]
  {:target [(double tx) (double ty) (double tz)]
   :distance distance :angle angle :elevation elevation
   :fov (/ Math/PI 3.0)
   :near 0.1 :far 500.0
   :aspect (/ 1024.0 768.0)})

(defn fps-camera-update
  "Update FPS camera from input. Returns updated camera."
  [cam input dt speed sensitivity]
  (let [dt (double dt) speed (double speed) sensitivity (double sensitivity)
        yaw (double (:yaw cam))
        pitch (double (:pitch cam))
        [mx my] (input/mouse-delta input)
        mx (double (or mx 0.0))
        my (double (or my 0.0))
        ;; Mouse look
        new-yaw (- yaw (* mx sensitivity dt))
        new-pitch (-> (+ pitch (* my sensitivity dt))
                      (max (- 0.02 (/ Math/PI 2.0)))
                      (min (- (/ Math/PI 2.0) 0.02)))
        ;; Forward/right vectors on XZ plane
        fx (Math/sin new-yaw)
        fz (- (Math/cos new-yaw))
        rx (Math/cos new-yaw)
        rz (Math/sin new-yaw)
        ;; WASD movement
        move-f (cond (input/key-down? input :w) 1.0
                     (input/key-down? input :s) -1.0
                     :else 0.0)
        move-r (cond (input/key-down? input :d) 1.0
                     (input/key-down? input :a) -1.0
                     :else 0.0)
        move-u (cond (input/key-down? input :space) 1.0
                     (input/key-down? input :left-shift) -1.0
                     :else 0.0)
        s (* speed dt)
        [px py pz] (:pos cam)
        px (double px) py (double py) pz (double pz)]
    (assoc cam
           :yaw new-yaw
           :pitch new-pitch
           :pos [(+ px (* fx move-f s) (* rx move-r s))
                 (+ py (* move-u s))
                 (+ pz (* fz move-f s) (* rz move-r s))])))

(defn orbit-camera-update
  "Update orbit camera from input. Returns updated camera."
  [cam input dt rot-speed zoom-speed]
  (let [dt (double dt) rot-speed (double rot-speed) zoom-speed (double zoom-speed)
        angle (double (:angle cam))
        elevation (double (:elevation cam))
        distance (double (:distance cam))
        da (cond (input/key-down? input :a) (- rot-speed)
                 (input/key-down? input :d) rot-speed
                 :else 0.0)
        de (cond (input/key-down? input :w) rot-speed
                 (input/key-down? input :s) (- rot-speed)
                 :else 0.0)
        dz (cond (input/key-down? input :q) zoom-speed
                 (input/key-down? input :e) (- zoom-speed)
                 :else 0.0)]
    (assoc cam
           :angle (+ angle (* da dt))
           :elevation (-> (+ elevation (* de dt))
                          (max 0.1)
                          (min (- Math/PI 0.1)))
           :distance (max 1.0 (+ distance (* dz dt))))))

(defn camera-view-matrix
  "Compute view matrix for an FPS camera."
  ^floats [cam]
  (let [[px py pz] (:pos cam)
        yaw (double (:yaw cam))
        pitch (double (:pitch cam))
        ;; Look direction from yaw/pitch
        cp (Math/cos pitch)
        fx (* (Math/sin yaw) cp)
        fy (Math/sin pitch)
        fz (* (- (Math/cos yaw)) cp)]
    (math/look-at [px py pz]
                  [(+ (double px) fx) (+ (double py) fy) (+ (double pz) fz)]
                  [0.0 1.0 0.0])))

(defn orbit-view-matrix
  "Compute view matrix for an orbit camera."
  ^floats [cam]
  (let [[tx ty tz] (:target cam)
        eye (math/orbit-eye (:angle cam) (:elevation cam) (:distance cam))]
    (math/look-at [(+ (double tx) (double (nth eye 0)))
                   (+ (double ty) (double (nth eye 1)))
                   (+ (double tz) (double (nth eye 2)))]
                  [tx ty tz]
                  [0.0 1.0 0.0])))

(defn camera-projection
  "Compute projection matrix for any camera."
  ^floats [cam]
  (math/perspective (:fov cam) (:aspect cam) (:near cam) (:far cam)))

(defn camera-vp-matrix
  "Compute combined view-projection matrix."
  ^floats [cam]
  (let [view (if (:target cam)
               (orbit-view-matrix cam)
               (camera-view-matrix cam))
        proj (camera-projection cam)]
    (math/mat4-mul proj view)))

(defn set-cursor-captured!
  "Enable/disable cursor capture (for FPS mouse look)."
  [^long window captured?]
  (GLFW/glfwSetInputMode window GLFW/GLFW_CURSOR
                         (if captured?
                           GLFW/GLFW_CURSOR_DISABLED
                           GLFW/GLFW_CURSOR_NORMAL)))
