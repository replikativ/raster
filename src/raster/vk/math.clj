(ns raster.vk.math
  "Shared matrix utilities for 3D rendering.
  All matrices are float[16] in column-major order (OpenGL/Vulkan convention).")

(set! *warn-on-reflection* true)

(defn perspective
  "4x4 perspective projection. Y-flipped, depth [0,1] for Vulkan."
  ^floats [^double fov-y ^double aspect ^double near ^double far]
  (let [f (/ 1.0 (Math/tan (* fov-y 0.5)))
        nf (/ 1.0 (- near far))]
    (float-array
     [(/ f aspect) 0 0 0,
      0 (- f) 0 0,
      0 0 (* far nf) -1,
      0 0 (* near far nf) 0])))

(defn mat4-mul
  "Multiply two 4x4 column-major matrices."
  ^floats [^floats a ^floats b]
  (let [out (float-array 16)]
    (dotimes [col 4]
      (dotimes [row 4]
        (let [idx (+ (* col 4) row)]
          (aset out idx
                (float (+ (* (aget a row) (aget b (* col 4)))
                          (* (aget a (+ row 4)) (aget b (+ (* col 4) 1)))
                          (* (aget a (+ row 8)) (aget b (+ (* col 4) 2)))
                          (* (aget a (+ row 12)) (aget b (+ (* col 4) 3)))))))))
    out))

(defn look-at
  "Look-at view matrix (column-major). eye/target/up are [x y z] vectors."
  ^floats [eye target up]
  (let [ex (double (nth eye 0)) ey (double (nth eye 1)) ez (double (nth eye 2))
        tx (double (nth target 0)) ty (double (nth target 1)) tz (double (nth target 2))
        ux (double (nth up 0)) uy (double (nth up 1)) uz (double (nth up 2))
        ;; forward = normalize(eye - target)
        fx (- ex tx) fy (- ey ty) fz (- ez tz)
        fl (Math/sqrt (+ (* fx fx) (* fy fy) (* fz fz)))
        fx (/ fx fl) fy (/ fy fl) fz (/ fz fl)
        ;; right = normalize(cross(up, forward))
        rx (- (* uy fz) (* uz fy))
        ry (- (* uz fx) (* ux fz))
        rz (- (* ux fy) (* uy fx))
        rl (Math/sqrt (+ (* rx rx) (* ry ry) (* rz rz)))
        rx (/ rx rl) ry (/ ry rl) rz (/ rz rl)
        ;; true up = cross(forward, right)
        upx (- (* fy rz) (* fz ry))
        upy (- (* fz rx) (* fx rz))
        upz (- (* fx ry) (* fy rx))
        ;; translation: dot(axis, -eye)
        tx2 (- (+ (* rx ex) (* ry ey) (* rz ez)))
        ty2 (- (+ (* upx ex) (* upy ey) (* upz ez)))
        tz2 (- (+ (* fx ex) (* fy ey) (* fz ez)))]
    (float-array
     [rx   upx  fx  0,
      ry   upy  fy  0,
      rz   upz  fz  0,
      tx2  ty2  tz2 1])))

(defn orbit-eye
  "Compute eye position for orbit camera."
  [^double angle ^double elevation ^double radius]
  [(* radius (Math/cos elevation) (Math/sin angle))
   (* radius (Math/sin elevation))
   (* radius (Math/cos elevation) (Math/cos angle))])

(defn rotation-y
  "4x4 rotation around Y axis."
  ^floats [^double angle]
  (let [c (Math/cos angle) s (Math/sin angle)]
    (float-array
     [c 0 (- s) 0,
      0 1 0 0,
      s 0 c 0,
      0 0 0 1])))

(defn rotation-x
  "4x4 rotation around X axis."
  ^floats [^double angle]
  (let [c (Math/cos angle) s (Math/sin angle)]
    (float-array
     [1 0 0 0,
      0 c s 0,
      0 (- s) c 0,
      0 0 0 1])))

(defn rotation-z
  "4x4 rotation around Z axis."
  ^floats [^double angle]
  (let [c (Math/cos angle) s (Math/sin angle)]
    (float-array
     [c s 0 0,
      (- s) c 0 0,
      0 0 1 0,
      0 0 0 1])))

(defn translation
  "4x4 translation matrix."
  ^floats [^double x ^double y ^double z]
  (float-array
   [1 0 0 0,
    0 1 0 0,
    0 0 1 0,
    x y z 1]))

(defn scale
  "4x4 scale matrix."
  ^floats [^double sx ^double sy ^double sz]
  (float-array
   [sx 0 0 0,
    0 sy 0 0,
    0 0 sz 0,
    0 0 0 1]))

(defn model-matrix
  "Compose a model matrix from position [x y z], rotation angles [rx ry rz], and uniform scale."
  ^floats [position rotation ^double s]
  (let [[px py pz] position
        [rx ry rz] rotation]
    (-> (scale s s s)
        (mat4-mul (rotation-x rx))
        (mat4-mul (rotation-y ry))
        (mat4-mul (rotation-z rz))
        (mat4-mul (translation px py pz)))))

(defn identity-mat4
  "4x4 identity matrix."
  ^floats []
  (float-array
   [1 0 0 0,
    0 1 0 0,
    0 0 1 0,
    0 0 0 1]))

(defn ortho
  "4x4 orthographic projection (column-major). Y-flipped, depth [0,1] for Vulkan."
  ^floats [left right bottom top near far]
  (let [left (double left) right (double right) bottom (double bottom)
        top (double top) near (double near) far (double far)
        rl (/ 1.0 (- right left))
        tb (/ 1.0 (- top bottom))
        fn (/ 1.0 (- far near))]
    (float-array
     [(* 2.0 rl) 0 0 0,
      0 (* -2.0 tb) 0 0,       ;; flip Y for Vulkan
      0 0 fn 0,                 ;; Vulkan depth [0,1], not OpenGL [-1,1]
      (- (* (+ right left) rl)) (* (+ top bottom) tb) (- (* near fn)) 1])))
