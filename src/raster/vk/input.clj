(ns raster.vk.input
  "Input system: GLFW callbacks -> event queue atom -> per-frame state diffing."
  (:import
   [org.lwjgl.glfw GLFW GLFWKeyCallback GLFWCursorPosCallback
    GLFWMouseButtonCallback GLFWScrollCallback]))

(set! *warn-on-reflection* true)

;; ================================================================
;; GLFW key code -> keyword mapping
;; ================================================================

(def ^:private key-map
  {GLFW/GLFW_KEY_W :w, GLFW/GLFW_KEY_A :a, GLFW/GLFW_KEY_S :s, GLFW/GLFW_KEY_D :d
   GLFW/GLFW_KEY_Q :q, GLFW/GLFW_KEY_E :e, GLFW/GLFW_KEY_R :r, GLFW/GLFW_KEY_F :f
   GLFW/GLFW_KEY_Z :z, GLFW/GLFW_KEY_X :x, GLFW/GLFW_KEY_C :c, GLFW/GLFW_KEY_V :v
   GLFW/GLFW_KEY_SPACE :space, GLFW/GLFW_KEY_ESCAPE :escape
   GLFW/GLFW_KEY_LEFT_SHIFT :left-shift, GLFW/GLFW_KEY_RIGHT_SHIFT :right-shift
   GLFW/GLFW_KEY_LEFT_CONTROL :left-control, GLFW/GLFW_KEY_RIGHT_CONTROL :right-control
   GLFW/GLFW_KEY_LEFT_ALT :left-alt, GLFW/GLFW_KEY_RIGHT_ALT :right-alt
   GLFW/GLFW_KEY_TAB :tab, GLFW/GLFW_KEY_ENTER :enter, GLFW/GLFW_KEY_BACKSPACE :backspace
   GLFW/GLFW_KEY_UP :up, GLFW/GLFW_KEY_DOWN :down, GLFW/GLFW_KEY_LEFT :left, GLFW/GLFW_KEY_RIGHT :right
   GLFW/GLFW_KEY_F1 :f1, GLFW/GLFW_KEY_F2 :f2, GLFW/GLFW_KEY_F3 :f3, GLFW/GLFW_KEY_F4 :f4
   GLFW/GLFW_KEY_F5 :f5, GLFW/GLFW_KEY_F6 :f6, GLFW/GLFW_KEY_F7 :f7, GLFW/GLFW_KEY_F8 :f8
   GLFW/GLFW_KEY_F9 :f9, GLFW/GLFW_KEY_F10 :f10, GLFW/GLFW_KEY_F11 :f11, GLFW/GLFW_KEY_F12 :f12
   GLFW/GLFW_KEY_0 :0, GLFW/GLFW_KEY_1 :1, GLFW/GLFW_KEY_2 :2, GLFW/GLFW_KEY_3 :3
   GLFW/GLFW_KEY_4 :4, GLFW/GLFW_KEY_5 :5, GLFW/GLFW_KEY_6 :6, GLFW/GLFW_KEY_7 :7
   GLFW/GLFW_KEY_8 :8, GLFW/GLFW_KEY_9 :9
   GLFW/GLFW_KEY_MINUS :minus, GLFW/GLFW_KEY_EQUAL :equal
   GLFW/GLFW_KEY_LEFT_BRACKET :left-bracket, GLFW/GLFW_KEY_RIGHT_BRACKET :right-bracket
   GLFW/GLFW_KEY_GRAVE_ACCENT :grave})

(def ^:private button-map
  {GLFW/GLFW_MOUSE_BUTTON_LEFT :left
   GLFW/GLFW_MOUSE_BUTTON_RIGHT :right
   GLFW/GLFW_MOUSE_BUTTON_MIDDLE :middle})

;; ================================================================
;; Input system
;; ================================================================

(def ^:private empty-state
  {:keys-down #{}
   :keys-pressed #{}
   :keys-released #{}
   :mouse-pos [0.0 0.0]
   :mouse-delta [0.0 0.0]
   :mouse-buttons #{}
   :mouse-pressed #{}
   :mouse-released #{}
   :scroll-delta [0.0 0.0]})

(defrecord InputSystem [state events window])

(defn create-input-system
  "Install GLFW callbacks. Returns InputSystem."
  [^long window]
  (let [state (atom empty-state)
        events (atom [])]

    ;; Key callback
    (GLFW/glfwSetKeyCallback window
                             (proxy [GLFWKeyCallback] []
                               (invoke [_win key _scancode action _mods]
                                 (when-let [k (get key-map key)]
                                   (cond
                                     (= action GLFW/GLFW_PRESS)
                                     (swap! events conj [:key-down k])
                                     (= action GLFW/GLFW_RELEASE)
                                     (swap! events conj [:key-up k]))))))

    ;; Cursor position callback
    (GLFW/glfwSetCursorPosCallback window
                                   (proxy [GLFWCursorPosCallback] []
                                     (invoke [_win x y]
                                       (swap! events conj [:mouse-move x y]))))

    ;; Mouse button callback
    (GLFW/glfwSetMouseButtonCallback window
                                     (proxy [GLFWMouseButtonCallback] []
                                       (invoke [_win button action _mods]
                                         (when-let [b (get button-map button)]
                                           (cond
                                             (= action GLFW/GLFW_PRESS)
                                             (swap! events conj [:mouse-down b])
                                             (= action GLFW/GLFW_RELEASE)
                                             (swap! events conj [:mouse-up b]))))))

    ;; Scroll callback
    (GLFW/glfwSetScrollCallback window
                                (proxy [GLFWScrollCallback] []
                                  (invoke [_win xoff yoff]
                                    (swap! events conj [:scroll xoff yoff]))))

    (->InputSystem state events window)))

(defn poll-input!
  "Call after glfwPollEvents. Drains events, computes diffs. Returns updated InputState."
  [^InputSystem input-sys]
  (let [input-atom (:state input-sys)
        events-atom (:events input-sys)
        evts (let [es @events-atom] (reset! events-atom []) es)
        prev @input-atom
        prev-pos (:mouse-pos prev)]
    (loop [evts evts
           keys-down (:keys-down prev)
           pressed #{}
           released #{}
           mouse-pos prev-pos
           mouse-buttons (:mouse-buttons prev)
           mouse-pressed #{}
           mouse-released #{}
           scroll-x 0.0
           scroll-y 0.0]
      (if (seq evts)
        (let [[tag & args] (first evts)
              rest-evts (rest evts)]
          (case tag
            :key-down
            (recur rest-evts (conj keys-down (first args))
                   (conj pressed (first args)) released
                   mouse-pos mouse-buttons mouse-pressed mouse-released scroll-x scroll-y)
            :key-up
            (recur rest-evts (disj keys-down (first args))
                   pressed (conj released (first args))
                   mouse-pos mouse-buttons mouse-pressed mouse-released scroll-x scroll-y)
            :mouse-move
            (recur rest-evts keys-down pressed released
                   [(first args) (second args)]
                   mouse-buttons mouse-pressed mouse-released scroll-x scroll-y)
            :mouse-down
            (recur rest-evts keys-down pressed released mouse-pos
                   (conj mouse-buttons (first args))
                   (conj mouse-pressed (first args)) mouse-released scroll-x scroll-y)
            :mouse-up
            (recur rest-evts keys-down pressed released mouse-pos
                   (disj mouse-buttons (first args))
                   mouse-pressed (conj mouse-released (first args)) scroll-x scroll-y)
            :scroll
            (recur rest-evts keys-down pressed released mouse-pos
                   mouse-buttons mouse-pressed mouse-released
                   (+ scroll-x (double (first args))) (+ scroll-y (double (second args))))))
        (let [dx (- (double (nth mouse-pos 0)) (double (nth prev-pos 0)))
              dy (- (double (nth mouse-pos 1)) (double (nth prev-pos 1)))
              new-state {:keys-down keys-down
                         :keys-pressed pressed
                         :keys-released released
                         :mouse-pos mouse-pos
                         :mouse-delta [dx dy]
                         :mouse-buttons mouse-buttons
                         :mouse-pressed mouse-pressed
                         :mouse-released mouse-released
                         :scroll-delta [scroll-x scroll-y]}]
          (reset! input-atom new-state)
          new-state)))))

;; ================================================================
;; Query helpers
;; ================================================================

(defn key-down?
  "Is key held this frame?"
  [state key]
  (contains? (:keys-down state) key))

(defn key-pressed?
  "Edge-triggered: was key pressed THIS frame?"
  [state key]
  (contains? (:keys-pressed state) key))

(defn key-released?
  "Edge-triggered: was key released THIS frame?"
  [state key]
  (contains? (:keys-released state) key))

(defn mouse-pos
  "Returns [x y]."
  [state]
  (:mouse-pos state))

(defn mouse-delta
  "Returns [dx dy] since last frame."
  [state]
  (:mouse-delta state))

(defn mouse-down?
  "Is mouse button held? button: :left :right :middle"
  [state button]
  (contains? (:mouse-buttons state) button))

(defn mouse-pressed?
  "Edge-triggered: was mouse button pressed THIS frame?"
  [state button]
  (contains? (:mouse-pressed state) button))

(defn scroll-delta
  "Returns [dx dy] scroll delta."
  [state]
  (:scroll-delta state))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-input-system!
  "Remove GLFW callbacks."
  [^InputSystem input-sys]
  (let [window (long (:window input-sys))]
    (GLFW/glfwSetKeyCallback window nil)
    (GLFW/glfwSetCursorPosCallback window nil)
    (GLFW/glfwSetMouseButtonCallback window nil)
    (GLFW/glfwSetScrollCallback window nil)))
