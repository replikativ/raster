(ns asteroids.jvm
  "JVM platform shell for cross-platform Geometric Asteroids — a Swing/Java2D
   window running the SAME asteroids.game (.cljc) as the browser build. Here the
   physics kernel (asteroids.kernels/move-shape) runs as JVM bytecode; in the
   browser the identical game runs it as wasm. No native deps (AWT/Swing ships
   with the JDK).

   Run:  clojure -Sdeps '{:paths [\"src\" \"examples\"]}' -M -m asteroids.jvm"
  (:require [asteroids.game :as game])
  (:import [javax.swing JFrame JPanel Timer]
           [java.awt Color Graphics2D RenderingHints BasicStroke Font]
           [java.awt.event KeyListener KeyEvent ActionListener]
           [java.awt.geom GeneralPath Ellipse2D$Double]))

(def W 1024) (def H 768)
(def palette {3 (Color. 148 103 189) 4 (Color. 214 39 40) 5 (Color. 44 160 44)
              6 (Color. 255 127 14) 8 (Color. 31 119 180) 7 (Color. 127 127 127) 10 (Color. 23 190 207)})

(defn- poly-path ^GeneralPath [x y r sides angle]
  (let [p (GeneralPath.) tau (* 2.0 Math/PI)]
    (dotimes [k (inc (int sides))]
      (let [a (+ angle (/ (* tau k) sides) (- (/ Math/PI 2.0)))
            px (+ x (* r (Math/cos a))) py (+ y (* r (Math/sin a)))]
        (if (zero? k) (.moveTo p px py) (.lineTo p px py))))
    p))

(defn- render [^Graphics2D g rd]
  (.setRenderingHint g RenderingHints/KEY_ANTIALIASING RenderingHints/VALUE_ANTIALIAS_ON)
  (.setColor g Color/BLACK) (.fillRect g 0 0 W H)
  (.setStroke g (BasicStroke. 1.6))
  (doseq [p (:polys rd)]
    (.setColor g (get palette (:sides p) Color/LIGHT_GRAY))
    (.draw g (poly-path (:x p) (:y p) (:size p) (:sides p) (:angle p))))
  (let [s (:ship rd)]
    (.setColor g (Color. 210 210 210))
    (.draw g (poly-path (:x s) (:y s) 12.0 3 (+ (:angle s) (/ Math/PI 2.0)))))
  (.setColor g Color/WHITE)
  (doseq [b (:bullets rd)] (.fill g (Ellipse2D$Double. (- (:x b) 2.2) (- (:y b) 2.2) 4.4 4.4)))
  (let [{:keys [score lives wave over]} (:hud rd)]
    (.setColor g (Color. 200 200 200)) (.setFont g (Font. "Serif" Font/PLAIN 18))
    (.drawString g (str "SCORE " score "    LIVES " lives "    WAVE " wave
                        (when over "    —  GAME OVER (space)")) 16 26)))

(defn -main [& _]
  (let [input (atom #{})
        st    (atom (game/init-state))
        last  (atom (System/nanoTime))
        panel (proxy [JPanel] []
                (paintComponent [^Graphics2D g]
                  (proxy-super paintComponent g)
                  (render g (game/render-data @st))))
        code->action {KeyEvent/VK_LEFT :left KeyEvent/VK_RIGHT :right
                      KeyEvent/VK_UP :up KeyEvent/VK_SPACE :fire}]
    (doto panel
      (.setPreferredSize (java.awt.Dimension. W H))
      (.setFocusable true)
      (.addKeyListener
       (reify KeyListener
         (keyPressed [_ e] (when-let [a (code->action (.getKeyCode e))] (swap! input conj a)))
         (keyReleased [_ e] (when-let [a (code->action (.getKeyCode e))] (swap! input disj a)))
         (keyTyped [_ _]))))
    (doto (JFrame. "Geometric Asteroids — JVM (raster deftm bytecode)")
      (.setDefaultCloseOperation JFrame/EXIT_ON_CLOSE)
      (.setContentPane panel) (.pack) (.setLocationRelativeTo nil) (.setVisible true))
    (.requestFocusInWindow panel)
    (.start (Timer. 16 (reify ActionListener
                         (actionPerformed [_ _]
                           (let [now (System/nanoTime)
                                 dt (min 0.05 (/ (- now @last) 1.0e9))]
                             (reset! last now)
                             (swap! st game/update-game @input dt)
                             (.repaint panel))))))))
