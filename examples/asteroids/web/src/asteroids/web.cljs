(ns asteroids.web
  "Browser platform shell for cross-platform Geometric Asteroids. The game logic
   is asteroids.game (.cljc, shared with the JVM); the physics is wasm via the
   generated asteroids.kernels. This ns is the genuinely-cljs part: Canvas2D
   render, DOM input, rAF loop, and loading the kernels module."
  (:require [asteroids.game :as game]
            [asteroids.kernels :as k]))

(def TAU (* 2 js/Math.PI))
(def palette {3 "rgb(148,103,189)" 4 "rgb(214,39,40)" 5 "rgb(44,160,44)"
              6 "rgb(255,127,14)" 8 "rgb(31,119,180)" 7 "rgb(127,127,127)" 10 "rgb(23,190,207)"})

(defonce input (atom #{}))   ; a set of #{:left :right :up :fire} — (:left input) works
(defonce st (atom nil))

(defn- key->action [code]
  (case code "ArrowLeft" :left "ArrowRight" :right "ArrowUp" :up "Space" :fire nil))

(defn- draw-poly [ctx x y r sides angle color]
  (set! (.-strokeStyle ctx) color) (set! (.-lineWidth ctx) 1.6)
  (set! (.-shadowColor ctx) color) (set! (.-shadowBlur ctx) 8)
  (.beginPath ctx)
  (dotimes [kk (inc sides)]
    (let [a (+ angle (/ (* TAU kk) sides) (- (/ js/Math.PI 2)))
          px (+ x (* r (js/Math.cos a))) py (+ y (* r (js/Math.sin a)))]
      (if (zero? kk) (.moveTo ctx px py) (.lineTo ctx px py))))
  (.stroke ctx) (set! (.-shadowBlur ctx) 0))

(defn- render [ctx rd]
  (set! (.-fillStyle ctx) "#000") (.fillRect ctx 0 0 1024 768)
  (doseq [p (:polys rd)]
    (draw-poly ctx (:x p) (:y p) (:size p) (:sides p) (:angle p) (get palette (:sides p) "#ccc")))
  (let [s (:ship rd)] (draw-poly ctx (:x s) (:y s) 12 3 (+ (:angle s) (/ js/Math.PI 2)) "rgb(210,210,210)"))
  (set! (.-fillStyle ctx) "#fff") (set! (.-shadowColor ctx) "#fff") (set! (.-shadowBlur ctx) 6)
  (doseq [b (:bullets rd)] (.beginPath ctx) (.arc ctx (:x b) (:y b) 2.2 0 TAU) (.fill ctx))
  (set! (.-shadowBlur ctx) 0)
  (let [{:keys [score lives wave over]} (:hud rd)]
    (set! (.-textContent (.getElementById js/document "hud"))
          (str "SCORE " score "    LIVES " lives "    WAVE " wave (when over "    —  GAME OVER (space)")))))

(defn- start []
  (let [ctx (.getContext (.getElementById js/document "game") "2d")
        last (atom (js/performance.now))]
    (reset! st (game/init-state))
    (.addEventListener js/document "keydown"
                       (fn [e] (when-let [a (key->action (.-code e))] (swap! input conj a)
                                         (when (= a :fire) (.preventDefault e)))))
    (.addEventListener js/document "keyup"
                       (fn [e] (when-let [a (key->action (.-code e))] (swap! input disj a))))
    ;; expose for headless validation
    (set! (.-__ast js/window) #js {:stats (fn [] (let [s @st]
                                                   #js {:nShapes (count (:asteroids s)) :score (:score s)
                                                        :wave (:wave s) :lives (:lives s) :over (:over s)
                                                        :checksum (reduce (fn [c p] (+ c (* 13.1 (:x (:kin p))) (* 7.7 (:y (:kin p))))) 0.0 (:asteroids s))}))})
    (letfn [(frame [now]
              (let [dt (min 0.05 (/ (- now @last) 1000))]
                (reset! last now)
                (swap! st game/update-game @input dt)
                (render ctx (game/render-data @st))
                (js/requestAnimationFrame frame)))]
      (js/requestAnimationFrame frame))))

(defn ^:export init []
  (-> (k/init! "/kernels.wasm")
      (.then (fn [_] (start)))
      (.catch (fn [e] (js/console.error "kernel init failed:" e)))))
