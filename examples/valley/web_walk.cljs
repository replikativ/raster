(ns valley.web-walk
  "WebGPU browser entry point for the valley infinite-streaming walker. Loads the wasm kernel
   module (k/init!), builds a WebGPU renderer, and runs the SHARED cross-platform shell
   (valley.shell) — identical game logic to the JVM (vk_walk). The only browser-specific bits
   are k/init!, the live-canvas aspect, and the __valley debug hooks.
   Controls: WASD move, F fly, arrows/mouse look (click to grab pointer), left=break
   right=place, Q/E hotbar, Space cull."
  (:require [raster.render.webgpu :as gpu]
            [valley.shell :as shell]
            [valley.kernels :as k]))

(defn ^:export init []
  (-> (k/init! "valley_kernels.wasm")     ; relative → works at site root (dev) and /raster/valley/ (Pages)
      (.then (fn [_] (gpu/make-renderer (.getElementById js/document "game"))))
      (.then (fn [rnd]
               (let [canvas (:canvas rnd)]
                 (shell/start! rnd
                             {:aspect   (fn [] (/ (.-width canvas) (.-height canvas)))
                              :on-state (fn [g]
                                          (set! (.-__valley js/window)
                                                #js {:player (fn [] (let [p ((:player g))]
                                                                      (clj->js {:pos (vec (:pos p))
                                                                                :on-ground (:on-ground p) :sel (:sel p)})))
                                                     :mobs   (fn [] ((:mobs g)))
                                                     :mob0   (fn [] (let [pp ((:mob0 g))]
                                                                      #js [(aget pp 0) (aget pp 1) (aget pp 2)]))}))}))))
      (.catch (fn [e] (js/console.error "valley walk init failed:" e)))))
