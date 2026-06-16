(ns fashr (:require [raster.umap.fit :as fit]))
(load-file "/home/christian-weilbach/Development/raster/dev/umap_viz.clj")

(defn- npy-doff [^bytes bs]
  (let [bb (doto (java.nio.ByteBuffer/wrap bs) (.order java.nio.ByteOrder/LITTLE_ENDIAN))
        hlen (do (.position bb 8) (bit-and (int (.getShort bb)) 0xFFFF))]
    [bb (+ 10 hlen)]))
(defn read-f64 ^doubles [path]
  (let [bs (java.nio.file.Files/readAllBytes (java.nio.file.Path/of path (make-array String 0)))
        [bb doff] (npy-doff bs) n (quot (- (alength bs) doff) 8) out (double-array n)]
    (.position bb (int doff)) (dotimes [i n] (aset out i (.getDouble bb))) out))
(defn read-i32 ^ints [path]
  (let [bs (java.nio.file.Files/readAllBytes (java.nio.file.Path/of path (make-array String 0)))
        [bb doff] (npy-doff bs) n (quot (- (alength bs) doff) 4) out (int-array n)]
    (.position bb (int doff)) (dotimes [i n] (aset out i (.getInt bb))) out))

;; brute 2D nearest-neighbour label agreement
(defn agree [^doubles emb ^ints y n]
  (loop [i 0 acc 0]
    (if (< i n)
      (let [xi (* 2 i)
            bj (loop [j 0 bd Double/MAX_VALUE bj -1]
                 (if (< j n)
                   (if (= j i) (recur (inc j) bd bj)
                     (let [dx (- (aget emb xi) (aget emb (* 2 j)))
                           dy (- (aget emb (+ xi 1)) (aget emb (+ (* 2 j) 1)))
                           dd (+ (* dx dx) (* dy dy))]
                       (if (< dd bd) (recur (inc j) dd j) (recur (inc j) bd bj))))
                   bj))]
        (recur (inc i) (+ acc (if (= (aget y bj) (aget y i)) 1 0))))
      (/ (double acc) n))))

;; compose two PNGs side by side
(defn compose [a b out]
  (let [ia (javax.imageio.ImageIO/read (java.io.File. a))
        ib (javax.imageio.ImageIO/read (java.io.File. b))
        w (+ (.getWidth ia) (.getWidth ib)) h (max (.getHeight ia) (.getHeight ib))
        c (java.awt.image.BufferedImage. w h java.awt.image.BufferedImage/TYPE_INT_RGB)
        g (.createGraphics c)]
    (.setColor g java.awt.Color/WHITE) (.fillRect g 0 0 w h)
    (.drawImage g ia 0 0 nil) (.drawImage g ib (.getWidth ia) 0 nil) (.dispose g)
    (javax.imageio.ImageIO/write c "png" (java.io.File. out)) out))

(def N 10000)
(def Xd (read-f64 "/tmp/umap_gold/fashion_X.npy"))
(def Xf (let [f (float-array (alength Xd))] (dotimes [i (alength Xd)] (aset f i (float (aget Xd i)))) f))
(def y (read-i32 "/tmp/umap_gold/fashion_y.npy"))
(def umap-emb (read-f64 "/tmp/umap_gold/fashion_emb_umap.npy"))

(def t0 (System/nanoTime))
(def res (fit/fit Xf N 784 :k 15 :init :spectral :seed 42))
(def secs (/ (- (System/nanoTime) t0) 1.0e9))
(def remb (:emb res))
(def ra (agree remb y N))
(def ua (agree umap-emb y N))
(umap-viz/scatter-png remb y N "/tmp/fashion_raster.png" :title (format "raster (cosine, spectral) — 2D-NN %.3f" ra))
(umap-viz/scatter-png umap-emb y N "/tmp/fashion_umap.png" :title (format "umap (cosine) — 2D-NN %.3f" ua))
(compose "/tmp/fashion_raster.png" "/tmp/fashion_umap.png" "/tmp/fashion_compare.png")
(spit "/tmp/fashion_out.txt"
  (format "raster fit %.1fs  raster 2D-NN %.3f  umap 2D-NN %.3f  (umap fit 46.9s)" secs ra ua))
(println (slurp "/tmp/fashion_out.txt"))
