(ns umap-viz
  "Visualization helpers for the raster UMAP port.
   - scatter-png : pure-JVM Java2D rasterizer (no deps, scales to 70k+ points),
                   colored by integer label. Used for inline preview.
   - ->ds / tableplot-html : Tableplot (Plotly) scatter rendered via Clay to a
                   standalone interactive HTML page (the scicloj deliverable)."
  (:import [java.awt Color RenderingHints BasicStroke]
           [java.awt.image BufferedImage]
           [javax.imageio ImageIO]
           [java.io File]))

;; matplotlib tab10 palette (digit -> RGB)
(def tab10
  [[31 119 180] [255 127 14] [44 160 44] [214 39 40] [148 103 189]
   [140 86 75] [227 119 194] [127 127 127] [188 189 34] [23 190 207]])

(defn scatter-png
  "Rasterize a 2D embedding (flat double[n*2]) to a PNG, coloring each point by
  its integer label (mod 10 -> tab10). Returns the path."
  [^doubles emb ^ints labels n path & {:keys [w h r bg legend title]
                                       :or {w 1500 h 1500 r 1.5 bg 255}}]
  (let [img (BufferedImage. (int w) (int h) BufferedImage/TYPE_INT_RGB)
        g (.createGraphics img)
        n (long n)
        xs (double-array n) ys (double-array n)
        _ (dotimes [i n] (aset xs i (aget emb (* 2 i))) (aset ys i (aget emb (+ 1 (* 2 i)))))
        x0 (areduce xs i m Double/POSITIVE_INFINITY (min m (aget xs i)))
        x1 (areduce xs i m Double/NEGATIVE_INFINITY (max m (aget xs i)))
        y0 (areduce ys i m Double/POSITIVE_INFINITY (min m (aget ys i)))
        y1 (areduce ys i m Double/NEGATIVE_INFINITY (max m (aget ys i)))
        pad 40
        sx (/ (- w (* 2 pad)) (max 1e-9 (- x1 x0)))
        sy (/ (- h (* 2 pad)) (max 1e-9 (- y1 y0)))
        d (int (max 2 (* 2 r)))]
    (.setRenderingHint g RenderingHints/KEY_ANTIALIASING RenderingHints/VALUE_ANTIALIAS_ON)
    (.setColor g (Color. (int bg) (int bg) (int bg)))
    (.fillRect g 0 0 (int w) (int h))
    (dotimes [i n]
      (let [px (+ pad (* sx (- (aget xs i) x0)))
            py (- h pad (* sy (- (aget ys i) y0)))
            c (nth tab10 (mod (int (aget labels i)) 10))]
        (.setColor g (Color. (int (c 0)) (int (c 1)) (int (c 2))))
        (.fillOval g (int (- px r)) (int (- py r)) d d)))
    ;; legend
    (when legend
      (.setColor g (Color. 0 0 0))
      (dotimes [k 10]
        (let [c (nth tab10 k) ly (+ 30 (* k 26))]
          (.setColor g (Color. (int (c 0)) (int (c 1)) (int (c 2))))
          (.fillOval g (- w 110) ly 16 16)
          (.setColor g (Color. 0 0 0))
          (.drawString g (str k) (- w 84) (+ ly 13)))))
    (when title
      (.setColor g (Color. 0 0 0))
      (.drawString g ^String title 45 28))
    (.dispose g)
    (ImageIO/write img "png" (File. ^String path))
    path))

;; ---- Tableplot + Clay (interactive HTML deliverable) ----
;; Required lazily so the PNG path has zero plotting-lib dependencies.

(defn ->ds [^doubles emb ^ints labels n]
  (let [tc (requiring-resolve 'tablecloth.api/dataset)
        n (long n)]
    (tc {:x (mapv #(aget emb (* 2 %)) (range n))
         :y (mapv #(aget emb (+ 1 (* 2 %))) (range n))
         :digit (mapv #(str (aget labels %)) (range n))})))

(defn tableplot-plot
  "Build a Tableplot Plotly scatter (kindly-annotated spec) colored by :digit."
  [ds title]
  (let [layer-point (requiring-resolve 'scicloj.tableplot.v1.plotly/layer-point)
        plot (requiring-resolve 'scicloj.tableplot.v1.plotly/plot)]
    (-> ds
        (layer-point {:=x :x :=y :y :=color :digit :=mark-size 3 :=mark-opacity 0.6})
        (plot {:=title title :=width 900 :=height 800}))))
