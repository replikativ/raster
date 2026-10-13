(ns raster.compiler.fixtures.mutable-aggregate-training
  "Small nested parameter/moment recurrence for mutable aggregate boundary tests."
  (:require [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm step!
  [parameters :- (HMap {:layer (HMap {:weight (Array float) :bias (Array float)})})
   moments :- (HMap {:layer (HMap {:mean (Array float) :variance (Array float)})})
   batch :- (Array float) clock :- (Array long) n :- Long] :- Double
  (let [weight (:weight (:layer parameters))
        bias (:bias (:layer parameters))
        mean (:mean (:layer moments))
        variance (:variance (:layer moments))
        clock-value (arrays/aget clock 0)
        loss (par/reduce sum (float 0.0) i n
                         (let [error (n/- (arrays/aget weight i) (arrays/aget batch i))]
                           (n/+ sum (n/* error error))))
        loss-value (double loss)]
    (par/map-void! i n
      (let [datum (arrays/aget batch i)
            old-mean (arrays/aget mean i)
            old-variance (arrays/aget variance i)
            gradient (n/+ datum (n/* (float 0.01) (float clock-value)))
            next-mean (n/+ (n/* (float 0.8) old-mean)
                            (n/* (float 0.2) gradient))
            next-variance (n/+ (n/* (float 0.9) old-variance)
                                (n/* (float 0.1) (n/* gradient gradient)))
            next-weight (n/- (arrays/aget weight i) (n/* (float 0.05) next-mean))
            next-bias (n/- (arrays/aget bias i) (n/* (float 0.025) next-variance))]
        (arrays/aset mean i next-mean)
        (arrays/aset variance i next-variance)
        (arrays/aset weight i next-weight)
        (arrays/aset bias i next-bias)))
    loss-value))

(deftm return-weight!
  [parameters :- (HMap {:layer (HMap {:weight (Array float) :bias (Array float)})})
   n :- Long]
  :- (Array float)
  (let [weight (:weight (:layer parameters))]
    (par/map-void! i n
      (arrays/aset weight i (arrays/aget weight i)))
    weight))

(def state-paths
  [[:parameters :layer :weight]
   [:parameters :layer :bias]
   [:moments :layer :mean]
   [:moments :layer :variance]])

(defn inputs
  ([] (inputs 0))
  ([batch-shift]
   {:parameters {:layer {:weight (float-array [0.5 -0.25 0.75 -0.5])
                         :bias (float-array [0.1 -0.2 0.3 -0.4])}}
    :moments {:layer {:mean (float-array [0.0 0.05 -0.05 0.1])
                      :variance (float-array [0.01 0.02 0.03 0.04])}}
    :batch (float-array (map #(+ batch-shift %) [0.25 -0.5 0.125 0.75]))
    :clock (long-array [1])
    :n 4}))

(defn arguments [{:keys [parameters moments batch clock n]}]
  [parameters moments batch clock n])

(defn snapshot
  [{:keys [parameters moments]}]
  {:weight (vec (get-in parameters [:layer :weight]))
   :bias (vec (get-in parameters [:layer :bias]))
   :mean (vec (get-in moments [:layer :mean]))
   :variance (vec (get-in moments [:layer :variance]))})

(defn oracle-step
  "Independent coordinate recurrence; it does not call the Raster training function."
  [state batch clock]
  (let [{:keys [weight bias mean variance]} state
        batch (vec batch)
        loss (float (reduce (fn [sum [w x]]
                              (let [error (float (- (float w) (float x)))]
                                (float (+ sum (float (* error error))))))
                            (float 0.0) (map vector weight batch)))
        coordinates
        (mapv (fn [w b m v x]
                (let [f01 (float 0.01) f08 (float 0.8) f02 (float 0.2)
                      f09 (float 0.9) f01m (float 0.1)
                      f005 (float 0.05) f0025 (float 0.025)
                      clock-value (float clock)
                      gradient (float (+ (float x) (float (* f01 clock-value))))
                      next-mean (float (+ (float (* f08 (float m)))
                                          (float (* f02 gradient))))
                      gradient-square (float (* gradient gradient))
                      next-variance (float (+ (float (* f09 (float v)))
                                              (float (* f01m gradient-square))))]
                  [(float (- (float w) (float (* f005 next-mean))))
                   (float (- (float b) (float (* f0025 next-variance))))
                   next-mean next-variance]))
              weight bias mean variance batch)]
    {:loss (double loss)
     :state {:weight (mapv first coordinates)
             :bias (mapv second coordinates)
             :mean (mapv #(nth % 2) coordinates)
             :variance (mapv #(nth % 3) coordinates)}}))
