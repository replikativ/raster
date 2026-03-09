(ns valley.lighting
  "Light propagation: sunlight (top-down) and point lights (BFS).
  Each block stores two 4-bit light channels:
    bits 0-3: daylight (sunlight, 15 = direct sun)
    bits 4-7: nightlight (torches, max 14)

  Light decays by 1 per block of distance from source."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.world :as w])
  (:import [java.util ArrayDeque]))

(set! *warn-on-reflection* true)

(def ^:const LIGHT-SUN 15)

;; Non-linear light decode table (0-15 → 0.0-1.0)
;; Loosely follows Minetest gamma curve
(def light-decode-table
  (float-array
    (mapv (fn [i]
            (let [x (/ (double i) 15.0)]
              (float (Math/pow (max 0.0 x) 1.6))))
          (range 16))))

;; ================================================================
;; Sunlight propagation (top-down per column)
;; ================================================================

(defn propagate-sunlight-chunk!
  "Propagate sunlight top-down through a single chunk.
  `above-sunlit` is a boolean 16x16 array indicating whether the block above
  each (x,z) column has sunlight (from the chunk above, or true if topmost)."
  [chunk ^booleans above-sunlit]
  (let [blocks ^ints (:blocks chunk)
        light ^bytes (:light chunk)]
    (dotimes [lz w/CHUNK-SIZE]
      (dotimes [lx w/CHUNK-SIZE]
        (let [col-idx (+ lx (* lz w/CHUNK-SIZE))]
          (loop [ly (dec w/CHUNK-SIZE)
                 sunlit? (boolean (aget above-sunlit col-idx))]
            (when (>= ly 0)
              (let [idx (w/block-index lx ly lz)
                    block-id (aget blocks idx)]
                (if (and sunlit? (w/sunlight-propagates? block-id))
                  (do
                    (w/set-day-light! light idx LIGHT-SUN)
                    (recur (dec ly) (boolean true)))
                  (do
                    ;; Not sunlit from here down (for this pass)
                    (recur (dec ly) (boolean false))))))))))))

(defn propagate-sunlight-column!
  "Propagate sunlight for one (cx,cz) column of chunks, top to bottom."
  [world ^long cx ^long cz ^long height-chunks]
  (let [above-sunlit (boolean-array (* w/CHUNK-SIZE w/CHUNK-SIZE) true)]
    ;; Top-down through chunk layers
    (loop [cy (dec height-chunks)]
      (when (>= cy 0)
        (when-let [chunk (get world [cx cy cz])]
          (propagate-sunlight-chunk! chunk above-sunlit)
          ;; Update above-sunlit for next chunk below: check bottom row
          (let [blocks ^ints (:blocks chunk)
                light ^bytes (:light chunk)]
            (dotimes [lz w/CHUNK-SIZE]
              (dotimes [lx w/CHUNK-SIZE]
                (let [idx (w/block-index lx 0 lz)
                      col-idx (+ lx (* lz w/CHUNK-SIZE))
                      day-light (bit-and (aget light idx) 0x0F)]
                  (aset above-sunlit col-idx
                    (= day-light LIGHT-SUN)))))))
        (recur (dec cy))))))

(defn propagate-sunlight-world!
  "Propagate sunlight for entire world."
  [world ^long height-chunks]
  ;; Find all unique (cx,cz) column pairs
  (let [columns (into #{} (map (fn [[[cx _ cz]]] [cx cz]) world))]
    (doseq [[cx cz] columns]
      (propagate-sunlight-column! world cx cz height-chunks))))

;; ================================================================
;; BFS light spreading (torch/point lights + sunlight scatter)
;; ================================================================

(defn- pack-light-entry ^long [^long x ^long y ^long z ^long brightness]
  ;; Pack into a single long: x(8) y(8) z(8) brightness(8)
  (bit-or (bit-shift-left (bit-and x 0xFF) 24)
          (bit-shift-left (bit-and y 0xFF) 16)
          (bit-shift-left (bit-and z 0xFF) 8)
          (bit-and brightness 0xFF)))

(defn- unpack-x ^long [^long packed]
  (bit-and (unsigned-bit-shift-right packed 24) 0xFF))

(defn- unpack-y ^long [^long packed]
  (bit-and (unsigned-bit-shift-right packed 16) 0xFF))

(defn- unpack-z ^long [^long packed]
  (bit-and (unsigned-bit-shift-right packed 8) 0xFF))

(defn- unpack-brightness ^long [^long packed]
  (bit-and packed 0xFF))

;; 6 neighbor offsets
(def ^:private neighbor-offsets
  (int-array [1 0 0  -1 0 0  0 1 0  0 -1 0  0 0 1  0 0 -1]))

(defn spread-light-chunk!
  "BFS-spread light within a single chunk for a given light bank.
  bank: :day or :night
  seeds: seq of [lx ly lz brightness] initial light sources."
  [chunk bank seeds]
  (let [blocks ^ints (:blocks chunk)
        light ^bytes (:light chunk)
        queue (ArrayDeque. 256)
        get-light (if (= bank :day)
                    (fn ^long [^long idx] (bit-and (aget light (int idx)) 0x0F))
                    (fn ^long [^long idx] (bit-and (unsigned-bit-shift-right (aget light (int idx)) 4) 0x0F)))
        set-light (if (= bank :day)
                    (fn [^long idx ^long val] (w/set-day-light! light idx val))
                    (fn [^long idx ^long val] (w/set-night-light! light idx val)))]
    ;; Seed the queue
    (doseq [[lx ly lz brightness] seeds]
      (let [idx (w/block-index lx ly lz)]
        (when (> (long brightness) (get-light idx))
          (set-light idx brightness)
          (.add queue (Long/valueOf (pack-light-entry lx ly lz brightness))))))
    ;; BFS
    (loop []
      (when-let [entry (.poll queue)]
        (let [packed (long entry)
              x (unpack-x packed)
              y (unpack-y packed)
              z (unpack-z packed)
              brightness (unpack-brightness packed)
              spread-val (dec brightness)]
          (when (pos? spread-val)
            (dotimes [i 6]
              (let [nx (+ x (aget ^ints neighbor-offsets (* i 3)))
                    ny (+ y (aget ^ints neighbor-offsets (+ (* i 3) 1)))
                    nz (+ z (aget ^ints neighbor-offsets (+ (* i 3) 2)))]
                (when (and (>= nx 0) (< nx w/CHUNK-SIZE)
                           (>= ny 0) (< ny w/CHUNK-SIZE)
                           (>= nz 0) (< nz w/CHUNK-SIZE))
                  (let [nidx (w/block-index nx ny nz)
                        nblock (aget blocks nidx)]
                    (when (w/light-propagates? nblock)
                      (when (> spread-val (get-light nidx))
                        (set-light nidx spread-val)
                        (.add queue (Long/valueOf (pack-light-entry nx ny nz spread-val)))))))))))
        (recur)))))

(defn scatter-sunlight-chunk!
  "After vertical sunlight propagation, scatter sunlight sideways and downward
  via BFS within a chunk. Sunlight value 15 decays to 14, 13, etc."
  [chunk]
  (let [blocks ^ints (:blocks chunk)
        light ^bytes (:light chunk)
        seeds (java.util.ArrayList.)]
    ;; Collect all blocks that have daylight > 0 as BFS seeds
    (dotimes [ly w/CHUNK-SIZE]
      (dotimes [lz w/CHUNK-SIZE]
        (dotimes [lx w/CHUNK-SIZE]
          (let [idx (w/block-index lx ly lz)
                day (bit-and (aget light idx) 0x0F)]
            (when (pos? day)
              (.add seeds [lx ly lz day]))))))
    (spread-light-chunk! chunk :day seeds)))

(defn spread-torch-lights-chunk!
  "Find all torch/emissive blocks in chunk and BFS-spread their night light."
  [chunk]
  (let [blocks ^ints (:blocks chunk)
        seeds (java.util.ArrayList.)]
    (dotimes [ly w/CHUNK-SIZE]
      (dotimes [lz w/CHUNK-SIZE]
        (dotimes [lx w/CHUNK-SIZE]
          (let [idx (w/block-index lx ly lz)
                block-id (aget blocks idx)
                emit (w/light-source block-id)]
            (when (pos? emit)
              (.add seeds [lx ly lz emit]))))))
    (spread-light-chunk! chunk :night seeds)))

;; ================================================================
;; Full chunk lighting
;; ================================================================

(defn light-chunk!
  "Compute full lighting for a chunk (both sunlight scatter and torch lights).
  Sunlight must already be propagated vertically before calling this."
  [chunk]
  (scatter-sunlight-chunk! chunk)
  (spread-torch-lights-chunk! chunk))

(defn light-world!
  "Compute lighting for the entire world.
  1. Propagate sunlight top-down through all columns
  2. Scatter sunlight sideways via BFS per chunk
  3. Spread torch lights via BFS per chunk"
  [world ^long height-chunks]
  (propagate-sunlight-world! world height-chunks)
  (doseq [[_pos chunk] world]
    (light-chunk! chunk)))

;; ================================================================
;; Incremental relighting (after block break/place)
;; ================================================================

(defn clear-light-chunk!
  "Zero out the light array for a chunk."
  [chunk]
  (java.util.Arrays/fill ^bytes (:light chunk) (byte 0)))

(defn relight-column!
  "Recompute lighting for the (cx,cz) column of chunks in the world.
  Call after modifying a block to update sunlight + torch propagation.
  Returns the set of chunk positions that were relighted."
  [world ^long cx ^long cz]
  ;; Find all cy values for this column
  (let [column-chunks (filterv (fn [[[x _ z]]] (and (= x cx) (= z cz))) world)
        cy-vals (sort > (map (fn [[[_ cy _]]] cy) column-chunks))]
    ;; 1. Clear light for all chunks in the column
    (doseq [cy cy-vals]
      (when-let [chunk (get world [cx cy cz])]
        (clear-light-chunk! chunk)))
    ;; 2. Re-propagate sunlight top-down
    (let [max-cy (if (seq cy-vals) (inc (long (apply max cy-vals))) 0)]
      (propagate-sunlight-column! world cx cz max-cy))
    ;; 3. Scatter + torch BFS for each chunk in column
    (doseq [cy cy-vals]
      (when-let [chunk (get world [cx cy cz])]
        (light-chunk! chunk)))
    ;; Return affected chunk positions
    (set (map (fn [[[x y z]]] [x y z]) column-chunks))))
