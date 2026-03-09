(ns valley.inventory
  "Player inventory: hotbar (9 slots) + storage (27 slots).
  Each slot is nil or {:item :keyword :count N :durability N}."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.items :as items]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Inventory structure
;; ================================================================

(defn create-inventory
  "Create an empty inventory."
  []
  {:hotbar (vec (repeat 9 nil))
   :storage (vec (repeat 27 nil))
   :selected 0})

;; ================================================================
;; Slot operations
;; ================================================================

(defn- slot-add
  "Add count items to a slot. Returns [updated-slot remaining-count]."
  [slot item-id ^long count]
  (let [max-stack (long (get-in items/item-props [item-id :max-stack] 64))]
    (if (nil? slot)
      ;; Empty slot
      (let [to-add (min count max-stack)]
        [{:item item-id :count to-add} (- count to-add)])
      ;; Existing slot
      (if (= (:item slot) item-id)
        (let [current (long (:count slot))
              space (- max-stack current)
              to-add (min count space)]
          [(assoc slot :count (+ current to-add)) (- count to-add)])
        ;; Different item, can't add
        [slot count]))))

;; ================================================================
;; Add / remove items
;; ================================================================

(defn add-item
  "Add items to inventory (hotbar first, then storage). Returns updated inventory."
  [inv item-id ^long count]
  (loop [inv inv
         remaining count
         ;; Try hotbar first
         section :hotbar
         idx 0]
    (if (zero? remaining)
      inv
      (let [slots (get inv section)
            max-idx (clojure.core/count slots)]
        (if (>= idx max-idx)
          ;; Move to next section
          (if (= section :hotbar)
            (recur inv remaining :storage 0)
            inv)  ;; Both full, items lost
          (let [slot (nth slots idx)
                [new-slot rem] (slot-add slot item-id remaining)]
            (recur (assoc-in inv [section idx] new-slot) rem section (inc idx))))))))

(defn remove-item
  "Remove count items of item-id from inventory. Returns updated inventory."
  [inv item-id ^long count]
  (loop [inv inv
         remaining count
         section :hotbar
         idx 0]
    (if (zero? remaining)
      inv
      (let [slots (get inv section)
            max-idx (clojure.core/count slots)]
        (if (>= idx max-idx)
          (if (= section :hotbar)
            (recur inv remaining :storage 0)
            inv)
          (let [slot (nth slots idx)]
            (if (and slot (= (:item slot) item-id))
              (let [current (long (:count slot))
                    to-remove (min remaining current)
                    new-count (- current to-remove)
                    new-slot (if (zero? new-count) nil (assoc slot :count new-count))]
                (recur (assoc-in inv [section idx] new-slot) (- remaining to-remove) section (inc idx)))
              (recur inv remaining section (inc idx)))))))))

(defn has-item?
  "Check if inventory contains at least count of an item."
  [inv item-id ^long count]
  (let [total (+ (reduce (fn [^long acc slot]
                           (if (and slot (= (:item slot) item-id))
                             (+ acc (long (:count slot)))
                             acc))
                         0 (:hotbar inv))
                 (reduce (fn [^long acc slot]
                           (if (and slot (= (:item slot) item-id))
                             (+ acc (long (:count slot)))
                             acc))
                         0 (:storage inv)))]
    (>= total count)))

;; ================================================================
;; Hotbar selection
;; ================================================================

(defn get-held-item
  "Get the item in the currently selected hotbar slot, or nil."
  [inv]
  (let [slot (nth (:hotbar inv) (:selected inv))]
    (when slot (:item slot))))

(defn get-held-slot
  "Get the full slot map for the held item."
  [inv]
  (nth (:hotbar inv) (:selected inv)))

(defn select-slot
  "Change the selected hotbar slot (0-8)."
  [inv ^long idx]
  (assoc inv :selected (max 0 (min 8 idx))))

(defn cycle-selection
  "Cycle hotbar selection by delta (+1 or -1)."
  [inv ^long delta]
  (assoc inv :selected (mod (+ (:selected inv) delta 9) 9)))

;; ================================================================
;; Tool durability
;; ================================================================

(defn damage-held-tool
  "Reduce durability of held tool by 1. Removes tool if broken. Returns updated inv."
  [inv]
  (let [idx (:selected inv)
        slot (nth (:hotbar inv) idx)]
    (if (and slot (items/tool? (:item slot)))
      (let [dur (long (get slot :durability
                        (get-in items/item-props [(:item slot) :durability] 100)))
            new-dur (dec dur)]
        (if (<= new-dur 0)
          (assoc-in inv [:hotbar idx] nil)  ;; Tool broke
          (assoc-in inv [:hotbar idx :durability] new-dur)))
      inv)))

;; ================================================================
;; Swap slots (for UI)
;; ================================================================

(defn swap-slots
  "Swap two inventory slots. section is :hotbar or :storage."
  [inv [sec1 idx1] [sec2 idx2]]
  (let [slot1 (nth (get inv sec1) idx1)
        slot2 (nth (get inv sec2) idx2)]
    (-> inv
        (assoc-in [sec1 idx1] slot2)
        (assoc-in [sec2 idx2] slot1))))

(defn count-item
  "Count total of an item in inventory."
  [inv item-id]
  (+ (reduce (fn [^long acc slot]
               (if (and slot (= (:item slot) item-id))
                 (+ acc (long (:count slot)))
                 acc))
             0 (:hotbar inv))
     (reduce (fn [^long acc slot]
               (if (and slot (= (:item slot) item-id))
                 (+ acc (long (:count slot)))
                 acc))
             0 (:storage inv))))
