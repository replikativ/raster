(ns raster.compiler.ir.buffer-view
  "Stable physical allocation and view descriptors.

   These values contain no backend handles. BufferAllocation identifies storage and its lifetime/
   coherence contract; BufferView identifies one typed, shaped byte range within that storage.
   Semantic AxisMaps and accelerator thread/register layouts remain separate concerns."
  (:require [raster.compiler.core.dtype :as dtype]))

(def ownership-kinds #{:owned :borrowed :external})
(def coherence-kinds #{:host-coherent :explicit-transfer :device-only})

(defrecord BufferAllocation
           [id byte-size memory-space device alignment coherence ownership])
(defrecord BufferView
           [id allocation byte-offset byte-length dtype shape strides])

(defn buffer-allocation? [x]
  (and x (= "raster.compiler.ir.buffer_view.BufferAllocation" (.getName (class x)))))

(defn buffer-view? [x]
  (and x (= "raster.compiler.ir.buffer_view.BufferView" (.getName (class x)))))

(defn- power-of-two? [n]
  (and (pos? n) (zero? (bit-and n (dec n)))))

(defn validate-allocation!
  [allocation]
  (when-not (buffer-allocation? allocation)
    (throw (ex-info "buffer allocation must be a BufferAllocation value"
                    {:allocation allocation :actual (type allocation)})))
  (let [{:keys [id byte-size memory-space device alignment coherence ownership]} allocation]
    (when (nil? id)
      (throw (ex-info "buffer allocation requires a stable identity" {})))
    (when-not (and (integer? byte-size) (not (neg? byte-size)))
      (throw (ex-info "buffer allocation byte size must be a non-negative integer"
                      {:byte-size byte-size})))
    (when-not (keyword? memory-space)
      (throw (ex-info "buffer allocation requires a memory-space keyword"
                      {:memory-space memory-space})))
    (when-not (or (nil? device) (keyword? device))
      (throw (ex-info "buffer allocation device must be nil or a device keyword"
                      {:device device})))
    (when-not (and (integer? alignment) (power-of-two? alignment))
      (throw (ex-info "buffer allocation alignment must be a positive power of two"
                      {:alignment alignment})))
    (when-not (contains? coherence-kinds coherence)
      (throw (ex-info "buffer allocation has an invalid coherence contract"
                      {:coherence coherence :allowed coherence-kinds})))
    (when-not (contains? ownership-kinds ownership)
      (throw (ex-info "buffer allocation has an invalid ownership contract"
                      {:ownership ownership :allowed ownership-kinds}))))
  allocation)

(defn allocation
  [{:keys [id byte-size memory-space device alignment coherence ownership]
    :or {alignment 1 coherence :device-only ownership :owned}}]
  (validate-allocation!
   (->BufferAllocation id byte-size memory-space device alignment coherence ownership)))

(defn dense-strides
  "Contiguous row-major element strides for a realized shape."
  [shape]
  (loop [remaining (rseq (vec shape)) stride 1 result ()]
    (if-let [extent (first remaining)]
      (recur (next remaining) (* stride extent) (conj result stride))
      (vec result))))

(defn required-byte-span
  "Byte span from the first addressed element through the last for shape/element strides."
  [dtype shape strides]
  (let [element-bytes (long (dtype/bytes-of dtype))]
    (if (some zero? shape)
      0
      (* element-bytes
         (inc (reduce + 0 (map (fn [extent stride]
                                 (* (dec extent) stride))
                               shape strides)))))))

(defn validate-view!
  [view]
  (when-not (buffer-view? view)
    (throw (ex-info "buffer view must be a BufferView value"
                    {:view view :actual (type view)})))
  (let [{:keys [id allocation byte-offset byte-length dtype shape strides]} view
        allocation (validate-allocation! allocation)]
    (when (nil? id)
      (throw (ex-info "buffer view requires a stable identity" {})))
    (dtype/canon dtype)
    (when-not (and (vector? shape)
                   (every? #(and (integer? %) (not (neg? %))) shape))
      (throw (ex-info "buffer view shape must be a realized non-negative integer vector"
                      {:shape shape})))
    (when-not (and (vector? strides) (= (count shape) (count strides))
                   (every? #(and (integer? %) (not (neg? %))) strides))
      (throw (ex-info "buffer view strides must be a non-negative integer per shape axis"
                      {:shape shape :strides strides})))
    (when-not (and (integer? byte-offset) (not (neg? byte-offset))
                   (integer? byte-length) (not (neg? byte-length)))
      (throw (ex-info "buffer view byte offset/length must be non-negative integers"
                      {:byte-offset byte-offset :byte-length byte-length})))
    (let [element-bytes (long (dtype/bytes-of dtype))
          required (required-byte-span dtype shape strides)]
      (when-not (zero? (mod byte-offset element-bytes))
        (throw (ex-info "buffer view byte offset is not aligned to its dtype"
                        {:byte-offset byte-offset :dtype dtype :element-bytes element-bytes})))
      (when-not (= required byte-length)
        (throw (ex-info "buffer view byte length differs from its shaped strided span"
                        {:byte-length byte-length :required required
                         :shape shape :strides strides :dtype dtype})))
      (when (> (+ byte-offset byte-length) (:byte-size allocation))
        (throw (ex-info "buffer view exceeds its allocation"
                        {:allocation (:id allocation) :allocation-bytes (:byte-size allocation)
                         :byte-offset byte-offset :byte-length byte-length}))))
    view))

(defn view
  "Construct a checked BufferView. `byte-length` defaults to the shaped strided span and
   `strides` defaults to dense row-major element strides."
  [allocation {:keys [id byte-offset byte-length dtype shape strides]
               :or {byte-offset 0}}]
  (when (nil? shape)
    (throw (ex-info "buffer view requires an explicit realized shape" {:shape shape})))
  (let [allocation (validate-allocation! allocation)
        shape (vec shape)
        strides (vec (or strides (dense-strides shape)))
        byte-length (or byte-length (required-byte-span dtype shape strides))]
    (validate-view!
     (->BufferView (or id [(:id allocation) byte-offset byte-length dtype shape strides])
                   allocation byte-offset byte-length (dtype/canon dtype) shape strides))))

(defn byte-end [view]
  (+ (:byte-offset (validate-view! view)) (:byte-length view)))

(defn overlaps?
  "True when two views address at least one common byte of the same allocation."
  [a b]
  (let [a (validate-view! a) b (validate-view! b)]
    (and (= (get-in a [:allocation :id]) (get-in b [:allocation :id]))
         (< (:byte-offset a) (byte-end b))
         (< (:byte-offset b) (byte-end a)))))

(defn disjoint? [a b] (not (overlaps? a b)))

(defn same-range?
  [a b]
  (let [a (validate-view! a) b (validate-view! b)]
    (and (= (get-in a [:allocation :id]) (get-in b [:allocation :id]))
         (= (:byte-offset a) (:byte-offset b))
         (= (:byte-length a) (:byte-length b)))))

(defn contiguous?
  [view]
  (let [{:keys [shape strides]} (validate-view! view)]
    (= strides (dense-strides shape))))

(defn subview
  "Construct a view whose byte range is contained in `base`. Shape/strides describe the new
   logical interpretation; `byte-offset` is relative to the base view."
  [base {:keys [byte-offset] :as opts}]
  (let [base (validate-view! base)
        relative (long (or byte-offset 0))
        child (view (:allocation base)
                    (assoc opts :byte-offset (+ (:byte-offset base) relative)
                           :dtype (or (:dtype opts) (:dtype base))))]
    (when (or (neg? relative) (> (byte-end child) (byte-end base)))
      (throw (ex-info "buffer subview exceeds its base view"
                      {:base (:id base) :view (:id child)})))
    child))
