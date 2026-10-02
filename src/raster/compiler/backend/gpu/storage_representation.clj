(ns raster.compiler.backend.gpu.storage-representation
  "Generated storage probes, not session authority or a numerical codec.

   Each dtype is measured independently. Explicit bit-pattern oracles never consult host
   byte order. The finite sentinels test storage under the declared IEEE/two's-complement ABI;
   they do not certify NaN arithmetic, arbitrary conversions, or another live session."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]))

(defn- sentinels [element-dtype]
  ;; These are probe fixtures, not a second dtype registry. Widths and type legality come
  ;; exclusively from dtype. Both elements differ, with asymmetric bytes within each word.
  (case (dtype/canon element-dtype)
    :byte [0x25 0x63]
    :int [0x12345678 0x23456719]
    :long [0x123456789abcdef 0x23456789abcdef1]
    :half [0x3d55 0x425b]
    :float [0x3fabcdef 0x40234567]
    :double [0x3ff123456789abcd 0x40023456789abcde]))

(defn expected-bytes
  "Unsigned byte vector for two exact sentinels in explicit little/big element order."
  [element-dtype byte-order]
  (when-not (contains? #{:little-endian :big-endian} byte-order)
    (throw (ex-info "storage probe requires explicit byte order"
                    {:reason :storage-probe-byte-order :byte-order byte-order})))
  (let [width (dtype/bytes-of element-dtype)
        offsets (if (= :little-endian byte-order) (range width) (reverse (range width)))]
    (vec (mapcat (fn [word]
                   (map #(bit-and 255 (unsigned-bit-shift-right (long word) (* 8 %))) offsets))
                 (sentinels element-dtype)))))

(defn classify-bytes
  "Classify an observation, without granting provenance. Byte storage is order-invariant.
   Reject wrong extents, permutations, corrupted bytes and unsupported representations."
  [element-dtype observed]
  (let [dt (dtype/canon element-dtype)
        observed (mapv #(bit-and 255 (long %)) observed)
        candidates (filterv #(= observed (expected-bytes dt %))
                            [:little-endian :big-endian])]
    (cond
      (and (= :byte dt) (= 2 (count candidates))) :order-invariant
      (= 1 (count candidates)) (first candidates)
      :else (throw (ex-info "device storage does not match either declared ABI byte order"
                            {:reason :storage-probe-representation :dtype dt
                             :observed observed :matches candidates})))))

(defn kernel-body
  "One work item writes two exactly representable finite values through typed stores."
  [element-dtype]
  (let [dt (dtype/canon element-dtype)
        values (mapv (fn [word]
                       (case dt
                         :half (Float/float16ToFloat (short word))
                         :float (Float/intBitsToFloat (int word))
                         :double (Double/longBitsToDouble (long word))
                         word))
                     (sentinels dt))]
    (body/make
     {:id [:storage-representation-probe-v1 dt]
      :parameters [(body/->KernelParameter 'out :output dt [2] :global
                                          (layout/row-major [2] dt) :result)]
      :operations (mapv (fn [index value]
                          (body/->ScalarStore 'out [index] (body/literal value dt) nil))
                        (range 2) values)
      :launch (launch/spec {:workgroup-size [1] :group-count [1]})
      :provenance {:kind :storage-representation-probe-v1}
      :attributes {:kind :scalar}})))

(defn emit-artifact
  "Use the common C-family emitter and preserve its compilation requirements.
   Unsupported dialects/types decline; no handwritten kernel or host-order fallback."
  [element-dtype target-dialect]
  (let [dt (dtype/canon element-dtype)
        target (case target-dialect :opencl-portable :opencl-c :cuda :cuda-c :hip :hip-cpp)
        name (str "rstr_storage_probe_" (clojure.core/name dt))
        kb (kernel-body dt)
        module (emitter/emit-scalar-module name kb {:target-dialect target-dialect
                                                   :parameter-names {'out "rstr_out"}})]
    (artifact/make
     {:kernel-name name :target target :source (:source module)
      :abi [(abi/slot 'out :output dt :c-name "rstr_out" :role :result)]
      :arguments '[out] :launch (:launch kb)
      :effects {:kind :storage-representation-probe}
      :provenance (:provenance kb)
      :attributes {:compilation (:compilation module {})}})))
