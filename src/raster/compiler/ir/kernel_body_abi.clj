(ns raster.compiler.ir.kernel-body-abi
  "Checked projection from target-neutral KernelBody parameters to target ABI contracts.

  This seam depends on both IRs so the foundational ordered ABI remains independent of the larger
  scheduled-body vocabulary. Target backends retain control of parameter spelling and physical
  packing; they cannot drop a semantic memory precondition while doing so."
  (:require [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-body :as kbody]))

(defn project-contracts
  "Project verified KernelBody memory preconditions onto an ordered target ABI.

  Every StableRead must have exactly one physical input slot with the same compiler identity.
  Target parameter spelling remains independent in `:c-name`; this function only carries the
  semantic no-write-alias fact across lowering."
  [abi kernel-body]
  (let [abi (kabi/validate! abi)
        kernel-body (kbody/validate! kernel-body)
        stable-buffers (set (map :buffer (:stable-reads kernel-body)))]
    (doseq [buffer stable-buffers]
      (let [matches (filterv #(and (= :input (:kind %)) (= buffer (:name %))) abi)]
        (when-not (= 1 (count matches))
          (throw (ex-info "kernel stable read does not have exactly one target ABI input"
                          {:reason :kernel-body-stable-read-abi
                           :buffer buffer :matches matches :abi abi})))))
    (mapv (fn [slot]
            (if (contains? stable-buffers (:name slot))
              (assoc slot :aliasing :no-write-alias)
              slot))
          abi)))
