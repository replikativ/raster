(ns raster.compiler.ir.kernel-body-abi
  "Checked projection from target-neutral KernelBody parameters to target ABI contracts.

  This seam depends on both IRs so the foundational ordered ABI remains independent of the larger
  scheduled-body vocabulary. Target backends retain control of parameter spelling and physical
  packing; they cannot drop a semantic memory precondition while doing so."
  (:require [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-body :as kbody]))

(defn retained-buffer-contracts
  "Check the ordered retained body/ABI projection before exposing global storage shapes.
   This is request-local admission, not a new artifact certificate or runtime ownership proof."
  [abi kernel-body]
  (let [abi (kabi/validate! abi) body (kbody/validate! kernel-body)
        parameters (:parameters body)]
    (when-not (= (count abi) (count parameters))
      (throw (ex-info "retained body parameter count differs from its physical ABI"
                      {:reason :kernel-body-capacity-projection})))
    (mapv (fn [slot parameter]
            (when-not (and (= (:name slot) (:id parameter))
                           (= (:kind slot) (:kind parameter))
                           (= (:role slot) (if (= '_nseg (:id parameter)) :bound (:role parameter)))
                           (= (:dtype parameter)
                              (if (= :scalar (:kind slot)) (:kernel-dtype slot) (:dtype slot))))
              (throw (ex-info "retained body parameter differs from its physical ABI"
                              {:reason :kernel-body-capacity-projection
                               :slot slot :parameter parameter})))
            (when (contains? #{:input :inout} (:kind slot))
              (let [layout (:layout parameter)]
                (when-not (and (= :global (:memory-space parameter))
                               (contains? #{:row-major :col-major} (:kind layout))
                               (or (nil? (:perm layout))
                                   (and (= (count (:shape parameter)) (count (:perm layout)))
                                        (= (set (range (count (:shape parameter)))) (set (:perm layout)))))
                               (or (nil? (:strides layout))
                                   (and (vector? (:strides layout))
                                        (not-any? seq? (:strides layout))
                                        (= (count (:shape parameter)) (count (:strides layout))))))
                  (throw (ex-info "retained storage capacity requires a supported global parameter layout"
                                  {:reason :kernel-body-capacity-layout :parameter parameter})))))
            parameter)
          abi parameters)))

(defn project-contracts
  "Project verified KernelBody memory preconditions onto an ordered target ABI.

  Every StableRead must have exactly one physical input slot with the same compiler identity.
  Target parameter spelling remains independent in `:c-name`; this function only carries the
  semantic no-write-alias fact across lowering."
  [abi kernel-body]
  (let [abi (kabi/validate! abi)
        kernel-body (kbody/validate! kernel-body)
        stable-buffers (set (map :buffer (:stable-reads kernel-body)))
        required-alignments (kbody/required-async-source-alignments kernel-body)]
    (doseq [buffer stable-buffers]
      (let [matches (filterv #(and (= :input (:kind %)) (= buffer (:name %))) abi)]
        (when-not (= 1 (count matches))
          (throw (ex-info "kernel stable read does not have exactly one target ABI input"
                          {:reason :kernel-body-stable-read-abi
                           :buffer buffer :matches matches :abi abi})))))
    (mapv (fn [slot]
            (cond-> slot
              (contains? stable-buffers (:name slot))
              (assoc :aliasing :no-write-alias)

              (contains? required-alignments (:name slot))
              (assoc :alignment (max (or (:alignment slot) 1)
                                     (get required-alignments (:name slot))))))
          abi)))
