(ns raster.compiler.passes.parallel.paged-kv-append-body
  "Target-neutral one-work-item-per-component schedule for routed K/V assignment."
  (:require [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.paged-kv-append :as append]))

(defn lower
  "Lower a validated PagedKVAppend to typed scalar/control KernelBody.

   A host-validated slot map promises unique active destinations. The device guard retains safe
   behavior for inactive (-1) and out-of-range slots; component masks cover unequal K/V widths."
  [problem workgroup-x]
  (let [{:keys [id batch-size key-elements-per-token value-elements-per-token
                key-rows value-rows slot-mapping key-pages value-pages]}
        (append/validate! problem)
        slots (append/physical-slots problem)
        width (max key-elements-per-token value-elements-per-token)
        group-x :append-group-x
        lane-x :append-lane-x
        lane :append-lane
        component :append-component
        slot :append-slot
        converted (fn [prefix input output mask]
                    (let [loaded (keyword (str prefix "-loaded"))
                          half-value (keyword (str prefix "-half"))]
                      [(body/->ScalarLoad (body/value loaded :float) input [lane component]
                                         mask (body/literal 0.0 :float) :streaming)
                       (body/->ScalarCompute
                        (body/value half-value :half)
                        (body/cast-expression loaded :half :nearest-even :ieee))
                       (body/->ScalarStore output [slot component] half-value mask)]))]
    (body/make
     {:id [:paged-kv-append-body id]
      :parameters [(body/->KernelParameter key-rows :input :float
                                           [batch-size key-elements-per-token] :global
                                           (layout/row-major [batch-size key-elements-per-token] :float)
                                           :key-rows)
                   (body/->KernelParameter value-rows :input :float
                                           [batch-size value-elements-per-token] :global
                                           (layout/row-major [batch-size value-elements-per-token] :float)
                                           :value-rows)
                   (body/->KernelParameter slot-mapping :input :int [batch-size] :global
                                           (layout/row-major [batch-size] :int) :slot-mapping)
                   (body/->KernelParameter key-pages :inout :half
                                           [slots key-elements-per-token] :global
                                           (layout/row-major [slots key-elements-per-token] :half)
                                           :key-pages)
                   (body/->KernelParameter value-pages :inout :half
                                           [slots value-elements-per-token] :global
                                           (layout/row-major [slots value-elements-per-token] :half)
                                           :value-pages)]
      :stable-reads [(body/stable-read key-rows)
                     (body/stable-read value-rows)
                     (body/stable-read slot-mapping)]
      :indices [(body/->IndexBinding group-x :group 0)
                (body/->IndexBinding lane-x :local 0)
                (body/->IndexBinding lane :group 1)
                (body/->IndexCompute
                 component
                 (body/index-cast
                  (body/expression :add
                                   (body/expression :mul group-x workgroup-x)
                                   lane-x)
                  :long :exact))]
      :masks [(body/->Mask :append-key-active
                           [(body/predicate :lt component key-elements-per-token)])
              (body/->Mask :append-value-active
                           [(body/predicate :lt component value-elements-per-token)])]
      :operations
      [(body/->ScalarLoad (body/value slot :int) slot-mapping [lane]
                          nil nil :cached)
       (body/->ScalarCompute
        (body/value :append-slot-nonnegative :predicate)
        (body/scalar-expression :le :predicate [(body/literal 0 :int) slot]))
       (body/->ScalarCompute
        (body/value :append-slot-bounded :predicate)
        (body/scalar-expression :lt :predicate [slot (body/literal slots :int)]))
       (body/->IfRegion
        :append-slot-nonnegative
        [(body/->IfRegion
          :append-slot-bounded
          (vec (concat (converted "append-key" key-rows key-pages :append-key-active)
                       (converted "append-value" value-rows value-pages :append-value-active)
                       [(body/->Yield [])]))
          [(body/->Yield [])]
          [])
         (body/->Yield [])]
        [(body/->Yield [])]
        [])]
      :schedule {:strategy :paged-kv-append-component
                 :assignment :unique-slot
                 :rounding :nearest-even}
      :launch (launch/spec {:workgroup-size [workgroup-x 1]
                            :group-count [(long (quot (+ width (dec workgroup-x)) workgroup-x))
                                          batch-size]})
      :provenance {:dialect :kernel-body :source-dialect :paged-kv-append}
      :attributes {:kind :paged-kv-append :operation-id id
                   :input-dtype :float :storage-dtype :half
                   :no-write-alias true}})))
