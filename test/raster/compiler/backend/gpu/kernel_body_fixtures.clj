(ns raster.compiler.backend.gpu.kernel-body-fixtures
  "Shared verifier, source-compile, and device fixtures for scheduled KernelBody operations."
  (:require [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]))

(defn workgroup-memory-body
  "A workgroup reverses one row through explicitly allocated local storage and a barrier."
  [width]
  (body/make
   {:id :c-family-workgroup-memory-gate
    :parameters [(body/->KernelParameter
                  'x :input :float [width] :global
                  (layout/row-major [width] :float) :input)
                 (body/->KernelParameter
                  'y :output :float [width] :global
                  (layout/row-major [width] :float) :result)]
    :stable-reads [(body/stable-read 'x)]
    :allocations [(body/->WorkgroupAllocation
                   'scratch :float [width] (layout/row-major [width] :float) 16)]
    :indices [(body/->IndexBinding 'lane :local 0)]
    :operations [(body/->ScalarLoad (body/value 'input-value :float)
                                    'x ['lane] nil nil :cached)
                 (body/->ScalarStore 'scratch ['lane] 'input-value nil)
                 (body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                                          (body/full-participation))
                 (body/->ScalarLoad
                  (body/value 'mirrored-value :float) 'scratch
                  [(body/expression :sub (dec width) 'lane)] nil nil :cached)
                 (body/->ScalarStore 'y ['lane] 'mirrored-value nil)]
    :schedule {}
    :launch (launch/spec {:workgroup-size [width] :group-count [1]
                          :shared-memory-bytes (* width 4)})
    :provenance {:dialect :compile-gate}
    :attributes {:kind :workgroup-memory}}))
