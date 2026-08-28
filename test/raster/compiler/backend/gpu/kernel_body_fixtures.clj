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

(defn async-staging-body
  "A workgroup stages one contiguous row, explicitly commits and waits, then consumes it."
  [width overlap]
  (body/make
   {:id :c-family-async-staging-gate
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
    :operations [(body/->AsyncWorkgroupCopy
                  'stage-x 'x [0] 'scratch [0] width 16 :cached overlap
                  (body/full-participation))
                 (body/->AsyncCommit 'stage-group ['stage-x])
                 (body/->AsyncWait ['stage-group] 0 :acquire (body/full-participation))
                 (body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                                          (body/full-participation))
                 (body/->ScalarLoad (body/value 'staged-value :float)
                                    'scratch ['lane] nil nil :cached)
                 (body/->ScalarStore 'y ['lane] 'staged-value nil)]
    :schedule {:async-staging {:groups 1 :depth 1}}
    :launch (launch/spec {:workgroup-size [width] :group-count [1]
                          :shared-memory-bytes (* width 4)})
    :provenance {:dialect :compile-gate}
    :attributes {:kind :async-staging}}))

(defn pipelined-staging-body
  "Two rotating workgroup stages. Each steady-state half waits and consumes one stage while the
  refill of the other stage remains pending across useful work and across the loop backedge."
  [width overlap]
  (let [barrier #(body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                                          (body/full-participation))
        copy (fn [id destination]
               (body/->AsyncWorkgroupCopy
                id 'x [0] destination [0] width 16 :cached overlap
                (body/full-participation)))]
    (body/make
     {:id :c-family-pipelined-staging-gate
      :parameters [(body/->KernelParameter
                    'x :input :float [width] :global
                    (layout/row-major [width] :float) :input)
                   (body/->KernelParameter
                    'y :output :float [width] :global
                    (layout/row-major [width] :float) :result)]
      :stable-reads [(body/stable-read 'x)]
      :allocations [(body/->WorkgroupAllocation
                     'stage-a :float [width] (layout/row-major [width] :float) 16)
                    (body/->WorkgroupAllocation
                     'stage-b :float [width] (layout/row-major [width] :float) 16)]
      :indices [(body/->IndexBinding 'lane :local 0)]
      :operations
      [(copy 'warm-a 'stage-a)
       (body/->AsyncCommit 'warm-group-a ['warm-a])
       (copy 'warm-b 'stage-b)
       (body/->AsyncCommit 'warm-group-b ['warm-b])
       (body/->PipelinedFor
        (body/value 'pipeline-iteration :int) 0 2 1 []
        [(body/->AsyncLoopArg 'carry-a 'warm-group-a)
         (body/->AsyncLoopArg 'carry-b 'warm-group-b)]
        [(body/->AsyncWait ['carry-a] 1 :acquire (body/full-participation))
         (barrier)
         (body/->ScalarLoad (body/value 'stage-a-value :float)
                            'stage-a ['lane] nil nil :cached)
         (barrier)
         (copy 'refill-a 'stage-a)
         (body/->AsyncCommit 'refill-group-a ['refill-a])
         (body/->AsyncWait ['carry-b] 1 :acquire (body/full-participation))
         (barrier)
         (body/->ScalarLoad (body/value 'stage-b-value :float)
                            'stage-b ['lane] nil nil :cached)
         (body/->ScalarCompute
          (body/value 'combined-stage-value :float)
          (body/scalar-expression :+ :float ['stage-a-value 'stage-b-value]))
         (body/->ScalarStore 'y ['lane] 'combined-stage-value nil)
         (barrier)
         (copy 'refill-b 'stage-b)
         (body/->AsyncCommit 'refill-group-b ['refill-b])
         (body/->PipelineYield [] ['refill-group-a 'refill-group-b])]
        [] ['pipeline-group-a 'pipeline-group-b]
        {:tail-policy :exact})
       (body/->AsyncWait ['pipeline-group-a 'pipeline-group-b] 0 :acquire
                         (body/full-participation))
       (barrier)]
      :schedule {:async-staging {:groups 2 :depth 2 :pipeline-loop true}}
      :launch (launch/spec {:workgroup-size [width] :group-count [1]
                            :shared-memory-bytes (* width 8)})
      :provenance {:dialect :compile-gate}
      :attributes {:kind :pipelined-staging}})))
