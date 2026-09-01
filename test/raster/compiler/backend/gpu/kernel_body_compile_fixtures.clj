(ns raster.compiler.backend.gpu.kernel-body-compile-fixtures
  "Generate production scheduled KernelBody sources for hardware-free CUDA/HIP CI gates."
  (:require [clojure.java.io :as io]
            [raster.arrays]
            [raster.compiler.backend.gpu.attention :as attention-emit]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as body-fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as body-emit]
            [raster.compiler.backend.gpu.segop-opencl :as segop-emit]
            [raster.compiler.backend.gpu.target :as gpu-target]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.attention-route :as attention-route]
            [raster.compiler.passes.parallel.contract-lower :as contract-lower]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.register-tiled-body :as register-tiled-body]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-schedule :as schedule]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.core :refer [deftm]]
            [raster.numeric]
            [raster.par]
            [raster.runtime.hardware :as hardware]))

(deftm public-c-family-dot
  "Public equation-first workload compiled by nvcc/hipcc without a physical device."
  (All [T] [left :- (Array T) right :- (Array T) n :- Long] :- Double
       (raster.par/reduce
        accumulator 0.0 index n
        (raster.numeric/+
         accumulator
         (raster.numeric/* (raster.arrays/aget left index)
                           (raster.arrays/aget right index))))))

(deftm public-c-family-map
  "Public equation-first scalar map compiled by nvcc/hipcc without a physical device."
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (raster.par/map! output index n float
                     (raster.numeric/* (float 2.0)
                                       (raster.arrays/aget input index)))))

(defn- reduction-artifact
  [dialect]
  (let [form (with-meta
               '(raster.par/reduce acc 0.0 i n
                                   (+ acc (* scale (clojure.core/aget values i))))
               {:raster.type/elem-type :float})
        node (soac/par-form->soac 'result form 901 :dtype :float)
        operation (first (soac-lower/lower-reduce node nil :dtype :float))]
    (segop-emit/generate-segred-kernel-body
     operation nil :dtype :float :scalar-types {'scale :float}
     :target-dialect dialect :kernel-name-prefix "workgroup_reduce")))

(defn- contraction-artifact
  [dialect descriptor]
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/*
                          (raster.numeric/+ acc (clojure.core/aget bias i)) scale)
                  :operands [{:sym 'bias :dtype :float
                              :map (axis-map/of-axes [['i 64]])}]
                  :scalars [{:sym 'scale :dtype :float}]
                  :dtype :float}
        form (concat
              '(raster.par/contract y [[i 64]] [[l 32]]
                                    (* (aget A (+ (* i 32) l)) (aget x l)))
              [:epilogue epilogue])
        verified (contraction-facts/contraction-facts form :dtype :half)
        segred (contract-lower/contract-form->segred
                form :dtype :half :facts verified)
        planned (contraction-schedule/plan-portable-body verified segred descriptor)]
    (when-not (:ok planned)
      (throw (ex-info "portable contraction compile fixture did not schedule" planned)))
    (segop-emit/generate-contraction-kernel-body
     (:body planned) :target-dialect dialect
     :kernel-name-prefix "portable_contraction")))

(defn- register-tiled-contraction-source
  [dialect]
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/*
                          (raster.numeric/+ acc (clojure.core/aget bias j)) scale)
                  :operands [{:sym 'bias :dtype :float
                              :map (axis-map/of-axes [['j 64]])}]
                  :scalars [{:sym 'scale :dtype :float}]
                  :dtype :float}
        facts (contraction-facts/from-components
               {:out 'C
                :free-axes [['i 64] ['j 64]]
                :contract-axes [['k 32]]
                :body '(raster.numeric/*
                        (clojure.core/aget A (clojure.core/+ (clojure.core/* i 32) k))
                        (clojure.core/aget B (clojure.core/+ (clojure.core/* k 64) j)))
                :opts {:epilogue epilogue}
                :dtype :float})
        kernel-body (:kernel-body (register-tiled-body/lower facts {}))]
    (body-emit/emit-scalar-kernel
     "register_tiled_contraction" kernel-body {:target-dialect dialect})))

(defn- segmented-fold-map-artifact
  [dialect]
  (let [source
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[segment nsegments]] index width
                 [[maximum -1.0E38 :float width
                   (clojure.core/max
                    maximum
                    (clojure.core/aget values
                                       (clojure.core/+ (clojure.core/* segment width) index)))]
                  [denominator 0.0 :float width
                   (clojure.core/+
                    denominator
                    (Math/exp
                     (clojure.core/-
                      (clojure.core/aget
                       values (clojure.core/+ (clojure.core/* segment width) index))
                      maximum)))]]
                 [(clojure.core//
                   (Math/exp
                    (clojure.core/-
                     (clojure.core/aget
                      values (clojure.core/+ (clojure.core/* segment width) index))
                     maximum))
                   denominator)])]
               effect)
        program (:program
                 (typed-route/attempt
                  source :float {'values :float 'out :float}
                  {:scalar-types {'nsegments :int 'width :int}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'values :float 'out :float}
                             :scalar-types {'nsegments :int 'width :int}}))
        operation (first (:operations (first (:equations scheduled))))]
    (segop-emit/generate-segfoldmap-kernel
     operation :target-dialect dialect :kernel-name-prefix "segmented_fold_map"
     :array-types {'values :float 'out :float}
     :scalar-types {'nsegments :int 'width :int})))

(defn- problem []
  (attention/make
   {:id :c-family-compile-gate
    :query (attention/packed-query-batch
            {:values 'q :row-offsets 'q-row-offsets :positions 'q-positions :total-tokens 4})
    :k-pages 'k-pages :v-pages 'v-pages
    :route (attention/dense-paged-route
            {:page-table 'page-table :lengths 'kv-lengths
             :start-positions 'kv-start-positions :pages-per-sequence 4})
    :output 'output :batch-size 2 :q-heads 8 :kv-heads 2
    :qk-head-dim 64 :value-head-dim 64 :page-size 16 :physical-pages 8
    :scale 0.125 :k-layout :kv-head-major :v-layout :page-major
    :visibility (attention/visibility {:causal? true :window-left 31 :window-right 0})}))

(def ^:private targets
  {:cuda {:suffix ".cu"
          :descriptor {:device-type :gpu :backend :cuda :vendor "NVIDIA"
                       :compute-capability [8 0]
                       :subgroup-size 32 :max-workgroup-size 1024}}
   :hip {:suffix ".hip"
         :descriptor {:device-type :gpu :backend :hip :vendor "AMD"
                      :subgroup-size 32 :max-workgroup-size 1024}}})

(defn- equation-first-artifacts
  [target descriptor]
  (let [device-id (keyword (str (name target) ":equation-first-compile-gate"))
        capabilities (cond-> {:subgroup-sizes [(:subgroup-size descriptor)]
                              :warp-size (:subgroup-size descriptor)
                              :max-workgroup-size (:max-workgroup-size descriptor)
                              :shared-local-memory 65536
                              :total-eus 64}
                       (:compute-capability descriptor)
                       (assoc :compute-capability (:compute-capability descriptor))

                       (= :hip target)
                       (assoc :gfx-arch :gfx1100))]
    (hardware/register-target-device!
     device-id {:type target
                :name (str "Synthetic " (name target) " public compile gate")
                :capabilities capabilities})
    (vec
     (concat
      (:kernels (equation-first/compile
                 #'public-c-family-dot {:target device-id :dtype :float}))
      (:kernels (equation-first/compile
                 #'public-c-family-map {:target device-id :dtype :float}))))))

(defn- write-artifact!
  [directory suffix label artifact]
  (let [file (io/file directory (str label suffix))]
    (spit file (:source artifact))
    (.getAbsolutePath file)))

(defn- write-source!
  [directory suffix label source]
  (let [file (io/file directory (str label suffix))]
    (spit file source)
    (.getAbsolutePath file)))

(defn emit-target!
  [root target]
  (let [{:keys [suffix descriptor]} (get targets target)
        dialect (gpu-target/kernel-body-c-dialect descriptor)
        _ (when-not dialect
            (throw (ex-info "unknown compile-gate target"
                            {:target target :supported (set (keys targets))})))
        directory (io/file root (name target))
        _ (.mkdirs directory)
        plan (:plan (attention-route/route!
                     (problem)
                     (assoc descriptor :segmented-weighted-reduction-schedule :reference)))
        cooperative-schedule (:schedule (schedule/plan-subgroup-online plan descriptor))
        pipelined-schedule (:schedule
                            (schedule/plan-subgroup-online-pipelined plan descriptor))
        swizzled-pipelined-schedule
        (:schedule
         (schedule/plan-subgroup-online-pipelined
          plan (assoc descriptor :segmented-weighted-reduction-layout-swizzle :xor-32)))
        tiled-schedule (:schedule
                        (schedule/plan-subgroup-online-tiled
                         plan (assoc descriptor
                                     :segmented-weighted-reduction-history-tile-size 4)))
        cooperative (attention-emit/emit-fp16-cooperative
                     plan cooperative-schedule dialect)
        pipelined (attention-emit/emit-fp16-pipelined
                   plan pipelined-schedule dialect descriptor)
        swizzled-pipelined (attention-emit/emit-fp16-pipelined
                            plan swizzled-pipelined-schedule dialect descriptor)
        tiled (attention-emit/emit-fp16-tiled-history plan tiled-schedule dialect)
        public-artifacts (equation-first-artifacts target descriptor)]
    (into [(write-artifact! directory suffix "cooperative" cooperative)
           (write-artifact! directory suffix "pipelined-attention" pipelined)
           (write-artifact! directory suffix "swizzled-pipelined-attention"
                            swizzled-pipelined)
           (write-artifact! directory suffix "workgroup-reduction"
                            (reduction-artifact dialect))
           (write-artifact! directory suffix "portable-contraction"
                            (contraction-artifact dialect descriptor))
           (write-artifact! directory suffix "segmented-fold-map"
                            (segmented-fold-map-artifact dialect))
           (write-source! directory suffix "register-tiled-contraction"
                          (register-tiled-contraction-source dialect))
           (write-source! directory suffix "workgroup-memory"
                          (body-emit/emit-scalar-kernel
                           "workgroup_memory" (body-fixtures/workgroup-memory-body 32)
                           {:target-dialect dialect}))
           (write-source! directory suffix "swizzled-workgroup-memory"
                          (body-emit/emit-scalar-kernel
                           "swizzled_workgroup_memory"
                           (body-fixtures/swizzled-workgroup-memory-body 32)
                           {:target-dialect dialect}))
           (write-source! directory suffix "async-staging"
                          (body-emit/emit-scalar-kernel
                           "async_staging"
                           (body-fixtures/async-staging-body 32 :preferred)
                           {:target-dialect dialect :target-features descriptor}))
           (write-source! directory suffix "pipelined-staging"
                          (body-emit/emit-scalar-kernel
                           "pipelined_staging"
                           (body-fixtures/pipelined-staging-body 32 :preferred)
                           {:target-dialect dialect :target-features descriptor}))]
          (concat
           (map-indexed
            (fn [index artifact]
              (write-artifact! directory suffix (str "equation-first-" index) artifact))
            public-artifacts)
           (map-indexed
            (fn [index artifact]
              (write-artifact! directory suffix (str "tiled-" index) artifact))
            (executable/artifacts tiled))))))

(defn -main
  [& [root]]
  (let [root (or root "gpu-compile-gates")]
    (doseq [target [:cuda :hip]
            file (emit-target! root target)]
      (println file))))
