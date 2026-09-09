(ns raster.compiler.passes.parallel.recursive-packed-stage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.passes.parallel.staged-scalar-body :as staged]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]))

(defn- source-components [width]
  (let [axes [['i 2] ['sb 2] ['blk 2] ['t width]]
        layout (am/of-axes axes)]
    {:out 'out :free-axes [['i 2]] :contract-axes (vec (rest axes)) :dtype :byte
     :body (list 'raster.numeric/* (list 'aget 'a (am/index-expr layout))
                 (list 'aget 'b (am/index-expr layout)))
     :opts {:operands [{:sym 'a :map layout} {:sym 'b :map layout}]
            :stages [{:axis 'sb :extent 2 :dtype :float :init 0.0 :lift 'inner}
                     {:axis 'blk :extent 2 :dtype :double :init 0.0 :lift 'inner}
                     {:axis 't :extent width :dtype :int :init 0}]}}))

(deftest recursive-packed-stages-refuse-unproved-contracts
  (doseq [[case-id components]
          [[:long-inner (assoc-in (source-components 4) [:opts :stages 2 :dtype] :long)]
           [:nonzero-init (assoc-in (source-components 4) [:opts :stages 2 :init] 1)]
           [:unaligned-extent (source-components 3)]
           [:wrong-map (assoc-in (source-components 4) [:opts :operands 0 :map]
                                (am/of-axes '[[sb 2] [i 2] [blk 2] [t 4]]))]
           [:extra-factor (update (source-components 4) :body #(list '* % 2))]
           [:escaped-axis (assoc-in (source-components 4) [:opts :stages 0 :lift] '(* inner t))]]]
    (is (thrown? clojure.lang.ExceptionInfo (staged/lower (facts/from-components components)))
        (name case-id))))

(deftest recursive-packed-stages-preserve-mixed-outer-dtypes
  (let [scheduled (staged/lower (facts/from-components (source-components 4)))
        loops (filter #(and (map? %) (contains? % :iter-args))
                      (tree-seq coll? seq (:body scheduled)))]
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (let [artifact (target/emit-artifact "recursive_packed_mixed" scheduled dialect)]
        (is (string? (:source artifact)))))
    (is (= [:float :double :int] (mapv #(get-in % [:results 0 :type]) loops)))))

(defn- reference [a b super-scale sub-scale]
  (vec
   (for [i (range 2) j (range 3)]
     (reduce
      (fn [outer sb]
        (let [middle
              (reduce
               (fn [mid blk]
                 (let [dot (reduce + 0
                                   (for [t (range 4)]
                                     (* (long (aget ^bytes a (+ (* i 16) (* sb 8) (* blk 4) t)))
                                        (long (aget ^bytes b (+ (* j 16) (* sb 8) (* blk 4) t))))))]
                   (float (+ mid (float (* (float dot)
                                           (aget ^floats sub-scale (+ (* j 2) blk))))))))
               (float 0) (range 2))]
          (float (+ outer (float (* middle (aget ^floats super-scale (+ (* i 2) sb))))))))
      (float 0) (range 2)))))

(deftest public-three-stage-packed-fold-preserves-each-rounding-boundary
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "recursive packed staged contraction")
    (let [a (byte-array (take 32 (cycle [-128 127 3 -5 0 17])))
          b (byte-array (take 48 (cycle [127 -128 -11 0 9])))
          super-scale (float-array [0.13 0.71 0.93 0.27])
          sub-scale (float-array [0.19 0.37 0.59 0.83 0.41 0.67])
          out (float-array (repeat 6 Float/NaN))
          compiled (equation-first/compile #'fixtures/packed-three-stage!
                                          {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compiled [a b super-scale sub-scale out])
          output-id (some (fn [[id node]] (when (identical? out (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (is (= :none (get-in compiled [:stats :fallback])))
        (is (= :staged-scalar (get-in compiled [:kernels 0 :attributes :kernel-body :schedule :strategy])))
        (link/run! executable)
        (is (= (reference a b super-scale sub-scale) (vec (link/download executable output-id))))
        (finally (link/close! executable))))))
