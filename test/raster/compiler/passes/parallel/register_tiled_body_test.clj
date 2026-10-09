(ns raster.compiler.passes.parallel.register-tiled-body-test
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as body-emit]
            [raster.compiler.backend.gpu.segop-opencl :as segop-emit]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.passes.parallel.register-tiled-body :as register-tiled]))

(def ^:private small-tile
  {:block-m 4 :block-n 4 :block-k 2 :thread-m 2 :thread-n 2})

(defn- contraction
  [& [epilogue init]]
  (facts/from-components
   {:out 'C
    :free-axes [['i 8] ['j 8]]
    :contract-axes [['k 8]]
    :body '(raster.numeric/*
            (clojure.core/aget A (clojure.core/+ (clojure.core/* i 8) k))
            (clojure.core/aget B (clojure.core/+ (clojure.core/* k 8) j)))
    :opts (cond-> {} epilogue (assoc :epilogue epilogue)
                  (some? init) (assoc :init init))
    :dtype :float}))

(defn- contraction-with-dimensions
  [m n k]
  (facts/from-components
   {:out 'C
    :free-axes [['i m] ['j n]]
    :contract-axes [['p k]]
    :body (list 'raster.numeric/*
                (list 'clojure.core/aget 'A
                      (list 'clojure.core/+ (list 'clojure.core/* 'i k) 'p))
                (list 'clojure.core/aget 'B
                      (list 'clojure.core/+ (list 'clojure.core/* 'p n) 'j)))
    :opts {:init (float 0.0)}
    :dtype :float}))

(defn- operation-kinds
  [operations]
  (mapcat
   (fn [operation]
     (concat [(some-> operation class .getSimpleName)]
             (when (instance? raster.compiler.ir.kernel_body.ForLoop operation)
               (operation-kinds (:operations operation)))))
   operations))

(deftest explicit-fused-register-products-use-canonical-typed-fma
  (doseq [policy [:decomposed :fused]]
    (let [kernel (:kernel-body (register-tiled/lower (contraction)
                               {:tile small-tile :multiply-add policy}))
          expressions (filter #(instance? raster.compiler.ir.kernel_body.ScalarExpr %)
                              (tree-seq coll? seq kernel))
          fmas (filter #(= :fma (:op %)) expressions)]
      (is (= policy (get-in kernel [:schedule :multiply-add])))
      (if (= policy :fused)
        (do (is (= 4 (count fmas)))
            (is (every? #(= :float (:result-type %)) fmas)))
        (is (empty? fmas)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [source (body-emit/emit-scalar-kernel "register_fma" kernel
                       {:target-dialect target})]
          (is (= (= policy :fused) (boolean (re-find #" = fma\(" source))))))))
  (is (thrown? clojure.lang.ExceptionInfo
               (register-tiled/lower (contraction) {:tile small-tile :multiply-add :unknown}))))

(deftest explicit-wide-arithmetic-retains-float-storage-and-rounded-boundary
  (doseq [policy [:decomposed :fused]]
    (let [kernel (:kernel-body (register-tiled/lower (contraction)
                               {:tile small-tile :multiply-add policy :arithmetic-dtype :double}))
          expressions (filter #(instance? raster.compiler.ir.kernel_body.ScalarExpr %)
                              (tree-seq coll? seq kernel))
          casts (filter #(= :cast (:op %)) expressions)
          arithmetic (filter #(contains? #{:* :+ :fma} (:op %)) expressions)]
      (is (= kernel (body/validate! kernel)))
      (is (every? #(= :float (:dtype %)) (:parameters kernel)))
      (is (every? #(= :float (:dtype %)) (:allocations kernel)))
      (is (= :double (get-in kernel [:schedule :arithmetic-dtype])))
      (is (= 4 (count (filter #(and (= :double (:result-type %))
                                   (= {:rounding :exact :overflow :exact} (:options %))) casts))))
      (is (= 4 (count (filter #(and (= :float (:result-type %))
                                   (= {:rounding :nearest-even :overflow :ieee} (:options %))) casts))))
      (is (every? #(= :double (:result-type %)) arithmetic))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [source (body-emit/emit-scalar-kernel "register_wide" kernel {:target-dialect target})]
          (is (str/includes? source "double"))
          (is (boolean (re-find (if (= :opencl-portable target)
                                 #"rstr_narrow_f64_f32_rte" #"__double2float_rn") source)))))))
  (is (= (:kernel-body (register-tiled/lower (contraction) {:tile small-tile}))
         (:kernel-body (register-tiled/lower (contraction) {:tile small-tile :arithmetic-dtype :float}))))
  (is (thrown? clojure.lang.ExceptionInfo
               (register-tiled/lower (contraction) {:tile small-tile :arithmetic-dtype :half}))))

(deftest register-tile-does-not-discard-unmodeled-product-terms
  (doseq [expression ['(* 2.0 (* (aget A (+ (* i 5) k)) (aget B (+ (* k 2) j))))
                      '(+ 1.0 (* (aget A (+ (* i 5) k)) (aget B (+ (* k 2) j))))]]
    (let [proof (facts/from-components
                 {:out 'C :free-axes [['i 3] ['j 2]] :contract-axes [['k 5]]
                  :body expression :opts {:init (float 0.0)} :dtype :float})
          error (try (register-tiled/lower proof {:tile small-tile}) nil
                     (catch clojure.lang.ExceptionInfo error error))]
      (is (facts/facts? proof))
      (is (= :body-has-unmodeled-terms (:reason (facts/dense-matrix-view proof))))
      (is (some? error) "a matrix schedule must not silently drop scalar work")
      (is (= :body-has-unmodeled-terms (:missing-rule (ex-data error)))))))

(deftest transposed-operands-stage-in-physical-order-into-canonical-tiles
  (doseq [[variant expression parameter-index stage-index flat tile-width allocation-shape physical-shape]
          [[:nt '(* (aget A (+ (* i 5) k)) (aget B (+ (* j 5) k)))
            1 1 'register-b-index 2 [2 4] [7 5]]
           [:tn '(* (aget A (+ (* k 3) i)) (aget B (+ (* k 7) j)))
            0 0 'register-a-index 4 [4 2] [5 3]]]]
   (let [proof (facts/from-components
               {:out 'C :free-axes [['i 3] ['j 7]] :contract-axes [['k 5]]
                :body expression
                :opts {:init (float 0.0)} :dtype :float})
        kernel (:kernel-body (register-tiled/lower proof {:tile small-tile}))
        outer (first (:operations kernel))
        staging (nth (:operations outer) stage-index)
        [load store] (:operations staging)
        contiguous (body/expression :mod flat tile-width)
        strided (body/expression :floor-div flat tile-width)]
    (is (= variant (get-in kernel [:schedule :variant])))
    (is (= physical-shape (:shape (nth (:parameters kernel) parameter-index))))
    (is (= (if (= variant :nt) 'B 'A) (:buffer load)))
    (is (= (if (= variant :nt)
             [(body/expression :add 'register-block-col strided)
              (body/expression :add 'register-k-block contiguous)]
             [(body/expression :add 'register-k-block strided)
              (body/expression :add 'register-block-row contiguous)])
           (:coordinates load)))
    (is (= [contiguous strided] (:coordinates store)))
    (is (= allocation-shape (:shape (nth (:allocations kernel) parameter-index))))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (string? (body-emit/emit-scalar-kernel
                    "transposed_register_tile" kernel {:target-dialect target})))))))

(deftest checked-zero-identities-reach-the-register-tiled-body
  (doseq [init '[0.0 (float 0.0) (double (float 0))]]
    (let [proof (contraction nil init)
          emitted (segop-emit/generate-register-tiled-kernel-body proof 'C :tile small-tile)]
      (is (body/kernel-body? (:kernel-body emitted)))
      (is (= init (:neutral (facts/scalar-reduction-view proof)))))))

(deftest cooperative-register-tile-is-a-verified-scalar-kernel-body
  (let [kernel-body (:kernel-body
                     (register-tiled/lower (contraction) {:tile small-tile}))
        kinds (frequencies (operation-kinds (:operations kernel-body)))
        opencl (body-emit/emit-scalar-kernel "register_tile" kernel-body)
        cuda (body-emit/emit-scalar-kernel
              "register_tile" kernel-body {:target-dialect :cuda})
        hip (body-emit/emit-scalar-kernel
             "register_tile" kernel-body {:target-dialect :hip})]
    (is (body/kernel-body? kernel-body))
    (is (= [[4 2] [2 4]] (mapv :shape (:allocations kernel-body))))
    (is (= [2 2] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [2 2] (mapv #(launch/resolve-expression {} %)
                       (get-in kernel-body [:launch :group-count]))))
    (is (= 4 (get kinds "ForLoop")) "outer K, inner K and two cooperative staging loops are explicit")
    (is (= 2 (get kinds "WorkgroupBarrier")))
    (is (str/includes? opencl "__local"))
    (is (str/includes? opencl "barrier(CLK_LOCAL_MEM_FENCE)"))
    (is (str/includes? cuda "__shared__"))
    (is (str/includes? cuda "__syncthreads()"))
    (is (str/includes? hip "__shared__"))
    (is (str/includes? hip "__syncthreads()"))
    (testing "the nested cooperative schedule is valid OpenCL C"
      (if-not (zero? (:exit (shell/sh "sh" "-c" "command -v clang")))
        (is true "clang unavailable")
        (let [compiled (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                 "-fsyntax-only" "-" :in opencl)]
          (is (zero? (:exit compiled)) (:err compiled)))))))

(deftest symbolic-bounds-stay-in-the-register-tiled-abi-and-launch
  (let [form '(raster.par/contract C [[i m] [j n]] [[p k]]
                (* (aget A (+ (* i k) p)) (aget B (+ (* p n) j)))
                :init (float 0.0))
        routed (route/route-contraction form :dtype :float
                                        :candidate-families #{:register-tiled :portable})
        long-routed (route/route-contraction
                     form :dtype :float
                     :candidate-families #{:register-tiled :portable}
                     :scalar-types '{m :long n :long k :long})
        kernel (:kernel-body routed)
        long-kernel (:kernel-body long-routed)
        values {'m 8 'n 7 'k 5}]
    (is (= :regtiled (:strategy routed)))
    (is (body/kernel-body? kernel))
    (is (= '[A B C m n k]
           (mapv :name (:abi routed))))
    (is (= [8 7 5]
           (mapv #(launch/resolve-expression (fn [id] (get values id)) (:value %))
                 (:scalar-args routed))))
    (let [arguments (mapv (fn [slot]
                            (if (= :scalar (:kind slot))
                              {:type (:kernel-dtype slot)
                               :value (get values (:name slot))}
                              (Object.)))
                          (:abi routed))
          call (kcall/make (:artifact routed) arguments)]
      (is (= 56 (kcall/resolve-value call (:out-elems routed)))
          "the output extent is derived from ABI-bound dimensions at invocation"))
    (is (= [1 1]
           (mapv #(launch/resolve-expression (fn [id] (get values id)) %)
                 (get-in kernel [:launch :group-count]))))
    (is (= :regtiled (:strategy long-routed)))
    (is (= [:long :long :long]
           (mapv :kernel-dtype (filter #(= :scalar (:kind %)) (:abi long-routed)))))
    (is (body/kernel-body? long-kernel)
        "long bounds and explicit exact widening of local tile offsets validate as one body")
    (doseq [dialect [:opencl-intel :cuda :hip]]
      (is (string? (body-emit/emit-scalar-kernel
                    "symbolic_register_tile" kernel {:target-dialect dialect})))
      (is (string? (body-emit/emit-scalar-kernel
                    "symbolic_long_register_tile" long-kernel
                    {:target-dialect dialect}))))))

(deftest result-transform-is-alpha-renamed-per-microtile-store
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/*
                          (raster.numeric/+ acc (clojure.core/aget bias j)) scale)
                  :operands [{:sym 'bias :dtype :float
                              :map (axis-map/of-axes [['j 8]])}]
                  :scalars [{:sym 'scale :dtype :float}]
                  :dtype :float}
        emitted (segop-emit/generate-register-tiled-kernel-body
                 (contraction epilogue) 'C :tile small-tile)
        kernel-body (:kernel-body emitted)]
    (is (body/kernel-body? kernel-body))
    (is (= '[A B C bias scale] (mapv :name (:abi emitted))))
    (is (= '[bias] (:epilogue-operands emitted)))
    (is (= '[scale] (:epilogue-scalars emitted)))
    (is (= '[A B bias] (mapv :buffer (:stable-reads kernel-body))))
    (is (str/includes? (:source emitted) "bias["))
    (is (str/includes? (:source emitted) "* scale"))))

(deftest target-resource-limits-select-or-refuse-a-finite-tile
  (let [lowered (register-tiled/lower
                 (contraction)
                 {:descriptor {:execution {:max-workgroup-size 64}
                               :shared-local-memory 4096}})]
    (is (= {:block-m 32 :block-n 32 :block-k 16 :thread-m 4 :thread-n 4}
           (:tile lowered)))
    (is (= [8 8] (get-in lowered [:kernel-body :launch :workgroup-size]))))
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"cannot host any register-tiled"
       (register-tiled/lower
        (contraction)
        {:descriptor {:execution {:max-workgroup-size 8}
                      :shared-local-memory 128}}))))

(deftest padded-tail-coordinates-stay-in-the-int-domain
  (let [tile {:block-m 3 :block-n 5 :block-k 7 :thread-m 1 :thread-n 1}
        limits (register-tiled/int-coordinate-constraints [1 1 1] tile)
        m-limit (:maximum-extent (first limits))]
    (is (= 2147483646 m-limit)
        "the exact limit accounts for the last padded coordinate, not only the logical extent")
    (is (body/kernel-body?
         (:kernel-body
          (register-tiled/lower (contraction-with-dimensions m-limit 1 1) {:tile tile}))))
    (try
      (register-tiled/lower (contraction-with-dimensions (inc m-limit) 1 1) {:tile tile})
      (is false "an overflowing padded coordinate must decline before KernelBody emission")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :register-tiled-kernel-body-declined (:reason (ex-data exception))))
        (is (= :padded-coordinate-domain (:missing-rule (ex-data exception))))
        (is (= {:axis :m :dimension (inc m-limit) :tile-width 3
                :maximum-extent m-limit}
               (select-keys (ex-data exception)
                            [:axis :dimension :tile-width :maximum-extent])))))
    (is (= (inc (long Integer/MAX_VALUE))
           (:maximum-extent
            (first (register-tiled/int-coordinate-constraints
                    [Integer/MAX_VALUE 1 1]
                    {:block-m 64 :block-n 64 :block-k 16}))))
        "ordinary power-of-two tiles retain the complete positive int extent domain")))

(deftest dynamic-dispatch-uses-the-same-padded-coordinate-limits
  (let [tile {:block-m 3 :block-n 5 :block-k 7 :thread-m 1 :thread-n 1}
        guarded (#'route/guard-register-tiled-selector
                 {:kind :fixed-strategy :strategy :regtiled}
                 [{:strategy :regtiled :kernel-body {:schedule tile}}
                  {:strategy :portable-segred}]
                 (contraction-with-dimensions 'm 'n 'k))
        ordinary (#'route/guard-register-tiled-selector
                  {:kind :fixed-strategy :strategy :regtiled}
                  [{:strategy :regtiled :kernel-body {:schedule register-tiled/default-tile}}
                   {:strategy :portable-segred}]
                  (contraction-with-dimensions 'm 'n 'k))
        coordinate-cases (take-last 3 (:cases guarded))]
    (is (= :runtime-expression-cases (:kind guarded)))
    (is (= :regtiled (:default guarded)))
    (is (= [{:expression 'm :op :> :value 2147483646
             :strategy :portable-segred}
            {:expression 'n :op :> :value 2147483645
             :strategy :portable-segred}
            {:expression 'k :op :> :value 2147483646
             :strategy :portable-segred}]
           coordinate-cases)
        "dynamic admission projects the exact constraints used by static lowering")
    (is (= 9 (count (:cases guarded)))
        "positive dimensions, dense products and padded coordinates are guarded")
    (is (= 6 (count (:cases ordinary)))
        "power-of-two tiles need only positivity and capacity guards")
    (is (= (mapv (fn [dimension]
                   {:expression dimension :op :<= :value 0 :strategy :portable-segred})
                 '[m n k])
           (vec (take 3 (:cases ordinary)))))))

(deftest register-tiled-loop-updates-cannot-overflow-at-the-int-boundary
  (let [kernel-body (:kernel-body
                     (register-tiled/lower
                      (contraction-with-dimensions 1 1 Integer/MAX_VALUE)
                      {:tile register-tiled/default-tile}))
        source (body-emit/emit-scalar-kernel "register_tile_int_limit" kernel-body)]
    (is (str/includes?
         source
         "for (int rstr_register_k_block = 0; rstr_register_k_block < 2147483647;)"))
    (is (str/includes?
         source
         (str "(uint)(2147483647) - (uint)(rstr_register_k_block)"
              " <= (uint)(16)"))
        "the outer K loop breaks before its exiting increment could overflow int")
    (doseq [staging-index ["rstr_register_a_index" "rstr_register_b_index"]]
      (is (str/includes?
           source
           (str "(uint)(1024) - (uint)(" staging-index ") <= (uint)(256)"))
          "cooperative staging loops also guard their dynamic starting indices"))
    (is (str/includes?
         source
         (str "for (int rstr_register_k_inner = 0; rstr_register_k_inner < 16;"
              " rstr_register_k_inner += 1)"))
        "the finite inner tile loop's exiting increment remains representable")))

(deftest symbolic-long-k-keeps-a-long-guarded-induction-variable
  (let [kernel-body (:kernel-body
                     (register-tiled/lower
                      (contraction-with-dimensions 1 1 'depth)
                      {:tile register-tiled/default-tile
                       :scalar-types {'depth :long}}))
        source (body-emit/emit-scalar-kernel "register_tile_long_k" kernel-body)]
    (is (str/includes?
         source
         "for (long rstr_register_k_block = (long)(0); rstr_register_k_block < rstr_depth;)"))
    (is (str/includes?
         source
         (str "(ulong)(rstr_depth) - (ulong)(rstr_register_k_block)"
              " <= (ulong)(16)"))
        "a long K is not narrowed merely to reuse the conservative dispatch constraint")))

(deftest a-destination-reading-result-transform-stores-through-one-read-write-parameter
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/+ acc (raster.numeric/* beta (clojure.core/aget C (clojure.core/+ (clojure.core/* i 8) j))))
                  :operands [{:sym 'C :dtype :float
                              :map (axis-map/of-axes [['i 8] ['j 8]])}]
                  :scalars [{:sym 'beta :dtype :float}]
                  :dtype :float}
        emitted (segop-emit/generate-register-tiled-kernel-body
                 (contraction epilogue) 'C :tile small-tile)
        kernel-body (:kernel-body emitted)]
    (is (body/kernel-body? kernel-body))
    (is (= '[[A :input] [B :input] [C :inout] [beta :scalar]]
           (mapv (juxt :name :kind) (:abi emitted))))
    (is (empty? (:epilogue-operands emitted)))
    (is (= '[A B] (mapv :buffer (:stable-reads kernel-body)))
        "the destination is written, so it carries no stable-read contract")
    (is (re-find #"__global float\* out" (:source emitted))
        "the destination pointer is writable: no const qualifier")
    (is (re-find #"\? out\[" (:source emitted))
        "the destination element is loaded under the same store mask")
    (is (str/includes? (:source emitted) "(beta * "))))

(deftest mixed-static-shapes-retain-impossible-capacity-obligations
  (let [dimensions [65536 65536 'depth]
        conditions (register-tiled/admission-preconditions
                    dimensions register-tiled/default-tile)
        static-product (launch/product 65536 65536)]
    (is (register-tiled/static-capacity-overflow? dimensions)
        "a static pair cannot hide behind an unrelated symbolic dimension")
    (is (some #(= {:expression static-product :op :<= :value Integer/MAX_VALUE} %)
              conditions)
        "legacy guarded dispatch must retain the impossible condition and fall back")
    (is (not (register-tiled/static-capacity-overflow? [65535 32768 'depth])))))
