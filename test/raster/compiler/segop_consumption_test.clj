(ns raster.compiler.segop-consumption-test
  "THE SEGOP BOUNDARY IS REAL: GPU and JVM SIMD backends CONSUME what segop-lower computed.
   Device-free.

   `segop-lower` lowers every par form with the real target device into a first-class typed equation.
   `opencl-pass` used to re-lower each form from scratch with `:ze:0` hardcoded. The equation is now
   consumed directly, with no binder metadata and no second lowering.

   Now the stored SegOp is consumed. A raw binding form first enters the same whole-program typed
   scheduler, while only bare compatibility expressions request singleton scheduling. Both paths
   remain counted so bypasses are visible rather than silent."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.segop-lower-pass :as slp]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed-frontend]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.backend.gpu.opencl-pass :as op]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.pipeline :as pipeline]))

(def ^:private map-form
  '(let* [o (raster.par/map! O i 8192 nil (clojure.core/* (clojure.core/aget X i) 2.0))] o))
(def ^:private reduce-form
  '(let* [r (raster.par/reduce acc 0.0 i 8192 (clojure.core/+ acc (clojure.core/aget X i)))] r))

(defn- lowered [form] (:form (slp/segop-lower-pass form {:target-device :ze:0 :dtype :double})))
(defn- run [form] (op/opencl-pass form :device-id :ze:0 :dtype :double :min-elements 0))
(defn- lowered-cpu [form dtype]
  (:form (slp/segop-lower-pass form {:target-device :cpu:0 :dtype dtype})))
(defn- run-simd [form] (par-simd/simd-pass form :min-elements 0))
(defn- norm [src] (str/replace (str src) #"_\d{3,}" "_N"))

(deftest a-lowered-binding-is-consumed-not-relowered
  (doseq [[label form key] [["par/map!" map-form :ze-maps] ["par/reduce" reduce-form :ze-reduces]]]
    (testing label
      (let [st (:stats (run (lowered form)))]
        (is (= 1 (get st key)) "one kernel")
        (is (= 1 (:segop-reused st)) "…from the SegOp segop-lower attached")
        (is (nil? (:segop-relowered st)) "…and NOT re-lowered")))))

(deftest an-unlowered-binding-form-is-scheduled-once-and-consumed
  (testing "door C may hand opencl-pass a walked body with no prior middle-end pass; the backend
            schedules the complete program once and then consumes its equations"
    (doseq [[label form] [["par/map!" map-form] ["par/reduce" reduce-form]]]
      (testing label
        (let [st (:stats (run form))]
          (is (= 1 (:segop-reused st)))
          (is (nil? (:segop-relowered st)))
          (is (= :typed-soac (get-in st [:direct-scheduling :route]))))))))

(deftest consuming-is-a-refactor-the-kernel-is-identical
  (testing "same kernel source whether scheduling happened before or at backend entry"
    (doseq [form [map-form reduce-form]]
      (is (= (norm (:source (first (:kernels (run (lowered form))))))
             (norm (:source (first (:kernels (run form))))))))))

(deftest direct-binding-scheduling-preserves-cross-equation-fusion
  (let [form '(let* [tmp (raster.par/pmap i n double
                                           (double
                                            ^double
                                            (* (clojure.core/aget X i)
                                               (clojure.core/aget X i))))
                      total (raster.par/reduce acc 0.0 j n
                                               (+ acc (clojure.core/aget tmp j)))]
                     total)
        compiled (op/opencl-pass
                  form :device-id :ze:0 :dtype :double :min-elements 0
                  :array-types {'X :double} :scalar-types {'n :long})
        kernel (first (:kernels compiled))]
    (is (= 1 (count (:kernels compiled))))
    (is (= 1 (get-in compiled [:stats :ze-reduces])))
    (is (= 1 (get-in compiled [:stats :segop-reused])))
    (is (nil? (get-in compiled [:stats :segop-relowered])))
    (is (= :typed-soac (get-in compiled [:stats :direct-scheduling :route])))
    (is (not (str/includes? (:source kernel) "TMP"))
        "the direct backend must not rematerialize a fused semantic intermediate")))

(deftest the-fallback-uses-the-real-device-id
  (testing "re-lowering used `:ze:0` hardcoded. A registered non-:ze:0 target must reach
            lower-soac — observable because compute-launch-params asks the descriptor for THAT id"
    (let [calls (atom [])
          orig raster.compiler.passes.parallel.soac-lower/lower-soac]
      (with-redefs [typed-frontend/form->program (fn [& _] nil)
                    raster.compiler.passes.parallel.soac-lower/lower-soac
                    (fn [soac device-id & opts] (swap! calls conj device-id) (apply orig soac :ze:0 opts))]
        (op/opencl-pass map-form :device-id :ze:1 :dtype :double :min-elements 0))
      (is (= [:ze:1] @calls) (str "lower-soac was called with " @calls ", not the pass's device-id")))))

(deftest a-body-position-form-is-consumed-too
  (testing "a body-position equation is consumed exactly like a binding equation"
    (let [form '(let* [n 8192] (raster.par/map! O i n nil (clojure.core/* (clojure.core/aget X i) 2.0)))
          st (:stats (run (lowered form)))]
      (is (= 1 (:ze-maps st)))
      (is (= 1 (:segop-reused st)) "consumed from the first-class equation")
      (is (nil? (:segop-relowered st))))))

(deftest a-body-position-reduction-keeps-the-existing-resultless-envelope
  (let [form '(let* [n 8192]
                    (raster.par/reduce acc 0.0 i n
                                       (+ acc (clojure.core/aget X i))))
        program (lowered form)
        equation (first (:equations program))
        st (:stats (run program))]
    (is (empty? (:results equation)))
    (is (nil? (:algorithm equation)))
    (is (= 1 (:segop-reused st)))
    (is (= 1 (:ze-reduces st)))))

(deftest jvm-simd-consumes-the-boundary-too
  (testing "matching map/reduce SegOps are reused rather than independently re-derived"
    (doseq [[label form stat-key]
            [["par/map!" map-form :simd-maps]
             ["par/reduce" reduce-form :simd-reduces]]]
      (testing label
        (let [st (:stats (run-simd (lowered-cpu form :double)))]
          (is (= 1 (get st stat-key)))
          (is (= 1 (:segop-reused st)))
          (is (nil? (:segop-relowered st))))))))

(deftest jvm-simd-treats-the-scheduled-dtype-as-authoritative
  (testing "the scheduled operation retains the compiler's dtype even when compatibility source
            lacks enough metadata to re-infer it"
    (let [st (:stats (run-simd (lowered-cpu map-form :float)))]
      (is (= 1 (:simd-maps st)))
      (is (= 1 (:segop-reused st)))
      (is (nil? (:segop-relowered st))))))

(deftest jvm-simd-does-not-repeat-fusion-after-scheduling
  (testing "typed fusion is the sole fusion authority; the backend consumes both scheduled maps"
    (let [form '(let* [tmp-step (raster.par/map! TMP i 8192 nil
                                                 (clojure.core/* (clojure.core/aget X i) 2.0))
                       out-step (raster.par/map! O j 8192 nil
                                                 (clojure.core/+ (clojure.core/aget TMP j) 1.0))]
                      out-step)
          st (:stats (run-simd (lowered-cpu form :double)))]
      (is (zero? (:fused st)))
      (is (= 2 (:simd-maps st)))
      (is (= 2 (:segop-reused st)))
      (is (nil? (:segop-relowered st))))))

(deftest jvm-simd-does-not-turn-an-implementation-bug-into-scalar-fallback
  (testing "an unstructured re-lowering exception is a compiler bug, not an unsupported SIMD form"
    (with-redefs [typed-frontend/form->program (fn [& _] nil)
                  soac/par-form->soac
                  (fn [& _] (throw (NullPointerException. "simulated SIMD lowering bug")))]
      (is (thrown-with-msg? NullPointerException #"simulated SIMD lowering bug"
                            (run-simd map-form))))))

(deftest fused-reduction-crosses-typed-soac-and-both-scheduled-backends
  (let [source '(let* [tmp (raster.par/pmap i n double
                                            (double
                                             ^double
                                             (* (clojure.core/aget X i)
                                                (clojure.core/aget X i))))
                       total (raster.par/reduce acc 0.0 j n
                                                (+ acc (clojure.core/aget tmp j)))]
                      total)
        schedule (fn [target]
                   (:form
                    (pipeline/schedule-parallel-form
                     source {:target-device target :dtype :double
                             :array-types {'X :double}
                             :scalar-types {'n :long}})))
        gpu-program (schedule :ze:0)
        cpu-program (schedule :cpu:0)
        equation (first (:equations gpu-program))
        gpu (run gpu-program)
        simd (run-simd cpu-program)
        artifact (first (:kernels gpu))]
    (testing "fusion is preserved as a typed functional algorithm in the common envelope"
      (is (= 1 (count (:equations gpu-program))))
      (is (some? (:algorithm equation)))
      (is (not-any? #{'tmp} (flatten (:algorithm equation))))
      (is (= :typed-soac (:algorithm-dialect (first (:operations equation))))))
    (testing "JVM SIMD consumes the SegRed derived from that algorithm"
      (is (= 1 (get-in simd [:stats :segop-reused])))
      (is (= 1 (get-in simd [:stats :simd-reduces]))))
    (testing "GPU emission consumes the same SegRed through verified KernelBody"
      (is (= 1 (get-in gpu [:stats :segop-reused])))
      (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
      (is (= :typed-soac (get-in artifact [:attributes :kernel-body
                                           :provenance :algorithm-dialect])))
      (is (re-find #"element_value_.* = .* \* " (:source artifact)))
      (is (not (str/includes? (:source artifact) "TMP"))))))
