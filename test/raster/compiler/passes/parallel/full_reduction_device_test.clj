(ns raster.compiler.passes.parallel.full-reduction-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl-probe]))

(deftm dot-into!
  [x :- (Array float) y :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/contract out [] [[i n]]
    (* (arrays/aget x i) (arrays/aget y i)) :init (float 0.0)))

(defn- emitted [n]
  (let [source (list 'let* ['result (list 'raster.par/contract 'out [] [['i n]]
                                        '(* (aget x i) (aget y i)))] 'out)
        scheduled (:form (pipeline/schedule-parallel-form
                          source {:dtype :float :target-device :ze:0
                                  :array-types {'x :float 'y :float 'out :float}
                                  :scalar-types {'n :long}}))]
    (opencl/opencl-pass scheduled :dtype :float :compile-spirv? false)))

(deftest rank-zero-contractions-use-the-complete-reduction-graph
  (doseq [[n phases] [[0 1] [8 1] [1025 2] ['n 2]]]
    (let [result (emitted n)
          graph (dispatch/default-alternative (first (:dispatches result)))]
      (is (= phases (count (:nodes graph))))
      (is (= ['out] (mapv :id (:outputs graph))))
      (is (= 1 (:elements (first (:outputs graph)))))
      (is (some #{'raster.compiler.pipeline/invoke-scheduled-executable!} (flatten (:form result))))
      (is (not-any? #{'raster.gpu.ze-runtime/invoke-registered-reduction-kernel 'clojure.core/aget}
                    (flatten (:form result))))
      (is (zero? (get-in result [:stats :segop-relowered] 0))))))

(deftest graph-scalars-retain-logical-types-with-checked-node-narrowing
  (let [graph (dispatch/default-alternative (first (:dispatches (emitted 'n))))
        buffers (into {} (map (fn [b] [(:id b) (Object.)]))
                      (concat (:inputs graph) (:outputs graph) (:temporaries graph)))
        call (graph-call/make graph buffers {'n {:type :long :value 1025}})
        uses (for [node (:nodes call)
                   [slot argument value] (map vector (get-in node [:call :artifact :abi])
                                              (get-in node [:call :artifact :arguments])
                                              (get-in node [:call :arguments]))
                   :when (= 'n argument)]
               [(:kernel-dtype slot) (:type value) (:value value)])]
    (is (= graph (executable/validate! graph)))
    (is (= #{[:int :int 1025] [:long :long 1025]} (set uses)))
    (is (thrown? ArithmeticException
                 (graph-call/make graph buffers {'n {:type :long :value (inc (long Integer/MAX_VALUE))}})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (graph-call/make graph buffers {'n {:type :long :value -1}})))
    (let [retyped (-> graph
                      (update :scalars #(mapv (fn [s] (assoc s :dtype :double)) %))
                      (update :abi #(mapv (fn [slot]
                                            (if (= :scalar (:kind slot))
                                              (assoc slot :dtype :double :kernel-dtype :double)
                                              slot)) %)))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar dtype differs"
                            (executable/validate! retyped))))))

(deftest rank-zero-admission-does-not-discard-unproved-coordinate-or-alias-contracts
  (doseq [contract ['(raster.par/contract out [] [[i 4] [j 6]] (aget x (+ (* i 6) j)))
                    '(raster.par/contract out [] [[i 8]] (aget x (* i 2)))
                    '(raster.par/contract out [] [[i 8]] (aget out i))]]
    (is (nil? (frontend/form->program (list 'let* ['result contract] 'out)
                                     {:dtype :float :array-types {'x :float 'out :float}})))))

(deftest public-rank-zero-contraction-is-resident
  (doseq [target [:ocl:0 :ze:0]]
    (let [descriptor (pipeline/compile-gpu-program #'dot-into! target :dtype :float)]
      (is (= [:executable] (mapv :convention (:steps descriptor))))
      (is (empty? (:allocs descriptor))))))

(deftest rank-zero-and-resident-scalar-reductions-compose
  (let [result (route/attempt
                '(let* [a (raster.par/contract out [] [[i n]] (* (aget x i) (aget y i)))
                        total (raster.par/reduce acc 0.0 j n (+ acc (aget x j)))
                        b (raster.par/map-void! k n (aset z k (* total (aget x k))))]
                   [out b])
                :float {'x :float 'y :float 'out :float 'z :float}
                {:resident-reductions? true})]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= 1 (get-in result [:stats :resident-reductions])))
    (is (get-in result [:stats :typed-validated]))))

(deftest jvm-rank-zero-contraction-preserves-caller-owned-storage
  (doseq [n [0 8 1025]]
    (let [x (float-array (max 1 n) 1.0)
          y (float-array (max 1 n) 0.5)
          out (float-array [123.0 456.0])]
      (is (identical? out (dot-into! x y out (long n))))
      (is (= [(* n 0.5) 456.0] (vec out))))))

(defn- run-device! [target]
  (let [descriptor (pipeline/compile-gpu-program #'dot-into! target :dtype :float)]
    (gpu/with-gpu-session [session target]
      (doseq [n [0 8 1025]]
        (let [x (float-array (map #(float (- (mod % 7) 3)) (range (max 1 n))))
              y (float-array (repeat (max 1 n) 0.5))
              out (float-array [123.0 456.0])
              arguments [x y out (long n)]
              expected (reduce + 0.0 (map #(* (double %1) %2) (take n x) (take n y)))
              program (fixture/instantiate! session descriptor arguments
                                            {'x :input 'y :input 'out :output})]
          (try
            (link/upload! (:executable program)
                          (get-in program [:lowering :certificate :bindings 'out]) out)
            (dotimes [_ 2]
              (let [actual (get (fixture/run! program arguments) 'out)]
                (is (= expected (double (aget ^floats actual 0))))
                (is (= 456.0 (double (aget ^floats actual 1))) "only out[0] is written")))
            (finally (fixture/close! program))))))))

(defn- emitted-reduce-into [n combine identity]
  (opencl/opencl-pass
   (with-meta
     (list 'raster.par/reduce-into 'out 'acc identity 'i n
           (list combine 'acc '(* (aget x i) (aget y i))))
     {:raster.type/elem-type :float})
   :device-id :ze:0 :dtype :float :compile-spirv? false
   :array-types {'x :float 'y :float 'out :float} :scalar-types {'n :long}))

(deftest direct-reduce-into-reuses-typed-storage-and-monoid-scheduling
  (doseq [[combine identity] [['+ 0.0] ['* 1.0]]
          [n phases] [[0 [:single]] [8 [:single]]
                      [1025 [:block-local :cross-block]] ['n [:block-local :cross-block]]]]
    (let [result (emitted-reduce-into n combine identity)
          graph (dispatch/default-alternative (first (:dispatches result)))]
      (is (= :typed-soac (get-in result [:stats :direct-scheduling :route])))
      (is (= phases (mapv #(get-in % [:operation :attributes :phase]) (:nodes graph))))
      (is (= ['out] (mapv :id (:outputs graph))))
      (is (= 1 (get-in graph [:outputs 0 :elements])))
      (is (zero? (get-in result [:stats :segop-relowered] 0))))))

(deftest direct-reduce-into-rejects-unproved-storage-and-coordinates
  (doseq [body ['(+ acc (aget out i)) '(+ acc (aget x (* i 2)))]]
    (is (nil? (frontend/form->program
               (list 'let* ['result (list 'raster.par/reduce-into 'out 'acc 0.0 'i 8 body)]
                     'out)
               {:dtype :float :array-types {'x :float 'out :float}})))))

(defn- run-staged-dispatch! [target kernel-dispatch combine identity]
  (let [register! (requiring-resolve
                   (symbol (if (= target :ocl:0)
                             "raster.gpu.ocl-runtime" "raster.gpu.ze-runtime")
                           "register-kernel-dispatch!"))]
    (register! kernel-dispatch)
    (doseq [n [0 8 1025]]
      (let [x (float-array (max 1 n) 1.0)
            y (float-array (max 1 n) 0.5)
            out (float-array [123.0 456.0])]
        (is (identical? out (gpu/invoke-staged-executable!
                            target (:id kernel-dispatch) [x y out (long n)])))
        (is (= [(float (reduce combine identity (repeat n 0.5))) 456.0] (vec out))
            "ordinary staged invocation preserves the unwritten output tail")))))

(defn- run-staged! [target]
  (doseq [[result combine identity] [[(emitted 'n) + 0.0]
                                     [(emitted-reduce-into 'n '+ 0.0) + 0.0]
                                     [(emitted-reduce-into 'n '* 1.0) * 1.0]]]
    (run-staged-dispatch! target (first (:dispatches result)) combine identity)))

(deftest opencl-rank-zero-contraction-writes-the-resident-result
  (if @opencl-probe/opencl-available?
    (do (run-device! :ocl:0) (run-staged! :ocl:0))
    (opencl-probe/opencl-skip! "rank-zero contraction graph")))

(deftest level-zero-rank-zero-contraction-writes-the-resident-result
  (if @gpu-probe/gpu-available?
    (do (run-device! :ze:0) (run-staged! :ze:0))
    (gpu-probe/gpu-skip! "rank-zero contraction graph")))
