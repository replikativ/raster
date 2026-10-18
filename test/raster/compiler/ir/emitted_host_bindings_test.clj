(ns raster.compiler.ir.emitted-host-bindings-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.contractions :as contractions]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted]
            [raster.compiler.ir.emitted-parallel-program-call :as call]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.dl.attention :as attention]
            [raster.runtime.hardware :as hardware]))

(defn- host-equation [values id operands result expression & [parameters]]
  (let [algorithm (soac/make
                   (soac/default-program-facts
                    {:values (select-keys values (conj operands result)) :inputs operands
                     :equations {id (soac/default-equation-facts)}})
                   [(list '= id [result]
                          (list 'scalar {:dtypes [(get-in values [result :dtype])]} operands
                                (soac/lambda-form (or parameters operands) [expression])))]
                   [result])]
    (program/map->ProgramEquation
     {:id id :site [:binding result] :operands operands :results [result]
      :algorithm algorithm :operations [] :effects #{} :provenance {}
      :attributes {:host-only true}})))

(defn- scalar-program [dtype]
  (let [values (zipmap '[x y a b] (repeat (av/tensor {:dtype dtype :shape []})))]
    (program/make
     {:dialect :opencl-parallel :values values :inputs '[x y] :outputs '[b]
      :equations [(host-equation values :first '[x] 'a '(clojure.core/inc x))
                  (host-equation values :second '[a y] 'b '(clojure.core/+ a y))]})))

(defn- make-call [dtype x y]
  (call/make (scalar-program dtype) {}
             {'x {:type dtype :value x} 'y {:type dtype :value y}} {}
             (partial scalar/evaluate-host-equation *ns*)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest selected-host-step-uses-the-same-scalar-and-conflict-contracts
  (let [p (scalar-program :long)
        calls (atom [])
        evaluator (fn [equation bindings]
                    (swap! calls conj (:id equation))
                    (scalar/evaluate-host-equation *ns* equation bindings))
        inputs {'x {:type :long :value 7} 'y {:type :long :value 10}}
        first-step (call/make-host-equation-step p :first {} inputs evaluator nil)
        second-step (call/make-host-equation-step
                     p :second {} (merge inputs (:results first-step)) evaluator nil)]
    (is (= [:first :second] @calls))
    (is (= {'a {:type :long :value 8}} (:results first-step)))
    (is (= {'b {:type :long :value 18}} (:results second-step)))
    (is (= :emitted-program-host-result-conflict
           (reason #(call/make-host-equation-step
                     p :first {} (assoc inputs 'a {:type :long :value 99}) evaluator nil))))
    (reset! calls [])
    (is (= :emitted-program-host-operand
           (reason #(call/make-host-equation-step p :second {} inputs evaluator nil))))
    (is (empty? @calls) "missing dependencies decline before invoking a host evaluator")))

(deftest host-steps-retain-source-ordered-input-and-result-bindings
  (let [original (make-call :long 7 10)
        proof (emitted/validate-with-physical-results! (:program original))]
    (is (= {'b {:type :long :value 18}} (:outputs original)))
    (is (= original (call/validate! original)))
    (doseq [[path expected]
            [[[:steps 0 :operands 'x :value] :emitted-program-host-operand-bindings]
             [[:steps 1 :operands 'a :value] :emitted-program-host-operand-bindings]
             [[:steps 1 :operands 'y :value] :emitted-program-host-operand-bindings]
             [[:steps 0 :results 'a :value] :emitted-program-host-result-bindings]
             [[:scalar-values 'a :value] :emitted-program-host-result-bindings]
             [[:steps 1 :results 'b :value] :emitted-program-host-result-bindings]]
            validate [call/validate!
                      #(call/preparation-plan % :host-bindings-test)
                      #(call/validate-with-retained-program! % proof)]]
      (is (= expected (reason #(validate (assoc-in original path 99)))) (pr-str path)))))

(deftest host-binding-comparison-preserves-ieee-values
  (doseq [x [Double/NaN Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY -0.0]]
    (let [original (make-call :double x 0.0)]
      (is (= original (call/validate! original)))
      (is (= (Double/doubleToRawLongBits (+ (inc x) 0.0))
             (Double/doubleToRawLongBits (get-in original [:outputs 'b :value])))))))

(deftest host-bindings-distinguish-ieee-payloads-and-retain-shape-only-scalars
  (let [values {'x (av/tensor {:dtype :double :shape []})
                'a (av/tensor {:dtype :double :shape []})
                'extent (av/tensor {:dtype :long :shape []})}
        p (program/make
           {:dialect :opencl-parallel :values values :inputs '[x] :outputs '[a]
            :equations [(host-equation values :identity '[x] 'a 'x)]})]
    (doseq [[source replacement]
            [[-0.0 0.0]
             [(Double/longBitsToDouble 0x7ff8000000000001)
              (Double/longBitsToDouble 0x7ff8000000000002)]]]
      (let [original (call/make p {} {'x {:type :double :value source}
                                      'extent {:type :long :value 17}} {}
                                (partial scalar/evaluate-host-equation *ns*))]
        (is (= original (call/validate! original)))
        (is (= {:type :long :value 17} (get (:scalar-values original) 'extent))
            "shape-specialization ports need not be numerical ABI inputs")
        (is (= :emitted-program-host-operand-bindings
               (reason #(call/validate! (assoc-in original [:steps 0 :operands 'x :value]
                                                  replacement)))))
        (is (= :emitted-program-host-result-bindings
               (reason #(call/validate! (assoc-in original [:steps 0 :results 'a :value]
                                                  replacement)))))))))

(deftest host-buffer-operands-retain-their-bound-storage-identity
  (let [values {'xs (av/tensor {:dtype :float :shape [3]})
                'n (av/tensor {:dtype :long :shape []})}
        p (program/make
           {:dialect :opencl-parallel :values values :inputs '[xs] :outputs '[n]
            :equations [(host-equation values :shape '[xs] 'n '(clojure.core/alength xs))]})
        original (call/make p {'xs :original-storage} {} {}
                            #(scalar/evaluate-host-equation *ns* %1
                                                           (assoc %2 :buffer-shapes {'xs [3]})))]
    (is (= {'n {:type :long :value 3}} (:outputs original)))
    (is (= (first (:steps original))
           (call/make-host-equation-step
            p :shape {'xs :original-storage} {}
            #(scalar/evaluate-host-equation *ns* %1 (assoc %2 :buffer-shapes {'xs [3]})) nil)))
    (doseq [path [[:steps 0 :operands 'xs] [:entry-buffers 'xs]]]
      (is (= :emitted-program-host-operand-bindings
             (reason #(call/validate! (assoc-in original path :different-storage))))))))

(deftest device-storage-cannot-masquerade-as-a-ready-host-scalar
  (let [compiled (equation-first/compile #'contractions/fixed-matmul
                                        {:target :ocl:0 :dtype :float})
        linked (equation-first/lower compiled [(float-array 15) (float-array 21)])
        original (get-in linked [:instances 0 :call])
        p (:program original)
        proof (emitted/validate-with-physical-results! p)
        producer (first (remove #(get-in % [:attributes :host-only]) (:equations p)))
        operation (first (:operations producer))
        ids (into (set (:results producer))
                  (vals (emitted-equation/physical-results operation)))]
    (is (= original (call/validate! original)))
    (is (seq ids))
    (doseq [id ids
            validate [call/validate!
                      #(call/preparation-plan % :device-scalar-shadow)
                      #(call/validate-with-retained-program! % proof)]]
      (is (= :emitted-program-device-result-scalar
             (reason #(validate (assoc-in original [:scalar-values id]
                                          {:type :float :value 3.0}))))
          (str "device result " id " cannot shadow its resident storage")))
    (let [result (first (:results producer))
          n 'host-length
          values (assoc (:values p) n (av/tensor {:dtype :long :shape []}))
          equation (host-equation values :late-shape [result] n
                                  '(clojure.core/alength device-result) '[device-result])
          extended (program/make (assoc p :values values :outputs [n]
                                       :equations (conj (:equations p) equation)))
          output {:type :long :value 35}
          fabricated (assoc original :program extended :outputs {n output}
                           :scalar-values (assoc (:scalar-values original) n output)
                           :steps (conj (:steps original)
                                        (call/->EvaluatedHostEquation
                                         equation {result (get (:buffers original) result)}
                                         {n output})))
          extended-proof (emitted/validate-with-physical-results! extended)]
      (let [calls (atom 0)]
        (is (= :emitted-program-host-device-dependency
               (reason #(call/make-host-equation-step
                         extended (:id equation) (:buffers original) (:scalar-values original)
                         (fn [& _] (swap! calls inc) {n output}) nil))))
        (is (zero? @calls) "selected construction retains the no-host-read-of-device-result boundary"))
      (doseq [validate [call/validate!
                        #(call/validate-with-retained-program! % extended-proof)]]
        (is (= :emitted-program-host-device-dependency
               (reason #(validate fabricated))))))))

(deftest entry-storage-shape-can-be-read-before-a-later-in-place-write
  ;; This is a hardware-free binding test. Use an explicit source target with checked
  ;; integer arithmetic, not whichever OpenCL device happens to be installed locally.
  (let [target :cuda:host-binding-order-test
        _ (hardware/register-target-device!
           target {:type :cuda :name "Synthetic host-binding order target"
                   :capabilities {:compute-capability [8 0] :warp-size 32
                                  :subgroup-sizes [32] :max-workgroup-size 1024
                                  :shared-local-memory 65536 :total-eus 108}})
        compiled (equation-first/compile #'attention/softmax-rows!
                                        {:target target :dtype :float})
        linked (equation-first/lower compiled [(float-array 6) 2 3])
        original (get-in linked [:instances 0 :call])
        p (:program original)
        proof (emitted/validate-with-physical-results! p)
        projected (fn [step]
                    (when (call/emitted-equation-call? step)
                      (let [operation (first (:operations (:equation step)))]
                        (into (set (:results (:equation step)))
                              (vals (emitted-equation/physical-results operation))))))
        witnesses (for [[i step] (map-indexed vector (:steps original))
                        :when (call/evaluated-host-equation? step)
                        id (keys (:operands step))
                        :when (and (not-any? #(contains? (projected %) id)
                                             (take i (:steps original)))
                                   (some #(contains? (projected %) id)
                                         (drop (inc i) (:steps original))))]
                    id)]
    (is (seq witnesses) "this control must read storage before its later physical update")
    (is (= original (call/validate! original)))
    (is (= original (call/validate-with-retained-program! original proof)))
    (is (map? (call/preparation-plan original :in-place-entry-shape)))))
