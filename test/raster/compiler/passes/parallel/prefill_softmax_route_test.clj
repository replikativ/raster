(ns raster.compiler.passes.parallel.prefill-softmax-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-ownership :as ownership]
            [raster.dl.attention :as attention]
            [raster.gpu.device-probe :as probe]))

(def ^:private compiled
  (delay (pipeline/show-pipeline #'attention/attn-prefill-softmax!
                                 :target-device :ocl:0 :dtype :float)))

(defn- prefill-program []
  (let [options {:dtype :float :array-types {'sc :float}
                 :scalar-types {'nrows :long 'n-q :long}}]
    (frontend/form->program
     (frontend/normalize-source (:loop-lifted @compiled) options) options)))

(deftest prefill-maximum-is-a-canonical-ordered-fold-before-admission
  (let [program (prefill-program)
        nodes (tree-seq coll? seq (dialect/equations program))
        folds (filter dialect/scalar-fold-form? nodes)]
    (is (= program (dialect/validate! program)))
    (is (= 1 (count folds)))
    (is (= :ordered
           (get-in (dialect/scalar-fold-parts (first folds)) [:attributes :association])))
    (is (not-any? #(and (seq? %) (contains? #{'loop 'loop*} (first %))) nodes))))

(deftest prefill-softmax-uses-proved-parallel-kernel-body-launch
  (let [p @compiled
        kernel (first (:kernels p))]
    (is (= 1 (count (:kernels p))))
    (is (= :typed-soac (get-in p [:soac-fused-stats :route])))
    (is (= 1 (get-in p [:soac-fused-stats :effect-row-ownership-proofs])))
    (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
    (is (= '[sc nrows _n_bound] (mapv :name (:abi kernel))))
    (is (= [:float :long :long] (mapv :dtype (:abi kernel))))
    (is (= [256] (get-in kernel [:launch :workgroup-size])))
    (is (= [{:value 'rstr_extent_0 :divisor 256}]
           (mapv #(into {} %) (get-in kernel [:launch :group-count])))
        "proved source coverage must retain the previously parallel row launch")))

(deftest ownership-proof-declines-cross-row-and-guarded-address-domains
  (let [program (prefill-program)
        rewrite (fn [f]
                  (dialect/make (dialect/facts program)
                                (mapv #(walk/postwalk f %) (dialect/equations program))
                                (dialect/outputs program)))
        shifted (rewrite
                 #(if (= '(clojure.core/+ (clojure.core/long rstr_local_0)
                                           (clojure.core/long rstr_loop_index_1)) %)
                    '(clojure.core/+ (clojure.core/long rstr_local_0)
                                     (clojure.core/long rstr_loop_index_1) 1)
                    %))
        guarded (rewrite
                 #(if (and (seq? %) (= 'effect (first %)))
                    (apply list (assoc (vec %) 4 false)) %))
        carry-parameter
        (some (fn [form]
                (when (dialect/effect-loop-form? form)
                  (get-in (dialect/effect-parts form) [:carry :parameter])))
              (tree-seq coll? seq (dialect/equations program)))
        carry-dependent
        (rewrite
         #(if (= '(clojure.core/+ (clojure.core/long rstr_local_0)
                                  (clojure.core/long rstr_loop_index_0)) %)
            (list 'clojure.core/+ % carry-parameter) %))
        mismatched-domain
        (rewrite
         #(if (and (dialect/effect-loop-form? %)
                   (nil? (:carry (dialect/effect-parts %))))
            (apply list (assoc (vec %) 2 (list 'clojure.core/+ (nth % 2) 1))) %))
        opaque-use
        (rewrite
         #(if (and (seq? %) (= 'effect (first %)))
            (apply list (assoc (vec %) 5 (symbol "%destination0"))) %))]
    (doseq [[label candidate] [[:neighboring-row shifted]
                               [:guarded guarded]
                               [:carry-dependent carry-dependent]
                               [:mismatched-domain mismatched-domain]
                               [:opaque-destination-use opaque-use]]]
      (testing (name label)
        (let [[result stats] (ownership/prove candidate)
              equation (second (dialect/equations result))]
          (is (zero? (:effect-row-ownership-proofs stats)))
          (is (= :sequential
                 (get-in (dialect/operation-parts equation)
                         [:attributes :iteration-order]))))))))

(deftest prefill-softmax-keeps-row-values-and-output-tails-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "parallel prefill softmax")
    (let [runtime (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve runtime 'register-kernel!)
          buffer-of-array (ns-resolve runtime 'buffer-of-array)
          bind-call (ns-resolve runtime 'bind-kernel-call)
          launch! (ns-resolve runtime 'launch-registered-bound!)
          read! (ns-resolve runtime 'buffer->array)
          free! (ns-resolve runtime 'free-buffer!)
          kernel (first (:kernels @compiled))]
      (register! (:kernel-name kernel) kernel)
      ;; Direct launches require a nonempty grid; frontend execution tests separately cover empty
      ;; loops. Width 129 crosses a workgroup boundary (258 rows).
      (doseq [width [1 3 8 129]]
        (let [heads 2 rows (* width heads) n (* rows width)
              input (vec (mapcat (fn [row]
                                  (map #(float (+ (* (- row 3) 1000) (* 0.5 %))) (range width)))
                                (range rows)))
              expected (vec (mapcat (fn [row]
                                     (let [m (apply max row)
                                           values (map #(Math/exp (- (double %) m)) row)
                                           total (reduce + values)]
                                       (map #(/ % total) values)))
                                   (if (pos? width) (partition width input) [])))
              scores (buffer-of-array (float-array (concat input [-77 -77])) :float)]
          (try
            (launch! (bind-call (call/make kernel [scores {:type :long :value width}
                                                   {:type :long :value rows}])))
            (let [actual (vec (read! scores))]
              (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) 1.0e-6)
                                     expected (take n actual))))
              (is (= [-77.0 -77.0] (subvec actual n))))
            (finally (free! scores))))))))
