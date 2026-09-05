(ns raster.compiler.passes.parallel.typed-ordered-effect-map-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.abm.firms.phases :as firms-phases]
            [raster.par]))

(def ^:private extent (av/tensor {:dtype :long :shape []}))
(def ^:private vector-value (av/tensor {:dtype :float :shape '[n]}))
(def ^:private scalar-slot (av/tensor {:dtype :float :shape '[1]}))

(defn- effect-program
  []
  (let [out-result [:effect 0 0]
        total-result [:effect 0 1]
        reduction (dialect/reducing-scatter-conflict '+ :float)
        equation
        (list '= 0 [out-result total-result]
              (list 'effect-map
                    {:index 'i :extent 'n :dtypes [:float :float]
                     :iteration-order :independent}
                    '[x] [] '[out total]
                    (dialect/effect-lambda-form
                     '[element out-destination total-destination]
                     [(dialect/local-value 'shifted :float '(+ element 1.0))]
                     [(list 'effect 'out-destination :unique 'i '(> element 0.0) 'shifted)
                      (list 'effect 'total-destination reduction 0 true 'element)])))
        storage [{:destination 'out :access :write :host-return :effect}
                 {:destination 'total :access :read-write :host-return :effect}]
        equation-facts
        (assoc (dialect/default-equation-facts)
               :effects #{:memory/write}
               :aliases {out-result 'out total-result 'total}
               :attributes {:result-storage storage})]
    (dialect/make
     (dialect/default-program-facts
      {:values {'n extent 'x vector-value 'out vector-value 'total scalar-slot
                out-result vector-value total-result scalar-slot}
       :inputs '[n x]
       :equations {0 equation-facts}
       :effects #{:memory/write}})
     [equation] [])))

(defn- nested-operations
  [operations]
  (mapcat (fn [operation]
            (cons operation
                  (concat (nested-operations (or (:operations operation) []))
                          (nested-operations (or (:then-operations operation) []))
                          (nested-operations (or (:else-operations operation) [])))))
          operations))

(defn- normalize-fresh-thread-index
  [operation]
  (assoc-in operation [:space :flat-idx] 'thread-index))

(def ^:private mixed-effect-source
  '(let* [effect
          (raster.par/map-void!
           i n
           (let* [^float previous
                  (clojure.core/aget out (clojure.core/aget slots i))]
             (do
               (if (> (clojure.core/aget x i) 0.0)
                 (clojure.core/aset
                  out (raster.par/unique-index (clojure.core/aget slots i))
                  (float (+ previous (clojure.core/aget x i)))))
               (raster.par/atomic-add! total 0 (float (clojure.core/aget x i))))))]
         effect))

(deftest ordered-effects-lower-to-portable-kernelbody-control-and-atomics
  (let [program (effect-program)
        operation (first (soac-lower/lower-typed-effect-map
                          program :ze:0 :dtype :float))
        scheduled (:form (segop-lower/segop-lower-pass
                          (route/program-envelope program)
                          {:target-device :ze:0 :dtype :float}))
        scheduled-operation (first (get-in scheduled [:equations 0 :operations]))]
    (is (= :ordered-effects (:write-conflict operation)))
    (is (= (normalize-fresh-thread-index operation)
           (normalize-fresh-thread-index scheduled-operation))
        "whole-program scheduling dispatches the same semantic effect-map lowering")
    (is (= #{'out 'total} (:outputs operation)))
    (is (= {:locals [{:id 'shifted :dtype :float :init '(+ (clojure.core/aget x i) 1.0)}]
            :effects
            [{:destination 'out :dtype :float :conflict :unique :destination-index 'i
              :predicate '(> (clojure.core/aget x i) 0.0) :value 'shifted}
             {:destination 'total
              :dtype :float
              :conflict (dialect/reducing-scatter-conflict '+ :float)
              :destination-index 0 :predicate true
              :value '(clojure.core/aget x i)}]
            :iteration-order :independent}
           (:scalar-region operation)))
    (doseq [[target atomic]
            [[:opencl-portable "atomic_add_float"]
             [:cuda "atomicAdd"]
             [:hip "atomicAdd"]]]
      (testing (name target)
        (let [artifact (segop-opencl/generate-scheduled-segmap-kernel
                        operation :dtype :float :target-dialect target
                        :array-types {'x :float 'out :float 'total :float}
                        :scalar-types {'n :long})
              kernel-body (get-in artifact [:attributes :kernel-body])
              operations (nested-operations (:operations kernel-body))
              pointer-slots (filterv #(not= :scalar (:kind %)) (:abi artifact))]
          (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
          (is (= ['x 'out 'total] (mapv :name pointer-slots)))
          (is (= [:input :output :inout] (mapv :kind pointer-slots)))
          (is (some #(= "IfRegion" (some-> % class .getSimpleName)) operations))
          (is (some #(= "ScalarStore" (some-> % class .getSimpleName)) operations))
          (is (some #(= "AtomicRMW" (some-> % class .getSimpleName)) operations))
          (is (= {:mode :reassociated :policy :certified-reducing-scatter
                  :accumulators [{:value 'total :dtype :float
                                  :rounding :implementation-defined
                                  :policy :proof-carrying-destination}]}
                 (get-in artifact [:attributes :numerics]))
              "a proof-carrying per-destination reduction is never certified as exact")
          (is (str/includes? (:source artifact) atomic)))))))

(deftest analyzed-source-selects-the-same-ordered-effect-schedule
  (let [result (route/attempt mixed-effect-source :float
                              {'x :float 'slots :int 'out :float 'total :float})
        parallel-program (:program result)
        typed-operation (-> parallel-program :equations first :algorithm
                            dialect/equations first dialect/operation-kind)
        scheduled (:form (segop-lower/segop-lower-pass
                          parallel-program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                  operation :dtype :float :target-dialect :opencl-portable
                  :array-types {'x :float 'slots :int 'out :float 'total :float}
                  :scalar-types {'n :long})
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x slots out total n] (:form jvm)))
        x (float-array [-1.0 2.0 3.0])
        out (float-array [10.0 20.0 30.0])
        total (float-array 1)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= 'effect-map typed-operation))
    (is (= :ordered-effects (:write-conflict operation)))
    (is (= ['slots 'x 'out 'total]
           (mapv :name (filterv #(not= :scalar (:kind %)) (:abi artifact)))))
    (is (= [:input :input :inout :inout]
           (mapv :kind (filterv #(not= :scalar (:kind %)) (:abi artifact)))))
    (is (str/includes? (:source artifact) "atomic_add_float"))
    (is (nil? (execute x (int-array [2 0 1]) out total 3)))
    (is (= [12.0 23.0 30.0] (mapv double out)))
    (is (= [4.0] (mapv double total)))
    (is (= 1 (get-in jvm [:stats :segop-reused])))
    (is (zero? (get-in jvm [:stats :fallback])))))

(deftest potentially-conflicting-effects-use-one-ordered-device-loop
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (do
                   (clojure.core/aset out (clojure.core/aget slots i)
                                      (float (clojure.core/aget x i)))
                   (raster.par/atomic-add! total 0 (float 1.0))))]
               effect)
        result (route/attempt source :float
                              {'x :float 'slots :int 'out :float 'total :float})
        program (:program result)
        equation (-> program :equations first :algorithm dialect/equations first)
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        artifacts
        (into {}
              (map (fn [target]
                     [target
                      (segop-opencl/generate-scheduled-segmap-kernel
                       operation :dtype :float :target-dialect target
                       :array-types {'x :float 'slots :int 'out :float 'total :float}
                       :scalar-types {'n :long})]))
              [:opencl-portable :cuda :hip])
        artifact (get artifacts :opencl-portable)
        kernel-body (get-in artifact [:attributes :kernel-body])
        operations (nested-operations (:operations kernel-body))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x slots out total n] (:form jvm)))
        out (float-array [10.0 20.0])
        total (float-array 1)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= :sequential (get-in (dialect/operation-parts equation)
                                [:attributes :iteration-order])))
    (is (= [:ordered :reduce]
           (mapv (fn [effect]
                   (let [conflict (:conflict (dialect/effect-parts effect))]
                     (if (keyword? conflict) conflict (:kind conflict))))
                 (:body-results
                  (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))))))
    (is (= :sequential (:effect-iteration-order operation)))
    (is (= [1] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [1] (get-in kernel-body [:launch :group-count])))
    (is (= :one-work-item-ordered-loop (get-in kernel-body [:schedule :strategy])))
    (is (some #(= "ForLoop" (some-> % class .getSimpleName)) operations))
    (doseq [[target emitted] artifacts]
      (testing (name target)
        (is (= :kernel-body (get-in emitted [:attributes :emission-route])))
        (is (= :one-work-item-ordered-loop
               (get-in emitted [:attributes :kernel-body :schedule :strategy])))
        (is (str/includes? (:source emitted) "for ("))))
    (is (nil? (execute (float-array [1.0 2.0 3.0]) (int-array [1 1 1]) out total 3)))
    (is (= [10.0 3.0] (mapv double out)) "the last source-order overwrite wins")
    (is (= [3.0] (mapv double total)))
    (is (= 1 (get-in jvm [:stats :segop-reused])))
    (is (zero? (get-in jvm [:stats :fallback])))))

(deftest firms-decision-queue-uses-the-certified-ordered-kernel
  (let [descriptor (pipeline/compile-gpu-program
                    #'firms-phases/execute-stay-switch-par!
                    :ze:0 :dtype :float :on-non-resident :nil)
        step (first (:steps descriptor))
        kernel-body (get-in step [:artifact :attributes :kernel-body])]
    (is (some? descriptor))
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))
    (is (= :one-work-item-ordered-loop (get-in kernel-body [:schedule :strategy])))
    (is (= :ordered (get-in kernel-body [:schedule :association])))
    (is (= [1] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [1] (get-in kernel-body [:launch :group-count])))
    (is (some #(= "ForLoop" (some-> % class .getSimpleName))
              (:operations kernel-body)))
    (is (= :sequential (get-in kernel-body [:attributes :effect-iteration-order])))))

(def ^:private row-loop-source
  ;; rms-norm-shaped: per row `r`, an ordered fold over the row whose exit divides the carry, then
  ;; a counted loop storing every element of the row at `r*feat + i`.
  '(let* [effect
          (raster.par/map-void!
           r rows
           (let* [^long offset (clojure.core/* r feat)
                  ^double ms (loop* [i 0 s 0.0]
                                (if (clojure.core/< i feat)
                                  (let* [^double v (double (clojure.core/aget
                                                            x (clojure.core/+ offset i)))]
                                    (recur (clojure.core/inc i)
                                           (clojure.core/+ s (clojure.core/* v v))))
                                  (clojure.core// s (double feat))))]
             (loop* [i 0]
               (if (clojure.core/< i feat)
                 (let* [^float v (float (clojure.core/aget x (clojure.core/+ offset i)))]
                   (clojure.core/aset out (clojure.core/+ offset i)
                                      (float (clojure.core/* v ms)))
                   (recur (clojure.core/inc i)))))))]
         effect))

(deftest row-store-loops-are-independent-effect-loops
  (let [result (route/attempt row-loop-source :float {'x :float 'out :float}
                              {:scalar-types {'rows :long 'feat :long}})
        program (:program result)
        algorithm (-> program :equations first :algorithm)
        equation (first (dialect/equations algorithm))
        {:keys [attributes lambda]} (dialect/operation-parts equation)
        {:keys [body-results]} (dialect/lambda-parts lambda)
        parts (mapv dialect/effect-parts body-results)
        leaves (dialect/effect-part-leaves parts)
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                  ;; The scheduled device is :ze:0. `r * feat` is ordinary long arithmetic,
                  ;; so this concrete artifact uses Intel OpenCL's checked-trap contract.
                  operation :dtype :float :target-dialect :opencl-intel
                  :array-types {'x :float 'out :float}
                  :scalar-types {'rows :long 'feat :long})
        portable-reason (try
                          (segop-opencl/generate-scheduled-segmap-kernel
                           operation :dtype :float :target-dialect :opencl-portable
                           :array-types {'x :float 'out :float}
                           :scalar-types {'rows :long 'feat :long})
                          nil
                          (catch clojure.lang.ExceptionInfo exception
                            (:reason (ex-data exception))))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x out rows feat] (:form jvm)))
        x (float-array [1.0 2.0 3.0 4.0])
        out (float-array 4)]
    (testing "the store loop is one effect-loop item whose store is injective across rows"
      (is (= :typed-soac (get-in result [:stats :route])))
      (is (= 'effect-map (dialect/operation-kind equation)))
      (is (= :independent (:iteration-order attributes)))
      (is (= [true] (mapv :loop parts)))
      (is (= [:unique] (mapv :conflict leaves)))
      (is (= (dialect/validate! algorithm) algorithm)))
    (testing "the scheduled SegMap carries the loop as structured data, not source"
      (is (nil? (:lambda operation)))
      (is (some :loop (get-in operation [:scalar-region :effects]))))
    (testing "the device kernel is a verified KernelBody with a nested loop"
      (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
      (is (nil? (get-in artifact [:attributes :kernel-body-decline])))
      (is (= :kernel-body-c-trap-unsupported portable-reason)
          "portable OpenCL does not pretend it can preserve checked long arithmetic")
      (is (str/includes? (:source artifact) "for (")))
    (testing "the JVM schedule executes the loop"
      (is (nil? (execute x out 2 2)))
      (is (= [2.5 5.0 37.5 50.0] (mapv double out)))
      (is (zero? (get-in jvm [:stats :fallback]))))))

(deftest effect-loops-must-be-closed-over-the-loop-index
  (let [program (:program (route/attempt row-loop-source :float {'x :float 'out :float}
                                         {:scalar-types {'rows :long 'feat :long}}))
        algorithm (-> program :equations first :algorithm)
        rebound (clojure.walk/postwalk
                 (fn [form]
                   (if (and (seq? form) (= 'effect-loop (first form)))
                     (let [[_ attributes extent [_ _ region]] form]
                       (list 'effect-loop attributes extent (list 'lambda '[other] region)))
                     form))
                 algorithm)]
    (is (= :typed-soac-effect-loop
           (try (dialect/validate! rebound) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest explicit-store-emission-refuses-typed-regions
  (let [program (:program (route/attempt row-loop-source :float {'x :float 'out :float}
                                         {:scalar-types {'rows :long 'feat :long}}))
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:target-device :ze:0 :dtype :float}))
        operation (first (get-in scheduled [:equations 0 :operations]))]
    (is (= :explicit-segmap-requires-source-lambda
           (try (segop-opencl/generate-explicit-segmap-kernel
                 operation :dtype :float :array-types {'x :float 'out :float}
                 :scalar-types {'rows :long 'feat :long})
                nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
