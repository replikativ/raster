(ns raster.compiler.passes.parallel.structured-control-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.parallel-program-opencl :as program-opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.dialects :as dialects]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.structured-control-frontend :as frontend]
            [raster.compiler.passes.parallel.structured-control-route :as route]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.pipeline :as pipeline]))

(defn- loop-decomposition
  []
  (let [extent (av/tensor {:dtype :int :shape []})
        index (av/tensor {:dtype :long :shape []})
        tensor (av/tensor {:dtype :float :shape '[n]})
        inner (av/tensor {:dtype :float :shape '[n-in]})
        equation (list '= 'advance '[u-next]
                       (list 'map {:index 'i :extent 'n-in}
                             '[u-in] '[]
                             (soac/lambda-form '[value] '[(+ value 1.0)])))
        body (soac/make
              (soac/default-program-facts
               {:values {'iteration index 'n-in extent 'u-in inner 'u-next inner}
                :inputs '[n-in u-in]
                :equations {'advance (soac/default-equation-facts)}})
              [equation] '[u-next])
        loop (control/make
              {:id 'time-loop :effects #{} :provenance {:source :test}
               :attributes {:association :sequential}}
              '[iteration steps]
              [{:outer 'n :parameter 'n-in}]
              [{:initial 'u :parameter 'u-in :result 'u-next :output 'u-final}]
              body
              {'steps index 'n extent 'u tensor 'u-final tensor})]
    {:loop loop :loop-binding 'time-loop :source '(let* [] nil)}))

(deftest structured-control-uses-the-common-program-equation-spine
  (let [typed (route/program-envelope (loop-decomposition))
        scheduled (route/schedule-program typed {:target-device :cpu:0 :dtype :float})
        typed-equation (first (:equations typed))
        scheduled-equation (first (:equations scheduled))]
    (is (program/parallel-program? typed))
    (is (= :typed-parallel (:dialect typed)))
    (is (= '[steps n u] (:inputs typed)))
    (is (= '[u-final] (:outputs typed)))
    (is (= (:algorithm typed-equation) (:algorithm scheduled-equation)))
    (is (= :scheduled-parallel (:dialect scheduled)))
    (is (schedule/scheduled-loop? (first (:operations scheduled-equation))))
    (testing "the scheduled equation retains the certified one-iteration graph"
      (is (= :kernel-graph
             (get-in scheduled-equation [:attributes :graph-dialect])))
      (is (= '[u-in]
             (mapv :id (get-in scheduled-equation [:operations 0 :graph :inputs])))))))

(deftest repeated-outer-operands-do-not-break-program-ssa
  (let [{:keys [loop] :as decomposition} (loop-decomposition)
        loop (control/make
              (control/facts loop)
              (assoc (control/loop-index loop) 1 'n)
              (control/invariants loop)
              (control/carried loop)
              (control/body loop)
              (control/outer-values loop))
        typed (route/program-envelope (assoc decomposition :loop loop))]
    (is (= '[n u] (:inputs typed)))
    (is (= '[n u] (:operands (first (:equations typed)))))))

(defn- mixed-source
  []
  '(let* [n (int (clojure.core/alength u0))
          u (clojure.core/aclone u0)
          scratch (clojure.core/double-array n)
          time-loop
          (dotimes [step steps]
            (raster.par/map! scratch i n double
                             (+ (clojure.core/aget u i) 1.0))
            (let* [next (raster.par/pmap j n double
                                         (+ (clojure.core/aget u j)
                                            (clojure.core/aget scratch j)
                                            (* 0.0 step)))]
                  (java.lang.System/arraycopy next 0 u 0 n)))
          after (raster.par/pmap k n double
                                 (* 2.0 (clojure.core/aget u k)))]
         after))

(defn- source-with-suffix
  [suffix-bindings result]
  (let [[_ bindings] (mixed-source)]
    (list 'let* (vec (concat (take 8 bindings) suffix-bindings)) result)))

(defn- reason-of
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest loop-output-feeds-an-ordinary-typed-suffix
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        decomposition (frontend/form->structured-loop (mixed-source) options)
        typed (route/program-envelope decomposition options)
        scheduled (route/schedule-program typed {:target-device :cpu:0 :dtype :double})
        [loop-equation suffix-equation] (:equations typed)
        [scheduled-loop scheduled-suffix] (:equations scheduled)
        loop-output (first (:results loop-equation))]
    (is (= 2 (count (:equations typed))))
    (is (= [loop-output 'n] (:operands suffix-equation)))
    (is (= '[after] (:outputs typed)))
    (is (= '[steps n u] (:inputs typed)))
    (is (= true (get-in typed [:attributes :mixed-algorithms])))
    (is (schedule/scheduled-loop? (first (:operations scheduled-loop))))
    (is (segop/segop-node? (first (:operations scheduled-suffix))))
    (is (not-any? #{'u} (:operands suffix-equation)))))

(deftest mixed-routing-never-drops-an-unsupported-suffix
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        source (source-with-suffix
                '[unsupported (java.lang.System/gc)]
                'unsupported)
        decomposition (frontend/form->structured-loop source options)]
    (is (= :structured-control-unsupported-suffix
           (reason-of #(route/program-envelope decomposition options))))))

(deftest suffix-shadowing-is-rejected-instead-of-breaking-program-ssa
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        source (source-with-suffix
                '[u (raster.par/pmap k n double
                                     (* 2.0 (clojure.core/aget u k)))]
                'u)
        decomposition (frontend/form->structured-loop source options)]
    (is (= :parallel-program-result-redefinition
           (reason-of #(route/program-envelope decomposition options))))))

(deftest algorithm-kind-and-operation-kind-must-remain-paired
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        decomposition (frontend/form->structured-loop (mixed-source) options)
        typed (route/program-envelope decomposition options)
        suffix-operation (get-in typed [:equations 1 :operations 0])
        malformed (assoc-in typed [:equations 0 :operations] [suffix-operation])]
    (is (= :parallel-program-algorithm
           (reason-of #(route/schedule-program malformed
                                               {:target-device :cpu:0 :dtype :double}))))))

(deftest public-scheduler-selects-the-structured-semantic-program
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        attempted (route/attempt (mixed-source) options)
        scheduled (pipeline/schedule-parallel-form (mixed-source) options)]
    (is (= :typed-structured-control (get-in attempted [:stats :route])))
    (is (dialects/valid-source-or-typed-soac? (:program attempted)))
    (is (= :scheduled-parallel (:dialect (:form scheduled))))
    (is (dialects/valid-scheduled-program? (:form scheduled)))
    (is (= :typed-structured-control (get-in scheduled [:stats :source-dialect])))
    (is (= 1 (get-in scheduled [:stats :structured-loops-scheduled])))
    (is (= 2 (count (:equations (:form scheduled)))))))

(deftest structured-attempt-exposes-a-coverage-decline
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        source (source-with-suffix '[unsupported (java.lang.System/gc)] 'unsupported)
        attempted (route/attempt source options)]
    (is (nil? (:program attempted)))
    (is (= :structured-control-unsupported-suffix
           (get-in attempted [:declined :reason])))))

(deftest cpu-public-scheduling-does-not-enter-the-gpu-structured-route
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :cpu:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        scheduled (pipeline/schedule-parallel-form (mixed-source) options)]
    (is (not= :scheduled-parallel (:dialect (:form scheduled))))
    (is (not= :typed-structured-control (get-in scheduled [:stats :source-dialect])))))

(deftest ordinary-suffix-equations-use-the-same-kernel-graph-builder
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        typed (:program (route/attempt (mixed-source) options))
        algorithm (get-in typed [:equations 1 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass
                          (typed-route/program-envelope algorithm) options))
        kernel-graph (equation-graph/make
                      algorithm scheduled
                      {:provenance {:source-dialect :mixed-suffix}})]
    (is (= kernel-graph (graph/validate! kernel-graph)))
    (is (= (vec (mapcat :operations (:equations scheduled)))
           (mapv :operation (:nodes kernel-graph))))
    (is (= :mixed-suffix (get-in kernel-graph [:provenance :source-dialect])))
    (is (nil? (:source scheduled)))))

(deftest equation-first-opencl-emits-loop-and-suffix-without-reading-source
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        scheduled (:form (pipeline/schedule-parallel-form (mixed-source) options))
        opaque-source (assoc scheduled :source ::must-not-be-inspected)
        {:keys [program kernels stats]} (program-opencl/emit-program opaque-source options)
        [loop-equation suffix-equation] (:equations program)]
    (is (= :opencl-parallel (:dialect program)))
    (is (= ::must-not-be-inspected (:source program)))
    (is (emitted-loop/emitted-loop? (first (:operations loop-equation))))
    (is (emitted-equation/emitted-equation? (first (:operations suffix-equation))))
    (is (every? artifact/kernel-artifact? kernels))
    (is (= {:structured-loops-emitted 1
            :typed-equations-emitted 1
            :host-scalar-equations 0}
           stats))
    (is (= program (program-opencl/validate-program! program)))
    (testing "emitted algorithm kinds cannot be exchanged between equations"
      (let [suffix-operation (get-in suffix-equation [:operations 0])]
        (is (= :parallel-program-algorithm
               (reason-of #(program-opencl/validate-program!
                            (assoc-in program [:equations 0 :operations]
                                      [suffix-operation])))))))))
