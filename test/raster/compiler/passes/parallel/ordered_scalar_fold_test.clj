(ns raster.compiler.passes.parallel.ordered-scalar-fold-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.segred-body :as fold]
            [raster.compiler.passes.parallel.ordered-scalar-fold-fixture :refer [operation]]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.device-probe :as ocl]
            [raster.gpu.runtime-backend :as backend]))

(deftest ordered-fold-retains-actual-seed-step-and-terminal-store
  (doseq [dtype [:float :double] bound [0 1 3 'n]]
    (let [seed (if (= :float dtype) (float 2.0) 2.0)
          op (operation dtype seed bound '*)
          certificate (fold/schedule op {:array-types {'a dtype} :scalar-types {'n :int}})
          body (:body certificate)
          [loop store] (:operations body)
          [load combine yield] (:operations loop)
          carry (get-in loop [:iter-args 0 :binding :id])]
      (is (= [] (:allocations body)))
      (is (= [1] (get-in body [:launch :workgroup-size])))
      (is (= [1] (get-in body [:launch :group-count])))
      (is (= 0 (get-in body [:launch :shared-memory-bytes])))
      (is (= seed (get-in loop [:iter-args 0 :initial :value])))
      (is (= dtype (get-in loop [:iter-args 0 :binding :type])))
      (is (= :ordered (get-in loop [:attributes :association])))
      (is (true? (get-in loop [:attributes :source-order])))
      (is (= 'a (:buffer load)))
      (is (= :* (get-in combine [:expression :op])))
      (is (= [carry (get-in load [:result :id])] (get-in combine [:expression :arguments])))
      (is (= [(get-in combine [:result :id])] (:values yield)))
      (is (= (get-in loop [:results 0 :id]) (:value store)))
      (is (= :source-ordered-seeded-fold (get-in certificate [:numerics :policy])))
      (is (= :exact (get-in certificate [:numerics :mode])))
      (is (numerics/rounding-policy? (get-in certificate [:numerics :rounding])))
      (is (nil? (get-in certificate [:legality :certified-monoid])))
      (is (= ['a 'result bound] (:arguments certificate))))))

(deftest ordered-fold-locals-retain-declared-types-instead-of-reparsing-flat-bindings
  (let [op (operation :double 2.0 1 '*)
        local (with-meta 'element {:raster.type/tag 'double})
        step (with-meta '(* acc element) {:raster.type/tag 'double})
        typed (-> op
                  (assoc-in [:reduction :step :bindings] [local '(clojure.core/aget a i)])
                  (assoc-in [:reduction :step :results] [step]))
        certificate (fold/schedule typed {:array-types {'a :double}})
        operations (get-in certificate [:body :operations 0 :operations])]
    (is (= 'a (:buffer (first operations))))
    (is (= :double (get-in operations [0 :result :type])))
    (is (= :* (get-in operations [1 :expression :op])))
    (is (= 3 (count operations)) "one load, one multiply, one yield")
    (is (= :ordered-fold-local-dtype
           (try (fold/schedule (assoc-in typed [:reduction :step :bindings]
                                        ['element '(clojure.core/aget a i)]) {}) nil
                (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e))))))))

(deftest ordered-fold-refuses-a-changed-physical-schedule
  (let [op (operation :double 2.0 1 '*)]
    (doseq [changed [(assoc op :grid (segop/->KernelGrid 2 1 0))
                     (assoc op :grid (segop/->KernelGrid 1 2 0))
                     (assoc op :grid (segop/->KernelGrid 1 1 8))
                     (assoc op :phase :block-local)
                     (assoc-in op [:schedule :numerical-mode :reassociation] :permitted)]]
      (is (= :segred-kernel-body-declined
             (try (fold/schedule changed {}) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest ordered-fold-reconstructs-body-storage-and-numerics-from-its-node
  (let [op (operation :double 2.0 1 '*)
        node (graph/->ScheduledKernel :fold op [(graph/->ValueUse 'a :read)
                                               (graph/->ValueUse 'result :write)] #{} [])
        g (graph/make {:inputs [(graph/buffer 'a :double 1 :device :input)]
                       :outputs [(graph/buffer 'result :double 1 :device :output)]
                       :scalars [] :nodes [node]})
        certificate (fold/schedule op {:array-types {'a :double}})]
    (is (= certificate (fold/validate-against-node! certificate node g)))
    (doseq [[changed reason]
            [[(assoc-in certificate [:body :operations 0 :iter-args 0 :initial :value] 1.0)
              :segred-kernel-body-declined]
             [(update-in certificate [:body :operations 0 :operations 1 :expression :arguments]
                         #(vec (reverse %))) :segred-kernel-body-declined]
             [(assoc-in certificate [:numerics :policy] :certified-workgroup-tree)
              :segred-kernel-body-declined]
             [(assoc-in certificate [:numerics :mode] :reassociated)
              :segred-kernel-body-declined]]]
      (is (= reason
             (try (fold/validate-against-node! changed node g) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest ordered-fold-requires-proved-load-footprints-even-with-large-backing-storage
  (let [base (operation :double 2.0 3 '*)]
    (doseq [[coordinate inputs] [['(+ i 1) #{'a}]
                                 ['(* 2 i) #{'a}]
                                 ['(int (clojure.core/aget indices i)) #{'a 'indices}]]]
      (let [step (with-meta (list '* 'acc (list 'clojure.core/aget 'a coordinate))
                   {:raster.type/tag 'double})
            op (-> base (assoc :inputs inputs) (assoc-in [:reduction :step :results] [step]))]
        (is (= :ordered-fold-read-footprint
               (try (fold/schedule op {:array-shapes {'a [100000] 'indices [100000]}
                                       :array-types {'a :double 'indices :double}}) nil
                    (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e))))))))))

(deftest ordered-fold-validates-capacity-separately-from-its-proved-minimum
  (let [op (operation :double 2.0 3 '*)]
    (doseq [capacity [5 2]]
      (let [node (graph/->ScheduledKernel :fold op [(graph/->ValueUse 'a :read)
                                                   (graph/->ValueUse 'result :write)] #{} [])
            g (graph/make {:inputs [(graph/buffer 'a :double capacity :device :input)]
                           :outputs [(graph/buffer 'result :double 1 :device :output)]
                           :scalars [] :nodes [node]})
            certificate (fold/schedule op {:array-types {'a :double} :array-shapes {'a [capacity]}})]
        (is (= '_n_bound (get-in certificate [:body :parameters 0 :minimum-elements])))
        (is (= [capacity] (get-in certificate [:body :parameters 0 :shape])))
        (if (= 5 capacity)
          (do
            (is (= certificate (fold/validate-against-node! certificate node g)))
            (is (thrown? clojure.lang.ExceptionInfo
                         (fold/validate-against-node!
                          (assoc-in certificate [:body :parameters 0 :minimum-elements] 0) node g))))
          (is (= :storage-extent
                 (try (fold/validate-against-node! certificate node g) nil
                      (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e)))))))))))

(deftest ordered-fold-symbolic-backing-capacity-requires-a-checked-graph-guard
  (let [op (assoc (operation :double 2.0 3 '*) :scalars #{'capacity})
        node (graph/->ScheduledKernel :fold op [(graph/->ValueUse 'a :read)
                                               (graph/->ValueUse 'result :write)] #{'capacity} [])
        certificate (fold/schedule op {:array-types {'a :double} :array-shapes {'a ['capacity]}
                                       :scalar-types {'capacity :long}})]
    (doseq [guard? [false true]]
      (let [g (graph/make
               {:inputs [(graph/buffer 'a :double 'capacity :device :input)]
                :outputs [(graph/buffer 'result :double 1 :device :output)]
                :scalars [(graph/scalar 'capacity :long)] :nodes [node]
                :preconditions (if guard? [{:expression 'capacity :op :>= :value 3}] [])})]
        (if guard?
          (is (= certificate (fold/validate-against-node! certificate node g)))
          (is (= :storage-extent
                 (try (fold/validate-against-node! certificate node g) nil
                      (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e)))))))))))

(deftest ordered-fold-closes-symbolic-capacity-and-explicit-scalar-captures
  (let [step (with-meta '(+ acc (* scale (clojure.core/aget a i)))
               {:raster.type/tag 'double})
        ;; Nested arithmetic retains its own result fact, independent of the fold dtype.
        step (with-meta (list '+ 'acc (with-meta (nth step 2) {:raster.type/tag 'double}))
               (meta step))
        op (-> (operation :double 2.0 'n '+)
               (assoc :scalars #{'scale})
               (assoc-in [:reduction :step :results] [step]))
        node (graph/->ScheduledKernel :fold op [(graph/->ValueUse 'a :read)
                                               (graph/->ValueUse 'result :write)] #{'n 'scale} [])
        g (graph/make {:inputs [(graph/buffer 'a :double 'n :device :input)]
                       :outputs [(graph/buffer 'result :double 1 :device :output)]
                       :scalars [(graph/scalar 'n :long) (graph/scalar 'scale :double)] :nodes [node]})
        certificate (fold/schedule op {:array-types {'a :double}
                                       :scalar-types {'n :long 'scale :double}})]
    (is (= '[a result scale n] (:arguments certificate)))
    (is (= :double (get-in certificate [:body :parameters 2 :dtype])))
    (is (= :long (get-in certificate [:scalar-bindings 1 :dtype])))
    (is (= :checked-range (get-in certificate [:scalar-bindings 1 :conversion])))
    (is (= certificate (fold/validate-against-node! certificate node g)))))

(deftest ordered-fold-terminal-singleton-read-retains-its-own-storage-type-and-minimum
  (let [transform (body/->ScalarRegion
                   '[completed resident] '(+ completed (double (clojure.core/aget resident 0)))
                   [{:sym 'resident :dtype :float :map (axis-map/singleton-broadcast)}] :double)
        op (-> (operation :double 2.0 1 '*)
               (update :inputs conj 'resident)
               (assoc-in [:reduction :attributes :resident-scalar-captures] #{'resident})
               (assoc-in [:reduction :attributes :result-region] transform))
        certificate (fold/schedule op {:array-types {'a :double 'resident :float}})
        resident (some #(when (= 'resident (:id %)) %) (get-in certificate [:body :parameters]))]
    (is (= :float (:dtype resident)))
    (is (= [1] (:shape resident)))
    (is (= 1 (:minimum-elements resident)))
    (is (= :double (get-in certificate [:numerics :result-transform :result-dtype])))
    (is (= '[a resident result 1] (:arguments certificate)))))

(deftest ordered-fold-emits-through-all-c-family-targets
  (doseq [dtype [:float :double] target [:opencl-portable :cuda :hip]]
    (let [op (operation dtype (if (= :float dtype) (float 2.0) 2.0) 1 '*)
          artifact (emit/generate-segred-kernel op nil :dtype dtype :array-types {'a dtype}
                                                :target-dialect target)
          certificate (get-in artifact [:provenance :scheduled-operation])]
      (is (string? (:source artifact)))
      (is (= ['a 'result 1] (:arguments artifact)))
      (is (= :source-ordered-seeded-fold (get-in certificate [:numerics :policy])))
      (is (= artifact (scheduled/validate-artifact-projection! certificate artifact))))))

(defn- run-device-folds [target]
  (let [runtime (backend/runtime-namespace target)
        api #(requiring-resolve (symbol (str runtime) %))
        arena ((api "make-kernel-arena!"))]
    (try
      (with-bindings {(api "*current-arena*") arena}
        (doseq [dtype [:float :double]
                [seed xs bound operator expected]
                [[2.0 [3.0] 1 '* 6.0]
                 [2.0 [42.0] 0 '* 2.0]
                 [-0.0 [3.0] 1 '* -0.0]
                 [0.0 [1.0e16 1.0 -1.0e16] 3 '+ 0.0]]]
          (let [array-of (if (= :float dtype) float-array double-array)
                seed (if (= :float dtype) (float seed) seed)
                op (operation dtype seed bound operator)
                artifact (emit/generate-segred-kernel op nil :dtype dtype :array-types {'a dtype}
                                                      :target-dialect :opencl-portable)
                input ((api "buffer-of-array") (array-of xs) dtype)
                output ((api "buffer-of-array") (array-of [42.0]) dtype)
                bits (if (= :float dtype) #(Float/floatToRawIntBits (float %))
                         #(Double/doubleToRawLongBits (double %)))]
            (try
              ((api "register-kernel!") (:kernel-name artifact) artifact)
              ((api "launch-registered-bound!")
               ((api "bind-kernel-call")
                (call/make artifact [input output {:type :int :value bound}])))
              (is (= (bits expected) (bits (first ((api "buffer->array") output)))))
              (finally
                ((api "free-buffer!") input)
                ((api "free-buffer!") output))))))
      (finally ((api "close-kernel-arena!") arena)))))

(deftest native-ordered-fold-executes-the-seeded-recurrence
  ;; Native schedule qualification only: no original nullable input/read completion is inferred.
  (if @ocl/opencl-available?
    (run-device-folds :ocl:0)
    (ocl/opencl-skip! "ordered scalar seeded fold on OpenCL"))
  (if @ze/gpu-available?
    (run-device-folds :ze:0)
    (ze/gpu-skip! "ordered scalar seeded fold on Level Zero")))
