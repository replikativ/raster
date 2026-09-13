(ns raster.compiler.ir.kernel-dispatch-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as kgcall]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

(def ^:private abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'width :scalar :long :role :shape)])

(defn- artifact
  [name strategy workgroup]
  (kart/make
   {:kernel-name name
    :source (str "__kernel void " name
                 "(__global const float* x, __global float* out, long width) {}")
    :abi abi
    :arguments '[x out width]
    :launch (klaunch/spec {:workgroup-size [workgroup]
                           :group-count [(klaunch/ceil-div 'width workgroup)]})
    :effects {:kind :map :reads ['x] :writes ['out]}
    :attributes {:strategy strategy}}))

(def ^:private reference
  (artifact "dispatch_reference" :reference 64))

(def ^:private subgroup
  (artifact "dispatch_subgroup" :subgroup-score-reuse 16))

(deftest compiler-requirements-cannot-collide-under-one-entry-point
  (let [cl3 (-> reference
                (assoc-in [:attributes :strategy] :cl3)
                (assoc-in [:attributes :compilation] {:language-standard "CL3.0"}))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"conflicting modules"
                         (kdispatch/make {:id "compilation-conflict"
                                          :alternatives [reference cl3]
                                          :default-strategy :reference
                                          :selector {:kind :fixed-strategy :strategy :reference}})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not yet consume"
                         (ze/register-kernel! "compile_contract_rejected" cl3)))
    (is (nil? (ze/kernel-registry-entry "compile_contract_rejected")))))

(deftest opencl-compilation-options-check-exact-device-extension-tokens
  (let [options (ns-resolve 'raster.gpu.ocl-runtime 'compilation-options)
        req {:language-standard "CL3.0" :extensions #{"cl_khr_integer_dot_product"}}]
    (is (nil? (options {} {})))
    (is (= "-cl-std=CL3.0"
           (options req {:extensions "cl_other cl_khr_integer_dot_product\ncl_last"})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lacks required"
                         (options req {:extensions "cl_khr_integer_dot_product_suffix"})))))

(def ^:private dispatch
  (kdispatch/make
   {:id "dispatch-test"
    :alternatives [reference subgroup]
    :default-strategy :reference
    :selector {:kind :runtime-scalar-threshold
               :argument 'width
               :threshold 256
               :at-least :subgroup-score-reuse
               :otherwise :reference}}))

(deftest binding-admission-separates-preference-from-applicability
  (let [arguments [:x :out 512]
        hazard [{:reason :kernel-graph-writable-alias :left 'x :right 'out}]
        calls (atom [])
        preflight (fn [executable]
                    (swap! calls conj (kexec/strategy executable))
                    (if (= executable subgroup) hazard []))
        admission (kdispatch/admit-alternative dispatch arguments preflight)]
    (is (= reference (:executable admission)))
    (is (= [:subgroup-score-reuse :reference] @calls))
    (is (= [{:strategy :subgroup-score-reuse :violations hazard}
            {:strategy :reference :violations []}]
           (:attempts admission)))
    (testing "a cached fixed preference is still subject to admission"
      (is (= reference
             (:executable (kdispatch/admit-alternative
                           (kdispatch/with-selector dispatch
                             {:kind :fixed-strategy :strategy :subgroup-score-reuse})
                           arguments :auto preflight)))))
    (testing "disjoint inputs retain the preferred executable"
      (is (= subgroup (:executable (kdispatch/admit-alternative
                                   dispatch arguments (constantly []))))))
    (testing "an explicit request never silently falls back"
      (reset! calls [])
      (let [error (try (kdispatch/admit-alternative dispatch arguments
                                                   :subgroup-score-reuse preflight)
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
        (is (= :kernel-dispatch-inapplicable (:reason error)))
        (is (= [:subgroup-score-reuse] @calls))))
    (testing "a rejected default is checked only once"
      (let [calls (atom 0)
            error (try (kdispatch/admit-alternative
                        dispatch [:x :out 32]
                        (fn [_] (swap! calls inc) hazard))
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
        (is (= 1 @calls))
        (is (= [{:strategy :reference :violations hazard}] (:attempts error)))))
    (testing "both rejections remain available to diagnostics"
      (let [error (try (kdispatch/admit-alternative dispatch arguments (constantly hazard))
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
        (is (= [:subgroup-score-reuse :reference] (mapv :strategy (:attempts error))))
        (is (= [hazard hazard] (mapv :violations (:attempts error))))))
    (testing "preflight failures are not treated as a reason to select another kernel"
      (let [failure (ex-info "broken preflight" {:reason :unexpected})]
        (is (identical? failure
                        (try (kdispatch/admit-alternative dispatch arguments
                                                        (fn [_] (throw failure)))
                             (catch Exception e e))))))
    (testing "a malformed preflight cannot accidentally admit a kernel"
      (doseq [result [nil false {} [nil] [{}] [{:reason "not-a-keyword"}]]]
        (is (= :kernel-dispatch-admission-result
               (try (kdispatch/admit-alternative dispatch arguments (constantly result))
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest admission-reuses-abi-alias-contract-authority
  (let [leaf (assoc-in subgroup [:abi 0 :aliasing] :no-write-alias)
        strict (kgraph/make
                {:inputs [(kgraph/buffer 'x :float 'width :device :input)]
                 :outputs [(kgraph/buffer 'out :float 'width :device :output)]
                 :temporaries [] :abi abi :arguments '[x out width]
                 :scalars [(kgraph/scalar 'width :long)]
                 :nodes [(kgraph/->ScheduledKernel
                          :compute leaf
                          [(kgraph/->ValueUse 'x :read) (kgraph/->ValueUse 'out :write)]
                          #{'width} [])]
                 :effects (:effects reference)
                 :attributes {:strategy :subgroup-score-reuse}})
        choice (assoc dispatch :alternatives [reference strict])
        admit (fn [arguments override]
                (kdispatch/admit-alternative
                 choice arguments override
                 #(kabi/alias-contract-violations
                   (if (= % strict) (:abi leaf) (kexec/abi %)) arguments =)))
        overlapping [:same :same 512]
        violations (kabi/alias-contract-violations (:abi leaf) overlapping =)]
    (is (= reference (:executable (admit overlapping :auto))))
    (is (= strict (:executable (admit [:left :right 512] :auto))))
    (is (= 1 (count violations)))
    (is (= :kernel-abi-no-write-alias (:reason (first violations))))
    (is (= (first violations)
           (try (kabi/validate-alias-contracts! (:abi leaf) overlapping =)
                (catch clojure.lang.ExceptionInfo e (ex-data e)))))
    (is (= violations
           (try (admit overlapping :subgroup-score-reuse)
                (catch clojure.lang.ExceptionInfo e
                  (get-in (ex-data e) [:attempts 0 :violations])))))
    (testing "all forbidden pairs are reported, rather than just the first"
      (let [abi [(kabi/slot 'a :input :float :aliasing :no-write-alias)
                 (kabi/slot 'b :input :float :aliasing :no-write-alias)
                 (kabi/slot 'c :output :float)]]
        (is (= 2 (count (kabi/alias-contract-violations abi [:same :same :same] =))))))))

(deftest forced-and-cached-choices-still-enforce-binding-preconditions
  (let [guarded (assoc subgroup :preconditions [{:expression 'width :op :>= :value 256}])
        choice (assoc dispatch :alternatives [reference guarded])
        arguments [:x :out {:type :long :value 32}]
        fixed (assoc choice :selector {:kind :fixed-strategy :strategy :subgroup-score-reuse})]
    (is (= reference (kdispatch/select-alternative choice arguments)))
    (doseq [selected [(kdispatch/select-alternative choice arguments :subgroup-score-reuse)
                      (kdispatch/select-alternative fixed arguments)]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar precondition failed"
                            (kcall/make selected arguments))))))

(deftest shared-entry-points-cannot-hide-different-preconditions
  (let [a (artifact "shared_constraint_entry" :a 64)
        b (assoc (artifact "shared_constraint_entry" :b 64)
                 :preconditions [{:expression 'width :op :>= :value 256}])]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"conflicting modules"
                          (kdispatch/make {:id "conflicting-constraints" :alternatives [a b]
                                           :default-strategy :a
                                           :selector {:kind :fixed-strategy :strategy :a}})))))

(deftest logical-scalars-narrow-only-at-the-target-abi
  (let [narrowed (assoc-in reference [:abi 2 :kernel-dtype] :int)]
    (is (= {:type :int :value 64}
           (last (kexec/typed-runtime-arguments
                  narrowed [:x :out {:type :long :value 64}]))))
    (is (thrown? ArithmeticException
                 (kexec/typed-runtime-arguments
                  narrowed [:x :out {:type :long :value (inc (long Integer/MAX_VALUE))}])))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"wrong logical ABI dtype"
         (kexec/typed-runtime-arguments
          narrowed [:x :out {:type :int :value 64}])))))

(defn- staged-graph
  []
  (let [temporary 'dispatch-temporary
        stage (fn [kernel-name input output]
                (kart/make
                 {:kernel-name kernel-name
                  :source (str "__kernel void " kernel-name
                               "(__global const float* " input ", __global float* " output
                               ", long width) {}")
                  :abi [(kabi/slot input :input :float)
                        (kabi/slot output :output :float)
                        (kabi/slot 'width :scalar :long)]
                  :arguments [input output 'width]
                  :launch (klaunch/spec {:workgroup-size [64]
                                         :group-count [(klaunch/ceil-div 'width 64)]})
                  :effects {:kind :stage}}))
        first-stage (stage "dispatch_stage_one" 'x temporary)
        second-stage (stage "dispatch_stage_two" temporary 'out)]
    (kgraph/make
     {:inputs [(kgraph/buffer 'x :float 'width :device :input)]
      :outputs [(kgraph/buffer 'out :float 'width :device :output)]
      :temporaries [(kgraph/buffer temporary :float 'width :device :temporary)]
      :abi abi
      :arguments '[x out width]
      :scalars [(kgraph/scalar 'width :long)]
      :nodes [(kgraph/->ScheduledKernel
               :stage-one first-stage
               [(kgraph/->ValueUse 'x :read)
                (kgraph/->ValueUse temporary :write)] #{'width} [])
              (kgraph/->ScheduledKernel
               :stage-two second-stage
               [(kgraph/->ValueUse temporary :read)
                (kgraph/->ValueUse 'out :write)] #{'width} [:stage-one])]
      :effects (:effects reference)
      :attributes {:strategy :two-stage}})))

(defn- direct-graph []
  (kgraph/make
   {:inputs [(kgraph/buffer 'x :float 'width :device :input)]
    :outputs [(kgraph/buffer 'out :float 'width :device :output)]
    :temporaries [] :abi abi :arguments '[x out width]
    :scalars [(kgraph/scalar 'width :long)]
    :nodes [(kgraph/->ScheduledKernel
             :direct (assoc-in subgroup [:abi 0 :aliasing] :no-write-alias)
             [(kgraph/->ValueUse 'x :read) (kgraph/->ValueUse 'out :write)] #{'width} [])]
    :effects (:effects reference) :attributes {:strategy :direct}}))

(defn- storage-dispatch []
  (kdispatch/make
   {:id "storage-dispatch" :alternatives [(staged-graph) (direct-graph)]
    :default-strategy :two-stage
    :selector {:kind :fixed-strategy :strategy :direct}}))

(deftest graph-storage-admission-includes-internal-abi-and-physical-ranges
  (let [allocation (bview/allocation {:id :admission :byte-size 64 :memory-space :device
                                     :ownership :borrowed})
        left (bview/view allocation {:id :left :dtype :float :shape [8]})
        overlap (bview/view allocation {:id :overlap :byte-offset 16 :dtype :float :shape [8]})
        right (bview/view allocation {:id :right :byte-offset 32 :dtype :float :shape [8]})
        check #(kgcall/binding-alias-violations %1 {'x left 'out %2} kcall/pointer-overlaps?)]
    (is (empty? (check (staged-graph) overlap))
        "the ordered copy stage snapshots the overlapping source")
    (is (empty? (check (direct-graph) right)))
    (let [violations (check (direct-graph) overlap)]
      (is (= #{:kernel-graph-writable-alias :kernel-abi-no-write-alias}
             (set (map :reason violations))))
      (is (every? #(= :direct (:node %)) violations)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exactly every public"
                         (kgcall/binding-alias-violations (direct-graph) {'x left}
                                                         kcall/pointer-overlaps?)))))

(deftest staged-storage-admission-precedes-session-and-preserves-snapshot-fallback
  (let [selected (atom [])
        input (float-array 3)
        separate (float-array 3)]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "kernel-dispatch-registry-entry" (constantly (storage-dispatch))
           "device-buffer?" (constantly false)))
       #'gpu/with-gpu-session* (fn [_ body] (body (atom {:device-id :probe})))
       #'gpu/alloc! (fn [& _])
       #'gpu/bind-kernel-executable!
       (fn [_ _ executable _] (swap! selected conj (kexec/strategy executable)) :handle)
       #'gpu/run-kernel-graph! (fn [& _])
       #'gpu/download-range! (fn [& _])}
      (fn []
        (gpu/invoke-staged-executable! :probe "storage-dispatch" [input input 3])
        (gpu/invoke-staged-executable! :probe "storage-dispatch" [input separate 3])
        (is (= [:two-stage :direct] @selected))))
    (testing "a rejected sole default cannot open a session"
      (with-redefs-fn
        {#'raster.gpu.core/rt-resolve
         (fn [_ function-name]
           (case function-name
             "kernel-dispatch-registry-entry"
             (constantly (kdispatch/make
                          {:id "strict-only" :alternatives [(direct-graph)]
                           :default-strategy :direct
                           :selector {:kind :fixed-strategy :strategy :direct}}))
             "device-buffer?" (constantly false)))
         #'gpu/with-gpu-session* (fn [& _] (throw (AssertionError. "opened session")))}
        #(is (= :kernel-dispatch-inapplicable
                (try (gpu/invoke-staged-executable! :probe "strict-only" [input input 3])
                     (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest staged-admission-does-not-impose-an-unselected-default-extent
  (let [choice (update (storage-dispatch) :alternatives
                       (fn [[snapshot direct]]
                         [(assoc-in snapshot [:inputs 0 :elements] 4) direct]))
        input (float-array 3)
        opened? (atom false)]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "kernel-dispatch-registry-entry" (constantly choice)
           "device-buffer?" (constantly false)))
       #'gpu/with-gpu-session*
       (fn [& _] (reset! opened? true) (throw (ex-info "past preflight" {:reason :past-preflight})))}
      (fn []
        (is (= :past-preflight
               (try (gpu/invoke-staged-executable! :probe "storage-dispatch"
                                                  [input (float-array 3) 3])
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is @opened?)
        (reset! opened? false)
        (is (= :staged-graph-buffer-capacity
               (try (gpu/invoke-staged-executable! :probe "storage-dispatch" [input input 3])
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (false? @opened?) "the selected fallback must meet its own capacity contract")))))

(deftest execution-info-requires-a-live-existing-binding
  (let [session (atom {:prepared {:manual {}}})]
    (is (nil? (gpu/execution-info session :manual))
        "manual bindings must not manufacture compiler evidence")
    (is (= :gpu-execution-info-missing
           (try (gpu/execution-info session :missing)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (swap! session assoc :closed? true)
    (is (= :gpu-execution-info-closed
           (try (gpu/execution-info session :manual)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest resident-storage-admission-precedes-backend-binding
  (let [step {:kernel-name "storage-dispatch" :phase :probe :convention :executable
              :dispatch (storage-dispatch)
              :strategy-selection {:path [:strategy] :mapping {:direct :direct} :default :auto}
              :argument-specs [{:kind :input :sym 'x} {:kind :output :sym 'out}
                               {:kind :scalar :type :long :value-fn (constantly 3)}]}
        selected (atom [])
        bind! (fn [input output schedule]
                (gpu/bind-step! (atom {:device-id :probe :buffers {'x input 'out output}})
                                step {} identity {:schedule schedule}))]
    (with-redefs-fn
      {#'raster.gpu.core/bind-selected-executable
       (fn [_ executable & _]
         (swap! selected conj (kexec/strategy executable))
         (gpu/->BoundExecutableStep [] {} []))}
      (fn []
        (let [fallback (gpu/execution-info (bind! :same :same {}) :probe)
              direct (gpu/execution-info (bind! :same :separate {}) :probe)]
          (is (= :two-stage (:strategy fallback)))
          (is (= :direct (:strategy direct)))
          (is (= [:direct :two-stage] (mapv :strategy (:admission fallback))))
          (is (seq (get-in fallback [:admission 0 :reasons])))
          (is (= [] (get-in fallback [:admission 1 :reasons])))
          (is (= [{:strategy :direct :reasons []}] (:admission direct)))
          (is (= (kexec/entry-points (kdispatch/default-alternative (storage-dispatch)))
                 (:entry-points fallback))))
        (is (= [:two-stage :direct] @selected))
        (is (= :kernel-dispatch-inapplicable
               (try (bind! :same :same {:strategy :direct})
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (= [:two-stage :direct] @selected)
            "explicit rejection did not enter the backend binder")
        (testing "distinct resident wrappers retain their physical range facts"
          (let [segment (java.lang.foreign.MemorySegment/ofArray (float-array 16))
                buffer (fn [offset] {:segment (.asSlice segment offset 32)
                                     :dtype :float :n-elements 8 :byte-size 32})]
            (reset! selected [])
            (bind! (buffer 0) (buffer 16) {})
            (bind! (buffer 0) (buffer 32) {})
            (is (= [:two-stage :direct] @selected))))))))

(deftest graph-node-compilation-requirements-invalidate-tuning
  (let [graph (staged-graph)
        changed (assoc-in graph [:nodes 1 :operation :attributes :compilation]
                          {:language-standard "CL3.0"})
        before (tuning/executable-signature graph)
        after (tuning/executable-signature changed)]
    (is (not= (:source-hash before) (:source-hash after)))
    (is (= (dissoc before :source-hash) (dissoc after :source-hash)))))

(deftest runtime-scalars-select-an-abi-compatible-artifact
  (is (kdispatch/kernel-dispatch? dispatch))
  (is (= "dispatch_reference"
         (:kernel-name (kdispatch/select-alternative dispatch [:x :out {:type :long :value 128}]))))
  (is (= "dispatch_subgroup"
         (:kernel-name (kdispatch/select-alternative dispatch [:x :out {:type :long :value 256}]))))
  (is (= "dispatch_subgroup"
         (:kernel-name (kdispatch/select-alternative dispatch [:x :out {:type :long :value 2}]
                                                     :subgroup-score-reuse)))
      "an explicit schedule override is authoritative"))

(deftest measured-runtime-ranges-select-without-runtime-tuning
  (let [measured (kdispatch/with-selector
                   dispatch
                   {:kind :runtime-scalar-ranges
                    :argument 'width
                    :below :reference
                    :ranges [{:at-least 192 :strategy :subgroup-score-reuse}
                             {:at-least 768 :strategy :reference}]})
        select #(kdispatch/alternative-strategy
                 (kdispatch/select-alternative measured [:x :out {:type :long :value %}]))]
    (is (= :reference (select 128)))
    (is (= :subgroup-score-reuse (select 192)))
    (is (= :subgroup-score-reuse (select 512)))
    (is (= :reference (select 768)))
    (is (= :subgroup-score-reuse
           (kdispatch/alternative-strategy
            (kdispatch/select-alternative measured [:x :out {:type :long :value 1}]
                                          :subgroup-score-reuse)))
        "an explicit override remains authoritative over measured selector data")))

(deftest ordered-expression-cases-select-from-checked-shape-arithmetic
  (let [scheduled
        (kdispatch/with-selector
          dispatch
          {:kind :runtime-expression-cases
           :cases [{:expression 'width :op :< :value 8 :strategy :reference}
                   {:expression (klaunch/product 'width 2)
                    :op :>= :value 512 :strategy :subgroup-score-reuse}]
           :default :reference})
        select #(kdispatch/alternative-strategy
                 (kdispatch/select-alternative
                  scheduled [:x :out {:type :long :value %}]))]
    (is (= :reference (select 4)))
    (is (= :reference (select 128)))
    (is (= :subgroup-score-reuse (select 256)))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"outside its scalar ABI"
         (kdispatch/with-selector
           dispatch
           {:kind :runtime-expression-cases
            :cases [{:expression 'not-in-the-abi :op :> :value 0 :strategy :reference}]
            :default :subgroup-score-reuse})))))

(deftest dispatch-preserves-one-interface-across-single-and-multi-kernel-schedules
  (let [graph (staged-graph)
        mixed (kdispatch/make
               {:id "mixed-executable-dispatch"
                :alternatives [reference graph]
                :default-strategy :reference
                :selector {:kind :runtime-scalar-threshold
                           :argument 'width :threshold 256
                           :at-least :two-stage :otherwise :reference}})]
    (is (= :kernel-artifact
           (kexec/kind (kdispatch/select-alternative
                        mixed [:x :out {:type :long :value 128}]))))
    (is (= :kernel-graph
           (kexec/kind (kdispatch/select-alternative
                        mixed [:x :out {:type :long :value 256}]))))
    (is (= '[x out width] (kexec/arguments graph)))
    (is (= ['dispatch-temporary] (mapv :id (:temporaries graph))))))

(deftest dispatch-rejects-incompatible-or-ambiguous-alternatives
  (testing "an alternative cannot silently change the ordered ABI"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"share target, ABI"
         (kdispatch/make
          {:id "bad-abi"
           :alternatives [reference (assoc subgroup :arguments '[out x width])]
           :default-strategy :reference
           :selector (:selector dispatch)}))))
  (testing "strategy identity is unique"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"strategies must be unique"
         (kdispatch/make
          {:id "duplicate"
           :alternatives [reference (assoc subgroup :attributes {:strategy :reference})]
           :default-strategy :reference
           :selector (:selector dispatch)}))))
  (testing "measured range boundaries are ordered and name available strategies"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"strictly increasing"
         (kdispatch/with-selector
           dispatch {:kind :runtime-scalar-ranges :argument 'width :below :reference
                     :ranges [{:at-least 256 :strategy :subgroup-score-reuse}
                              {:at-least 128 :strategy :reference}]})))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"absent strategy"
         (kdispatch/with-selector
           dispatch {:kind :runtime-scalar-ranges :argument 'width :below :unknown
                     :ranges []}))))
  (testing "a runtime selector must name exactly one scalar ABI position"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"must name a scalar"
         (kdispatch/make
          {:id "pointer-selector"
           :alternatives [reference subgroup]
           :default-strategy :reference
           :selector {:kind :runtime-scalar-threshold :argument 'x :threshold 1
                      :at-least :subgroup-score-reuse :otherwise :reference}}))))
  (testing "shared entry points are legal when they name the same module"
    (is (kdispatch/kernel-dispatch?
         (kdispatch/make
          {:id "shared-entry-point"
           :alternatives [reference
                          (assoc subgroup :kernel-name (:kernel-name reference)
                                 :source (:source reference))]
           :default-strategy :reference
           :selector (:selector dispatch)}))))
  (testing "one entry point cannot name conflicting emitted modules"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"conflicting modules"
         (kdispatch/make
          {:id "duplicate-entry-point"
           :alternatives
           [reference
            (assoc subgroup
                   :kernel-name (:kernel-name reference)
                   :source (-> (:source subgroup)
                               (str/replace "dispatch_subgroup" "dispatch_reference")
                               (str/replace "{}" "{ long gid = get_global_id(0); }")))]
           :default-strategy :reference
           :selector (:selector dispatch)})))))

(deftest both-resident-backends-register-the-same-pure-dispatch
  (doseq [[register! entry] [[ze/register-kernel-dispatch!
                              ze/kernel-dispatch-registry-entry]
                             [ocl/register-kernel-dispatch!
                              ocl/kernel-dispatch-registry-entry]]]
    (register! dispatch)
    (is (identical? dispatch (entry (:id dispatch))))))

(deftest staged-executable-normalizes-the-abi-and-uses-the-common-graph-runner
  (let [graph (staged-graph)
        graph-dispatch (kdispatch/make
                        {:id "staged-graph-dispatch"
                         :alternatives [graph]
                         :default-strategy :two-stage
                         :selector {:kind :fixed-strategy :strategy :two-stage}})
        x (float-array [1.0 2.0 3.0])
        out (float-array [123.0 456.0 789.0])
        calls (atom [])
        session (atom {:device-id :probe})]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "kernel-dispatch-registry-entry" #(when (= % "staged-graph-dispatch")
                                               graph-dispatch)
           "device-buffer?" (constantly false)
           (throw (ex-info "unexpected runtime resolution" {:function function-name}))))
       #'gpu/with-gpu-session*
       (fn [device-id body]
         (swap! calls conj [:session device-id])
         (body session))
       #'gpu/alloc!
       (fn [_ specs] (swap! calls conj [:alloc specs]))
       #'gpu/bind-kernel-executable!
       (fn [_ key executable arguments]
         (swap! calls conj [:bind key executable arguments])
         :handle)
       #'gpu/run-kernel-graph!
       (fn [_ handle] (swap! calls conj [:run handle]))
       #'gpu/download-range!
       (fn [_ key dst spec]
         (swap! calls conj [:download key dst spec])
         (dotimes [i (alength ^floats dst)] (aset ^floats dst i (float (inc i))))
         dst)}
      (fn []
        (is (identical? out
                        (gpu/invoke-staged-executable!
                         :probe "staged-graph-dispatch" [x out 3])))
        (is (= [1.0 2.0 3.0] (mapv double out)))
        (let [[_ _ bound-executable bound-arguments]
              (first (filter #(= :bind (first %)) @calls))]
          (is (identical? graph bound-executable))
          (is (= [:staged-executable-pointer 0] (first bound-arguments)))
          (is (= [:staged-executable-pointer 1] (second bound-arguments)))
          (is (= {:type :long :value 3} (nth bound-arguments 2))))
        (let [allocations (filter #(= :alloc (first %)) @calls)
              output-spec (get (second (second allocations)) [:staged-executable-pointer 1])]
          (is (= 2 (count allocations)))
          (is (identical? out (nth output-spec 2))
              "write-only ABI access does not prove full overwrite of caller storage"))
        (is (= 1 (count (filter #(= :download (first %)) @calls))))
        (reset! calls [])
        (is (nil? (gpu/invoke-staged-executable!
                   :probe "staged-graph-dispatch" [x out 3] :none)))
        (is (= 1 (count (filter #(= :download (first %)) @calls))))))))

(deftest direct-descriptor-graph-binding-checks-capacities-before-driver-work
  (let [graph (assoc-in (staged-graph) [:inputs 0 :elements] (klaunch/product 'width 2))
        step {:convention :executable :artifact graph :phase :test
              :argument-specs [{:kind :input :sym 'x} {:kind :output :sym 'out}
                               {:kind :scalar :type :long :value-fn first}]}
        driver-calls (atom [])]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name] (fn [& _] (swap! driver-calls conj function-name)))}
      (fn []
        (doseq [[input-capacity output-capacity input-dtype width]
                [[5 3 :float 3] [6 2 :float 3] [6 3 :double 3]
                 [Long/MAX_VALUE Long/MAX_VALUE :float Long/MAX_VALUE]]]
          (let [session (atom {:device-id :probe
                               :buffers {:x {:dtype input-dtype :n-elements input-capacity}
                                         :out {:dtype :float :n-elements output-capacity}}})]
            (is (thrown? Exception
                         (gpu/bind-step! session step [width] {'x :x 'out :out})))
            (is (empty? @driver-calls))))))))

(deftest effect-only-staging-copies-every-output-without-a-primary-result
  (let [base (staged-graph)
        extra 'dispatch-temporary
        graph (kexec/validate!
               (-> base
                   (assoc :temporaries [])
                   (update :outputs conj (assoc (first (:temporaries base)) :role :output))
                   (assoc :abi [(first abi) (second abi)
                                (kabi/slot extra :output :float :role :result) (last abi)]
                          :arguments ['x 'out extra 'width])))
        dispatch (kdispatch/make
                  {:id "multi-effect" :alternatives [graph] :default-strategy :two-stage
                   :selector {:kind :fixed-strategy :strategy :two-stage}})
        out (float-array 3)
        intermediate (float-array 3)
        args [(float-array [1 2 3]) out intermediate 3]
        copies (atom [])]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "kernel-dispatch-registry-entry" (constantly dispatch)
           "device-buffer?" (constantly false)))
       #'gpu/with-gpu-session* (fn [_ f] (f (atom {})))
       #'gpu/alloc! (fn [& _])
       #'gpu/bind-kernel-executable! (fn [& _] :handle)
       #'gpu/run-kernel-graph! (fn [& _])
       #'gpu/download-range! (fn [_ _ array _]
                              (swap! copies conj array)
                              (aset ^floats array 0 (float 7)))}
      (fn []
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"more than one primary result"
                             (gpu/invoke-staged-executable! :probe "multi-effect" args)))
        (is (empty? @copies))
        (is (nil? (gpu/invoke-staged-executable! :probe "multi-effect" args :none)))
        (is (= [out intermediate] @copies))
        (is (= 7.0 (aget out 0) (aget intermediate 0)))))))

(deftest staged-graph-capacities-fail-before-opening-a-device-session
  (let [graph (staged-graph)
        graph-dispatch (kdispatch/make
                        {:id "staged-capacity-dispatch"
                         :alternatives [graph]
                         :default-strategy :two-stage
                         :selector {:kind :fixed-strategy :strategy :two-stage}})
        opened? (atom false)]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "kernel-dispatch-registry-entry" (constantly graph-dispatch)
           "device-buffer?" (constantly false)))
       #'gpu/with-gpu-session*
       (fn [& _] (reset! opened? true) (throw (ex-info "session opened" {})))}
      (fn []
        (doseq [[arguments reason]
                [[[ (float-array 2) (float-array 3) 3]
                  :staged-graph-buffer-capacity]
                 [[(float-array 3) (float-array 3) -1]
                  :staged-graph-buffer-extent]]]
          (try
            (gpu/invoke-staged-executable! :probe "staged-capacity-dispatch" arguments)
            (is false "invalid graph storage must fail in pure preflight")
            (catch clojure.lang.ExceptionInfo exception
              (is (= reason (:reason (ex-data exception)))))))
        (is (false? @opened?))))))

(deftest staged-preflight-resolves-opaque-external-extent-leaves
  (let [graph (-> (staged-graph)
                  (assoc-in [:inputs 0 :elements] '(extent x))
                  (assoc-in [:outputs 0 :elements] '(extent x))
                  kgraph/validate!)
        graph-dispatch (kdispatch/make
                        {:id "staged-opaque-extent-dispatch"
                         :alternatives [graph]
                         :default-strategy :two-stage
                         :selector {:kind :fixed-strategy :strategy :two-stage}})
        opened? (atom false)
        invoke (fn [out]
                 (gpu/invoke-staged-executable!
                  :probe "staged-opaque-extent-dispatch"
                  [(float-array 3) out 3]))]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "kernel-dispatch-registry-entry" (constantly graph-dispatch)
           "device-buffer?" (constantly false)))
       #'gpu/with-gpu-session*
       (fn [& _]
         (reset! opened? true)
         (throw (ex-info "past pure preflight" {:reason :past-preflight})))}
      (fn []
        (try (invoke (float-array 3))
             (catch clojure.lang.ExceptionInfo exception
               (is (= :past-preflight (:reason (ex-data exception))))))
        (is @opened?)
        (reset! opened? false)
        (try
          (invoke (float-array 2))
          (is false "cross-buffer opaque extent must still validate output capacity")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :staged-graph-buffer-capacity (:reason (ex-data exception))))))
        (is (false? @opened?))))))

(deftest staged-executable-preserves-pointer-identity-for-in-place-calls
  (let [same (float-array 3)
        calls (atom [])]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "kernel-dispatch-registry-entry" (constantly dispatch)
           "device-buffer?" (constantly false)))
       #'gpu/with-gpu-session* (fn [_ body] (body (atom {:device-id :probe})))
       #'gpu/alloc! (fn [_ specs] (swap! calls conj [:alloc specs]))
       #'gpu/bind-kernel-executable!
       (fn [_ _ _ arguments] (swap! calls conj [:arguments arguments]) :handle)
       #'gpu/run-kernel-graph! (fn [_ _])
       #'gpu/download-range! (fn [_ _ dst _] dst)}
      (fn []
        (gpu/invoke-staged-executable! :probe "dispatch-test" [same same 3])
        (let [arguments (second (first (filter #(= :arguments (first %)) @calls)))]
          (is (= (first arguments) (second arguments))))
        (is (= 1 (count (filter #(= :alloc (first %)) @calls))))))))

(deftest resident-step-selects-before-the-backend-binder
  (let [step {:kernel-name (:kernel-name reference)
              :phase :probe
              :convention :contract
              :artifact reference
              :dispatch dispatch
              :argument-specs [{:kind :input :sym 'x}
                               {:kind :output :sym 'out}
                               {:kind :scalar :type :long
                                :value-fn (fn [args] (:width args))}]}
        selected
        (fn [width]
          (let [session (atom {:device-id :probe
                               :buffers {'x :resident-x 'out :resident-out}})]
            (with-redefs-fn
              {#'raster.gpu.core/rt-resolve
               (fn [_ function-name]
                 (case function-name
                   "register-kernel!" (fn [_ _])
                   "bind-kernel-call" identity
                   (throw (ex-info "unexpected runtime resolution"
                                   {:function function-name}))))}
              #(gpu/bind-step! session step {:width width} identity))
            (get-in @session [:prepared :probe :prepareds 0
                              :artifact :attributes :strategy])))]
    (is (= :reference (selected 128)))
    (is (= :subgroup-score-reuse (selected 256)))))

(deftest resident-step-flattens-and-releases-a-selected-kernel-graph
  (let [graph (staged-graph)
        mixed (kdispatch/make
               {:id "resident-mixed-executable"
                :alternatives [reference graph]
                :default-strategy :reference
                :selector {:kind :runtime-scalar-threshold
                           :argument 'width :threshold 256
                           :at-least :two-stage :otherwise :reference}})
        step {:kernel-name (:kernel-name reference)
              :phase :probe
              :convention :contract
              :artifact reference
              :dispatch mixed
              :argument-specs [{:kind :input :sym 'x}
                               {:kind :output :sym 'out}
                               {:kind :scalar :type :long
                                :value-fn (fn [args] (:width args))}]}
        session (atom {:device-id :probe
                       :buffers {'x {:id :resident-x :dtype :float :n-elements 256}
                                 'out {:id :resident-out :dtype :float :n-elements 256}}
                       :prepared {} :graphs {}})
        recorded (atom [])
        destroyed (atom [])
        freed (atom [])
        resolver
        (fn [_ function-name]
          (case function-name
            "register-kernel!" (fn [_ _])
            "make-buffer" (fn [elements dtype] {:elements elements :dtype dtype})
            "bind-kernel-call" (fn [call] {:kernel-call call})
            "record-graph!" (fn [prepareds]
                              (let [recording {:prepareds (vec prepareds)}]
                                (swap! recorded conj recording)
                                recording))
            "free-buffer!" #(swap! freed conj %)
            (throw (ex-info "unexpected runtime resolution" {:function function-name}))))
        soft-resolver
        (fn [_ function-name]
          (case function-name
            "destroy-prepared!" #(swap! destroyed conj [:prepared %])
            "destroy-graph!" #(swap! destroyed conj [:graph %])
            nil))]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve resolver
       #'raster.gpu.core/rt-resolve-soft soft-resolver}
      (fn []
        (gpu/bind-step! session step {:width 256} identity)
        (is (= 2 (count (get-in @session [:prepared :probe :prepareds]))))
        (is (= 1 (count (get-in @session [:prepared :probe :temporary-buffers]))))
        (gpu/record-graph! session [:probe] :linked)
        (is (= 2 (count (get-in @recorded [0 :prepareds])))
            "recording flattens the semantic step into its ordered kernel launches")
        (gpu/bind-step! session step {:width 128} identity)
        (is (= 2 (count (filter #(= :prepared (first %)) @destroyed)))
            "replacing the phase releases every old graph-node binding")
        (is (= 1 (count @freed)) "replacing the phase releases graph-private storage")
        (gpu/record-graph! session [:probe] :linked)
        (is (= 1 (count (filter #(= :graph (first %)) @destroyed)))
            "re-recording releases the superseded command graph")))))

(deftest resident-step-materializes-and-owns-checked-buffer-views
  (let [root {:root true :dtype :float :n-elements 64 :byte-size 256}
        allocation (bview/allocation {:id :root :byte-size 256 :memory-space :device
                                      :device :ocl:0 :coherence :explicit-transfer
                                      :ownership :owned})
        session (atom {:device-id :ocl:0 :session-id :view-session
                       :buffers {:storage root} :allocations {:storage allocation}
                       :prepared {} :graphs {} :closed? false})
        x-view (gpu/buffer-view session :storage {:shape [16]})
        out-view (gpu/buffer-view session :storage {:byte-offset 64 :shape [16]})
        slices (atom [])
        freed (atom [])
        destroyed (atom [])
        resolver
        (fn [_ name]
          (case name
            "slice-buffer" (fn [buffer byte-offset byte-length dt]
                             (let [slice {:slice true :buffer buffer :byte-offset byte-offset
                                          :byte-size byte-length :n-elements (quot byte-length 4)
                                          :dtype dt}]
                               (swap! slices conj slice)
                               slice))
            "register-kernel!" (fn [& _])
            "bind-kernel-call" (fn [call] {:call call})
            (throw (ex-info "unexpected runtime resolution" {:name name}))))
        soft-resolver
        (fn [_ name]
          (case name
            "destroy-prepared!" #(swap! destroyed conj %)
            "free-buffer!" #(swap! freed conj %)
            nil))
        step {:kernel-name (:kernel-name reference)
              :phase :view-step
              :convention :map
              :artifact reference
              :argument-specs [{:kind :input :sym 'x}
                               {:kind :output :sym 'out}
                               {:kind :scalar :type :long
                                :value-fn (fn [_] 16)}]}]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve resolver
       #'raster.gpu.core/rt-resolve-soft soft-resolver}
      (fn []
        (gpu/bind-step! session step {} {'x x-view 'out out-view})
        (is (= 2 (count @slices))
            "both partial OpenCL ranges are owned cl_mem sub-buffers, including a zero-origin prefix")
        (is (= 2 (count (get-in @session [:prepared :view-step :owned-view-buffers]))))
        (gpu/release-prepared! session :view-step)
        (is (= 1 (count @destroyed)))
        (is (= @slices @freed) "view handles follow the bound step lifetime")
        (is (not-any? #(identical? root %) @freed)
            "the session-owned root allocation survives step release")))))
