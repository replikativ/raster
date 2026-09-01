(ns raster.compiler.ir.kernel-dispatch-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.core :as gpu]
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
      :nodes [(kgraph/->ScheduledKernel
               :stage-one first-stage
               [(kgraph/->ValueUse 'x :read)
                (kgraph/->ValueUse temporary :write)] [])
              (kgraph/->ScheduledKernel
               :stage-two second-stage
               [(kgraph/->ValueUse temporary :read)
                (kgraph/->ValueUse 'out :write)] [:stage-one])]
      :effects (:effects reference)
      :attributes {:strategy :two-stage}})))

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
        out (float-array 3)
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
        (is (= 2 (count (filter #(= :alloc (first %)) @calls))))
        (is (= 1 (count (filter #(= :download (first %)) @calls))))))))

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
                       :buffers {'x :resident-x 'out :resident-out}
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
