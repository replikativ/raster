(ns raster.compiler.ir.link-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link]))

(def ^:private kernel
  (artifact/make
   {:kernel-name "link_axpy"
    :source (str "__kernel void link_axpy(__global const float* x, "
                 "__global const float* w, __global float* y, long n) {}")
    :abi [(kabi/slot 'x :input :float)
          (kabi/slot 'w :input :float)
          (kabi/slot 'y :output :float)
          (kabi/slot 'n :scalar :long)]
    :arguments '[x w y n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[x w] :writes '[y]}
    :attributes {:strategy :reference}}))

(defn- descriptor []
  {:dtype :float
   :all-params '[x w n]
   :array-params '[x w]
   :scalar-params '[n]
   :array-roles {'x :input 'w :input}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [args] (long (nth args 2)))}]
   :steps [{:phase :map :kernel-name "link_axpy" :convention :map
            :artifact kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :input :sym 'w}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 2)))}]}]
   :result-sym 'y})

(defn- n [id role & [source]]
  (link/node {:id id :dtype :float :shape [16] :device :ze:0
              :role role :source source}))

(defn- instance [id x w y]
  (link/instance {:id id :descriptor (descriptor)
                  :bindings {'x x 'w w 'y y}
                  :scalars {'n 16}}))

(defn- valid-plan []
  (let [x (float-array 16) w0 (float-array 16) w1 (float-array 16)]
    (link/make
     {:id :two-layers :target :ze:0
      :nodes [(n :x :input x) (n :w0 :constant w0) (n :w1 :constant w1)
              (n :hidden :internal) (n :out :output)]
      :instances [(instance :layer-0 :x :w0 :hidden)
                  (instance :layer-1 :hidden :w1 :out)]
      :outputs [:out]})))

(deftest data-valued-descriptor-instances-compose-through-stable-node-identities
  (let [plan (valid-plan)]
    (is (link/link-plan? plan))
    (is (= [:layer-0 :layer-1] (mapv :id (:instances plan))))
    (is (= :hidden (get-in plan [:instances 1 :bindings 'x])))
    (is (= [nil nil 16] (link/instance-arguments (first (:instances plan)))))
    (is (= {'x :input 'w :constant 'y :scratch}
           (link/instance-roles plan (first (:instances plan)))))))

(deftest validation-closes-bindings-shapes-dtypes-and-ordered-effects
  (testing "every pointer symbol is explicitly bound"
    (let [i (assoc (instance :layer :x :w0 :out) :bindings {'x :x 'y :out})]
      (is (= :link-bindings
             (:reason (ex-data (try
                                 (link/make {:id :missing :target :ze:0
                                             :nodes [(n :x :input (float-array 16))
                                                     (n :w0 :constant (float-array 16))
                                                     (n :out :output)]
                                             :instances [i] :outputs [:out]})
                                 (catch clojure.lang.ExceptionInfo e e))))))))
  (testing "descriptor allocation shapes are exact"
    (let [bad-out (link/node {:id :out :dtype :float :shape [15] :device :ze:0 :role :output})]
      (is (= :link-allocation-shape
             (:reason (ex-data (try
                                 (link/make {:id :shape :target :ze:0
                                             :nodes [(n :x :input (float-array 16))
                                                     (n :w :constant (float-array 16)) bad-out]
                                             :instances [(instance :layer :x :w :out)]
                                             :outputs [:out]})
                                 (catch clojure.lang.ExceptionInfo e e))))))))
  (testing "ABI storage dtype is authoritative"
    (let [bad-x (link/node {:id :x :dtype :int :shape [16] :device :ze:0
                            :role :input :source (int-array 16)})]
      (is (= :link-node-dtype
             (:reason (ex-data (try
                                 (link/make {:id :dtype :target :ze:0
                                             :nodes [bad-x (n :w :constant (float-array 16))
                                                     (n :out :output)]
                                             :instances [(instance :layer :x :w :out)]
                                             :outputs [:out]})
                                 (catch clojure.lang.ExceptionInfo e e))))))))
  (testing "a flat node cannot silently stand in for a multi-buffer SoA value"
    (let [soa-kernel
          (artifact/make
           {:kernel-name "link_soa"
            :source "__kernel void link_soa(float* x_a, float* x_b, float* y, long n) {}"
            :abi [(kabi/slot 'x_a :input :float :binding 'x)
                  (kabi/slot 'x_b :input :float :binding 'x)
                  (kabi/slot 'y :output :float)
                  (kabi/slot 'n :scalar :long)]
            :arguments '[x_a x_b y n]
            :launch (launch/spec {:workgroup-size [64]
                                  :group-count [(launch/ceil-div 'n 64)]})
            :effects {:kind :map :reads '[x_a x_b] :writes '[y]}})
          descriptor {:dtype :float
                      :all-params '[x n]
                      :array-params '[x]
                      :scalar-params '[n]
                      :allocs [{:sym 'y :dtype :float
                                :size-fn (fn [args] (long (nth args 1)))}]
                      :steps [{:phase :map :kernel-name "link_soa" :convention :map
                               :artifact soa-kernel :logical-bindings? true
                               :argument-specs [{:kind :pointer :sym 'x}
                                                {:kind :output :sym 'y}
                                                {:kind :scalar :type :long
                                                 :value-fn (fn [args] (long (nth args 1)))}]}]
                      :result-sym 'y}
          i (link/instance {:id :soa :descriptor descriptor
                            :bindings {'x :x 'y :out} :scalars {'n 16}})]
      (is (= :link-composite-binding-required
             (:reason (ex-data
                       (try
                         (link/make {:id :soa :target :ze:0
                                     :nodes [(n :x :input (float-array 16)) (n :out :output)]
                                     :instances [i] :outputs [:out]})
                         (catch clojure.lang.ExceptionInfo e e))))))))
  (testing "an internal consumer cannot precede its producer"
    (is (= :link-read-before-write
           (:reason (ex-data (try
                               (let [x (float-array 16) w0 (float-array 16) w1 (float-array 16)]
                                 (link/make
                                  {:id :order :target :ze:0
                                   :nodes [(n :x :input x) (n :w0 :constant w0)
                                           (n :w1 :constant w1) (n :hidden :internal)
                                           (n :out :output)]
                                   :instances [(instance :consumer :hidden :w1 :out)
                                               (instance :producer :x :w0 :hidden)]
                                   :outputs [:out]}))
                               (catch clojure.lang.ExceptionInfo e e))))))))
  (testing "external storage does not initialize an internal value"
    (let [hidden (link/node {:id :hidden :dtype :float :shape [16] :device :ze:0
                             :role :internal :ownership :external})]
      (is (= :link-read-before-write
             (:reason (ex-data (try
                                 (link/make
                                  {:id :external-order :target :ze:0
                                   :nodes [(n :w :constant (float-array 16)) hidden
                                           (n :out :output)]
                                   :instances [(instance :consumer :hidden :w :out)]
                                   :outputs [:out]})
                                 (catch clojure.lang.ExceptionInfo e e))))))))

(deftest physical-view-aliases-are-explicit-and-effect-checked
  (let [allocation (bview/allocation {:id :shared :byte-size 96 :memory-space :device
                                      :device :ze:0 :ownership :owned})
        x (link/node {:id :x :view (bview/view allocation
                                               {:id :x :byte-offset 0 :dtype :float :shape [16]})
                      :role :input :source (float-array 16)})
        out (link/node {:id :out :view (bview/view allocation
                                                   {:id :out :byte-offset 32 :dtype :float
                                                    :shape [16]})
                        :role :output})
        w (n :w :constant (float-array 16))
        make-plan #(link/make {:id :alias :target :ze:0 :nodes [x w out]
                               :instances [(instance :layer :x :w :out)]
                               :outputs [:out] :aliases %})]
    (is (= :link-undeclared-alias
           (:reason (ex-data (try (make-plan #{})
                                  (catch clojure.lang.ExceptionInfo e e))))))
    (is (= :link-same-step-alias-hazard
           (:reason (ex-data (try (make-plan #{#{:x :out}})
                                  (catch clojure.lang.ExceptionInfo e e))))))))

(deftest scatter-expansion-retains-a-pure-link-contract
  (let [descriptor {:dtype :float
                    :all-params '[out src index n]
                    :array-params '[out src index]
                    :scalar-params '[n]
                    :array-roles {'out :output 'src :input 'index :input}
                    :allocs []
                    :steps [{:phase :scatter :kernel-name "scatter"
                             :convention :scatter :arrays '[out src index]
                             :n-fn (fn [args] (long (nth args 3))) :scalar-specs []}]
                    :result-sym 'out}
        nodes [(n :out :output)
               (link/node {:id :src :dtype :float :shape [16] :device :ze:0
                           :role :input :source (float-array 16)})
               (link/node {:id :index :dtype :int :shape [16] :device :ze:0
                           :role :input :source (int-array 16)})]
        instance (link/instance {:id :scatter :descriptor descriptor
                                 :bindings {'out :out 'src :src 'index :index}
                                 :scalars {'n 16}})]
    (is (link/link-plan? (link/make {:id :scatter :target :ze:0 :nodes nodes
                                     :instances [instance] :outputs [:out]})))
    (is (= :link-target-convention
           (:reason (ex-data
                     (try
                       (link/make {:id :scatter-ocl :target :ocl:0
                                   :nodes (mapv #(update-in % [:view :allocation]
                                                            assoc :device :ocl:0)
                                                nodes)
                                   :instances [instance] :outputs [:out]})
                       (catch clojure.lang.ExceptionInfo e e))))))))
