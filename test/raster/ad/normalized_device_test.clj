(ns raster.ad.normalized-device-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-device :as device]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.link-module :as link-module]
            [raster.numeric :as numeric]))

(defn- reason [invoke]
  (try (invoke) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- map-program [body]
  (let [tensor (av/tensor {:dtype :float :shape '[n]})
        n (av/tensor {:dtype :long :shape []})
        equation (list '= 'map-0 '[y]
                       (list 'map {:index 'i :extent 'n} '[x] []
                             (soac/lambda-form '[element] [] [body])))]
    (soac/make
     (soac/default-program-facts
      {:values {'n n 'x tensor 'y tensor}
       :inputs '[n x] :equations {'map-0 (soac/default-equation-facts)}})
     [equation] '[y])))

(defn- canonical-owner [body]
  (canonical/adapt-map
   (normalized/linearize-map-program
    (map-program body) (the-ns 'raster.ad.normalized-device-test))))

(defn- arguments [owner xs]
  (mapv (fn [{:keys [id]}] (case id n (long (count xs)) x xs))
        (get-in owner [:interface :inputs])))

(defn- synthetic-descriptor [target] {:device-id target})

(defn- fmul [left right] (float (* (float left) (float right))))
(defn- fadd [left right] (float (+ (float left) (float right))))
(defn- bits [values] (mapv #(Float/floatToRawIntBits (float %)) values))
(defn- primal-oracle [xs] (mapv #(fmul % %) xs))
(defn- transpose-oracle [xs seeds]
  (mapv (fn [x seed] (fadd (fmul seed x) (fmul seed x))) xs seeds))

(defn- download [runtime lease value-id n]
  (let [payload ((ns-resolve 'raster.gpu.link-module 'opaque-payload) runtime)
        resident (get-in @lease [:outputs value-id])
        output (float-array n)]
    ;; Named-entry leases intentionally expose resident views only. This test projection uses the
    ;; private owner session and does not become part of the public normalized-device surface.
    (gpu/download-range! (:session payload) resident output {:elements n})
    output))

(deftest fresh-resident-layout-cloning-preserves-canonical-value-facets
  (let [target :ocl:layout-clone
        template-av (av/tensor {:dtype :float :shape '[n] :ownership :owned
                                :attributes {:canonical :template}})
        target-av (av/tensor {:dtype :float :shape '[n] :ownership :borrowed
                              :attributes {:canonical :seed}})
        node (link-plan/node {:id :template-node :allocation-id :template-allocation
                              :dtype :float :shape [3] :device target :role :output})
        value (link-plan/value {:id :template-value
                                :abstract (assoc template-av :shape [3])
                                :leaves [{:name :value :node :template-node}]})
        layout {:target target :nodes {:template-node node}
                :values {:template-value value}
                :buffer-bindings {'n :n 'result :template-value}
                :scalar-values {'n {:type :long :value (long 3)}
                                'm {:type :long :value (long 3)}}
                :public-inputs []}
        request {:template-value :template-value :template-compiler-value 'result
                 :template-abstract template-av :compiler-value 'seed :abstract target-av
                 :value-id :seed-value :node-id :seed-node :allocation-id :seed-allocation
                 :role :input :project-ownership? true}
        cloned (invocation-link/clone-resident-value-layout layout request)
        projected (get-in cloned [:value :abstract])]
    (is (= [3] (:shape projected)))
    (is (= :owned (:ownership projected)))
    (is (= (dissoc target-av :shape :ownership)
           (dissoc projected :shape :ownership)))
    (is (= [3] (get-in cloned [:node :view :shape])))
    (is (= :seed-allocation (get-in cloned [:node :view :allocation :id])))
    (doseq [[label changed]
            [[:collision (assoc request :node-id :template-node)]
             [:pairwise-collision (assoc request :node-id :seed-value)]
             [:cross-domain-collision (assoc request :allocation-id :template-value)]
             [:dtype (assoc request :abstract (assoc target-av :dtype :double))]
             [:ownership (assoc request :abstract (assoc target-av :ownership :owned))]
             [:symbolic-shape (assoc request :abstract (assoc target-av :shape '[m]))]
             [:template-abstract
              (assoc request :template-abstract
                     (assoc template-av :attributes {:canonical :forged}))]]]
      (testing (name label)
        (is (some? (reason #(invocation-link/clone-resident-value-layout
                             layout changed))))))
    (testing "the compiler template identity is authenticated"
      (is (= :resident-layout-clone-contract
             (reason #(invocation-link/clone-resident-value-layout
                       (assoc-in layout [:buffer-bindings 'result] :other-value)
                       request)))))
    (testing "subviews and shared allocation identities refuse"
      (let [subview-node (link-plan/node
                          {:id :template-node :allocation-id :subview-allocation
                           :dtype :float :shape [3] :byte-offset 4 :byte-size 16
                           :device target :role :output})
            subview (assoc-in layout [:nodes :template-node] subview-node)
            alias-node (link-plan/node
                        {:id :alias-node :view (get-in layout [:nodes :template-node :view])
                         :role :internal})]
        (is (= :resident-layout-clone-storage
               (reason #(invocation-link/clone-resident-value-layout subview request))))
        (is (= :resident-layout-clone-storage
               (reason #(invocation-link/clone-resident-value-layout
                         (assoc-in layout [:nodes :alias-node] alias-node) request))))))))

(deftest zero-residual-maps-use-an-empty-dependency-boundary
  (let [owner (canonical-owner 'element)
        residuals (canonical/execution-residual-plan owner)
        runtime (reify java.io.Closeable (close [_]))
        calls (atom [])
        captured-module (atom nil)]
    (is (empty? (:owned-arrays residuals)))
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [module _]
                                             (reset! captured-module module)
                                             runtime)
                  link-module/run-and-lease!
                  (fn [_ entry request]
                    (swap! calls conj [entry request])
                    (reify java.io.Closeable (close [_])))]
      (let [prepared (device/prepare-owner
                      owner (arguments owner (float-array [1.0 2.0 3.0]))
                      {:target :ocl:normalized-device :schedule {:precision :f32-scalar}})
            ports (:ports prepared)
            seed-abstract (get-in @captured-module [:spine :values (:seed-input ports) :abstract])
            gradient-abstract
            (get-in @captured-module [:spine :values (:gradient-output ports) :abstract])
            primal ((:primal! prepared) (float-array [2.0 3.0 4.0]))
            transpose ((:transpose! prepared) (float-array [1.0 1.0 1.0]) {})
            extra-dependencies {::extra (:lease primal)}
            extra ((:transpose! prepared) (float-array [2.0 2.0 2.0])
                   extra-dependencies)]
        (is (= {} (:dependencies primal)))
        (is (= {} (get-in @calls [1 1 :dependencies])))
        (is (= extra-dependencies (get-in @calls [2 1 :dependencies])))
        (is (empty? (:frontiers @captured-module)))
        (is (= [:raster.ad.normalized-device/primal
                :raster.ad.normalized-device/transpose]
               (mapv :id (get-in @captured-module [:spine :instances]))))
        (is (= :owned (:ownership seed-abstract)))
        (is (= (dissoc (get-in owner [:interface :cotangent-input :value]) :shape :ownership)
               (dissoc seed-abstract :shape :ownership)))
        (is (= (:ownership (get-in owner [:interface :cotangent-result :value]))
               (:ownership gradient-abstract)))
        (is (= (dissoc (get-in owner [:interface :cotangent-result :value]) :shape)
               (dissoc gradient-abstract :shape)))
        (is (every? (comp nil? :source val)
                    (get-in @captured-module [:spine :nodes])))
        (.close ^java.io.Closeable (:lease primal))
        (.close ^java.io.Closeable (:lease transpose))
        (.close ^java.io.Closeable (:lease extra))
        ((:close! prepared))))))

(deftest canonical-proofs-run-during-preparation-not-resident-replay
  (let [owner (canonical-owner '(raster.numeric/* element element))
        validate canonical/validate!
        validations (atom 0)
        runs (atom 0)
        closes (atom 0)
        runtime (reify java.io.Closeable (close [_]))]
    (with-redefs [canonical/validate!
                  (fn [candidate] (swap! validations inc) (validate candidate))
                  hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [_ _] runtime)
                  link-module/run-and-lease!
                  (fn [& _]
                    (swap! runs inc)
                    (reify java.io.Closeable (close [_] (swap! closes inc))))]
      (let [prepared (device/prepare-owner
                      owner (arguments owner (float-array [1.0 2.0 3.0]))
                      {:target :ocl:normalized-device :schedule {:precision :f32-scalar}})
            preparation-validations @validations]
        (try
          (is (pos? preparation-validations))
          (dotimes [_ 2]
            (let [primal ((:primal! prepared) (float-array [2.0 3.0 4.0]))
                  transpose ((:transpose! prepared) (float-array [1.0 2.0 3.0])
                             (:dependencies primal))]
              (.close ^java.io.Closeable (:lease transpose))
              (.close ^java.io.Closeable (:lease primal))))
          (is (= 4 @runs))
          (is (= 4 @closes))
          (is (= preparation-validations @validations)
              "hot closures guard execution identities without reconstructing canonical proofs")
          (finally ((:close! prepared))))))))

(deftest result-publication-closes-an-issued-lease-and-preserves-the-primary-error
  (let [publish (ns-resolve 'raster.ad.normalized-device 'publish-result!)
        primary (ex-info "publication failed" {:reason :publication-primary})
        cleanup (ex-info "lease close failed" {:reason :publication-cleanup})
        closes (atom 0)
        lease (reify java.io.Closeable
                (close [_]
                  (swap! closes inc)
                  (throw cleanup)))]
    (try
      (publish lease #(throw primary) identity)
      (is false "publication failure must escape")
      (catch Throwable actual
        (is (identical? primary actual))
        (is (= 1 @closes))
        (is (= [cleanup] (vec (.getSuppressed actual))))))))

(deftest stale-canonical-authority-stops-before-compilation-and-run
  (let [owner (canonical-owner '(raster.numeric/* element element))
        revision (normalized/revisions)
        compiles (atom 0)
        instantiations (atom 0)]
    (with-redefs [normalized/revisions (fn [] (update revision :compiler-revision inc))
                  equation-first/compile-typed
                  (fn [& _] (swap! compiles inc) (throw (Exception. "compile observed")))
                  link-module/instantiate!
                  (fn [& _] (swap! instantiations inc) (throw (Exception. "instantiate observed")))]
      (is (= :ad-normalized-stale-rules
             (reason #(device/prepare-owner
                       owner (arguments owner (float-array [1.0 2.0 3.0]))
                       {:target :ocl:normalized-device}))))
      (is (zero? @compiles))
      (is (zero? @instantiations)))
    (is (= :normalized-device-target
           (reason #(device/prepare-owner
                     owner (arguments owner (float-array [1.0 2.0 3.0])) {}))))))

(deftest post-run-staleness-closes-the-issued-lease-without-publication
  (let [owner (canonical-owner '(raster.numeric/* element element))
        revision (atom (normalized/revisions))
        runs (atom 0)
        closes (atom 0)
        runtime (reify java.io.Closeable (close [_]))]
    (with-redefs [normalized/revisions #(deref revision)
                  hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [_ _] runtime)
                  link-module/run-and-lease!
                  (fn [& _]
                    (swap! runs inc)
                    (swap! revision update :compiler-revision inc)
                    (reify java.io.Closeable (close [_] (swap! closes inc))))]
      (let [prepared (device/prepare-owner
                      owner (arguments owner (float-array [1.0 2.0 3.0]))
                      {:target :ocl:normalized-device :schedule {:precision :f32-scalar}})]
        (is (= :ad-normalized-stale-rules
               (reason #((:primal! prepared) (float-array [2.0 3.0 4.0])))))
        (is (= 1 @runs))
        (is (= 1 @closes))
        ((:close! prepared))))))

(deftest callee-replacement-stops-every-run-before-the-runtime
  (let [owner (canonical-owner '(raster.numeric/* element element))
        runs (atom 0)
        runtime (reify java.io.Closeable (close [_]))]
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [_ _] runtime)
                  link-module/run-and-lease!
                  (fn [& _] (swap! runs inc) (throw (Exception. "run observed")))]
      (let [prepared (device/prepare-owner
                      owner (arguments owner (float-array [1.0 2.0 3.0]))
                      {:target :ocl:normalized-device :schedule {:precision :f32-scalar}})]
        (with-redefs [numeric/* (fn [& _] (float 999.0))]
          (is (= :ad-normalized-execution-callee
                 (reason #((:primal! prepared) (float-array [2.0 3.0 4.0]))))))
        (is (zero? @runs))
        ((:close! prepared))))))

(defn- run-device-case! [target]
  (let [owner (canonical-owner '(raster.numeric/* element element))
        xs-values [-2.0 0.5 4.0]
        xs (float-array (map float xs-values))
        xs-before (vec xs)
        captured (atom nil)
        instantiate link-module/instantiate!]
    (with-redefs [link-module/instantiate!
                  (fn [module options]
                    (let [runtime (instantiate module options)]
                      (reset! captured runtime)
                      runtime))]
      (let [prepared (device/prepare-owner
                      owner (arguments owner xs)
                      {:target target :schedule {:precision :f32-scalar}})]
        (try
          (let [primal ((:primal! prepared) xs)
                primal-lease (:lease primal)]
            (try
              (is (= (bits (primal-oracle xs-values))
                     (bits (download @captured primal-lease (:output-id primal) 3))))
              (is (= xs-before (vec xs)))
              (dotimes [index (alength xs)] (aset-float xs index (float 99.0)))
              (doseq [seed-values [[0.5 -0.25 2.0] [-1.0 1.0 0.125]]]
                (let [seed (float-array (map float seed-values))
                      seed-before (vec seed)
                      transpose ((:transpose! prepared) seed (:dependencies primal))]
                  (try
                    (is (= (bits (transpose-oracle xs-values seed-values))
                           (bits (download @captured (:lease transpose)
                                           (:output-id transpose) 3))))
                    (is (= seed-before (vec seed)))
                    (finally (.close ^java.io.Closeable (:lease transpose))))))
              (is (= :link-module-runtime-dependencies
                     (reason #((:transpose! prepared)
                               (float-array [1.0 1.0 1.0]) {}))))
              (finally (.close ^java.io.Closeable primal-lease)))
            (is (= :link-module-runtime-lease-released
                   (reason #((:transpose! prepared)
                             (float-array [1.0 1.0 1.0])
                             (:dependencies primal))))))
          (let [changed-values [1.0 2.0 -4.0]
                changed (float-array (map float changed-values))
                primal ((:primal! prepared) changed)]
            (try
              (is (= (bits (primal-oracle changed-values))
                     (bits (download @captured (:lease primal) (:output-id primal) 3))))
              (finally (.close ^java.io.Closeable (:lease primal)))))
          (finally ((:close! prepared))))))))

(deftest normalized-device-owner-runs-on-opencl
  (if @ocl-probe/opencl-available?
    (run-device-case! :ocl:0)
    (ocl-probe/opencl-skip! "normalized device owner acceptance")))

(deftest normalized-device-owner-runs-on-level-zero
  (if @ze-probe/gpu-available?
    (run-device-case! :ze:0)
    (ze-probe/gpu-skip! "normalized device owner acceptance")))
