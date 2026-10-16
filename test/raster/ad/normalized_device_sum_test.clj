(ns raster.ad.normalized-device-sum-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-device :as device]
            [raster.ad.normalized-device-test :as maps]
            [raster.ad.normalized-map-sum-test]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-segmented-reduction-test :as products]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.link-module :as link-module]))

(defn- reason [invoke]
  (try (invoke) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- private-fn [name]
  (let [v (ns-resolve 'raster.ad.normalized-map-sum-test name)]
    (when-not (var? v) (throw (ex-info "missing map-sum fixture helper" {:name name})))
    v))

(defn- owner []
  (canonical/adapt-map-sum
   ((private-fn 'build) ((private-fn 'source) :float) 'xs)))

(defn- multi-active-owner []
  (canonical/adapt-map-sum
   (normalized/linearize-map-sum-program
    ((private-fn 'source) :float) (the-ns 'raster.ad.normalized-map-sum-test)
    {:active-inputs '[weights xs]})))

(defn- reordered-owner []
  (let [source ((private-fn 'source) :float)
        reordered (soac/make (assoc (soac/facts source) :inputs '[weights xs size])
                             (soac/equations source) (soac/outputs source))]
    (canonical/adapt-map-sum
     ((private-fn 'build) reordered 'xs))))

(defn- device-private-fn [name]
  (let [v (ns-resolve 'raster.ad.normalized-device name)]
    (when-not (var? v) (throw (ex-info "missing normalized-device helper" {:name name})))
    v))

(defn- zero-residual-owner []
  (let [source ((private-fn 'source) :float)
        equations (vec (soac/equations source))
        map-equation
        (list '= 'map-0 '[mapped]
              (list 'map {:index 'i :extent 'size} '[xs weights] []
                    (soac/lambda-form '[x w] '[x])))
        source (soac/make (soac/facts source) (assoc equations 0 map-equation)
                          (soac/outputs source))]
    (canonical/adapt-map-sum ((private-fn 'build) source 'xs))))

(defn- arguments [owner xs weights]
  (mapv (fn [{:keys [id]}]
          (case id xs xs weights weights size (long (alength ^floats xs))))
        (get-in owner [:interface :inputs])))

(defn- bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- fmul [left right] (float (* (float left) (float right))))
(defn- fadd [left right] (float (+ (float left) (float right))))

(defn- loss-oracle [xs weights]
  (reduce (fn [acc [x weight]]
            (fadd acc (fmul (fmul x x) weight)))
          (float 0.0) (map vector xs weights)))

(defn- gradient-oracle [xs weights seed]
  (mapv (fn [x weight]
          (fmul seed (fadd (fmul x weight) (fmul x weight))))
        xs weights))

(defn- synthetic-descriptor [target] {:device-id target})

(defn- producer-loss-source [producer dtype]
  (let [result (first (soac/outputs producer))
        extent (first (get-in (soac/facts producer) [:values result :shape]))
        init (if (= :float dtype) (float 0.0) 0.0)
        body '(raster.numeric/+ total element)
        reduction (list '= 'ordered-loss '[loss]
                        (list 'reduce {:index 'p :extent extent :accumulators '[total]
                                       :identities [init] :dtypes [dtype]
                                       :algebra [(scan/certify-reassociation
                                                  {:acc 'total :init init :lambda body} dtype)]}
                              [result] [] (soac/lambda-form '[total element] [body])))
        ;; Construct an ordinary pristine scalar source. The separately tested host-owned
        ;; scalar fixture must still refuse resident promotion; no ownership is erased here.
        facts (-> (soac/facts producer)
                  (assoc-in [:values 'loss] (av/tensor {:dtype dtype :shape []}))
                  (assoc-in [:equations 'ordered-loss] (soac/default-equation-facts)))]
    (soac/make facts (conj (vec (soac/equations producer)) reduction) '[loss])))

(defn- contract-loss-source [dtype abi]
  (producer-loss-source (#'products/with-contract-intent (#'products/activated-source dtype abi)) dtype))

(defn- map-spine-loss-source []
  (producer-loss-source (:source (#'maps/three-boundary-owner 3 true)) :float))

(defn- checked-producer [source active]
  (canonical/adapt (normalized/linearize-equation-program source *ns* {:active-input active})))

(defn- contract-values [dtype changed?]
  (let [array (if (= :float dtype) float-array double-array)]
    {'A (array (if changed? [0.5 1.0 -1.5 2.0 -0.25 0.75] [1.0 -2.0 0.5 0.25 1.5 -0.75]))
     'W (array (if changed? [-0.5 0.25 1.0 -1.5 2.0 0.5 -0.25 0.75 0.125 -1.0 0.5 1.5]
                   [0.5 -1.0 0.25 2.0 -0.5 1.5 0.75 -0.25 1.0 0.5 -1.5 0.125]))}))

(deftest marked-producer-loss-binds-through-the-common-module-without-device-contact
  (doseq [dtype [:float :double] active '[A W] abi ['[A W] '[W A]]]
    (let [owner (checked-producer (contract-loss-source dtype abi) active)
          values (contract-values dtype false)
          module (atom nil) closed (atom 0)]
      (with-redefs [hardware/descriptor-for synthetic-descriptor
                    link-module/instantiate!
                    (fn [plan _] (reset! module plan)
                      (reify java.io.Closeable (close [_] (swap! closed inc))))]
        (let [prepared (device/prepare-sum-owner owner (mapv values abi)
                                                (if (= dtype :float) (float -0.5) -0.5)
                                                {:target :ocl:producer-loss
                                                 :schedule {:precision :f32-scalar}})]
          (try
            (is (= abi (mapv :id (get-in prepared [:ports :inputs]))))
            (is (= 2 (count (get-in @module [:spine :instances]))))
            (is (= [:primal :transpose] (mapv :name (:entries @module))))
            (is (seq (get-in prepared [:ports :residuals])))
            (finally ((:close! prepared)))))
        (is (= 1 @closed))))))

(deftest producer-loss-preserves-intent-and-host-ownership-refusals
  (let [values (contract-values :double false)
        unmarked (producer-loss-source (#'products/activated-source :double '[W A]) :double)
        host-owned (#'products/with-contract-intent (#'products/summed-activation-source :double '[W A]))
        dynamic (producer-loss-source (:source (#'maps/three-boundary-owner 'n true)) :float)
        compile equation-first/compile-typed
        compilations (atom 0) instantiations (atom 0)]
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  equation-first/compile-typed (fn [& args] (swap! compilations inc) (apply compile args))
                  link-module/instantiate! (fn [& _] (swap! instantiations inc))]
      (is (= :normalized-device-interface
             (reason #(device/prepare-sum-owner (checked-producer unmarked 'A) (mapv values '[W A]) 1.0
                                               {:target :ocl:producer-loss}))))
      (is (= :normalized-device-interface
             (reason #(device/prepare-sum-owner (checked-producer dynamic 'x)
                                               [3 (float-array 3) (float-array 3) (float-array 3)]
                                               (float 1.0) {:target :ocl:producer-loss}))))
      (is (zero? @compilations))
      (is (= :typed-soac-resident-output-demand
             (reason #(device/prepare-sum-owner (checked-producer host-owned 'A) (mapv values '[W A]) 1.0
                                               {:target :ocl:producer-loss}))))
      (is (zero? @instantiations)))))

(deftest repeated-input-map-spine-loss-binds-the-same-module-path
  (let [owner (checked-producer (map-spine-loss-source) 'x)
        values {'x (float-array [1.0 -2.0 0.5]) 'q (float-array [0.25 1.0 -0.5])
                'r (float-array [2.0 -0.5 1.0])}
        module (atom nil)]
    (is (= '[map map map reduce] (mapv soac/operation-kind (soac/equations (:source owner)))))
    (is (= 3 (count (get-in owner [:derivation :bundle :map-programs :proof :nodes]))))
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [plan _] (reset! module plan)
                                             (reify java.io.Closeable (close [_])))]
      (let [prepared (device/prepare-sum-owner owner
                                              (mapv #(get values (:id %)) (get-in owner [:interface :inputs]))
                                              (float -0.5) {:target :ocl:producer-loss
                                                            :schedule {:precision :f32-scalar}})]
        (try
          (is (= '[r q x] (mapv :id (get-in prepared [:ports :inputs]))))
          (is (= 2 (count (get-in @module [:spine :instances]))))
          (finally ((:close! prepared))))))))

(defn- download [runtime lease value-id elements]
  (let [payload ((ns-resolve 'raster.gpu.link-module 'opaque-payload) runtime)
        resident (get-in @lease [:outputs value-id])
        output (float-array elements)]
    ;; The public API intentionally returns resident leases. This remains a test-only projection.
    (gpu/download-range! (:session payload) resident output {:elements elements})
    output))

(defn- contract-loss-oracle [active values seed]
  (let [a (vec (get values 'A)) w (vec (get values 'W))
        product (#'products/float-product 2 3 4 a w)
        cotangent (mapv #(fadd (fmul seed %) (fmul seed %)) product)]
    {:losses [(reduce fadd (float 0.0) (map #(fmul % %) product))]
     :gradient (#'products/float-pull active 2 3 4 a w cotangent)}))

(defn- map-spine-values [changed?]
  {'x (float-array (if changed? [4097.0 0.33333334 -4096.0] [1.0 -2.0 0.5]))
   'q (float-array (if changed? [-0.125 3.1415927 0.0009765625] [0.25 1.0 -0.5]))
   'r (float-array (if changed? [0.33333334 -2.5 2.7182817] [2.0 -0.5 1.0]))})

(defn- map-spine-oracle [values seed]
  (let [xs (vec (get values 'x)) qs (vec (get values 'q)) rs (vec (get values 'r))]
    {:losses
     ;; This source explicitly carries a revalidated commutative-monoid certificate. Its
     ;; terminal sum may regroup/interleave, unlike each map/reverse scalar checkpoint.
     ;; Enumerate ALL binary trees and permutations of this finite three-term fixture,
     ;; independently of workgroup width or emitted schedule, with no numerical tolerance.
     (let [terms (mapv #(fmul (fmul (float (- %1 %2)) %1) %3) xs qs rs)]
       (vec (set (mapcat (fn [[a b c]] [(fadd (fadd a b) c) (fadd a (fadd b c))])
                         (for [i (range 3) j (range 3) k (range 3)
                               :when (= 3 (count (distinct [i j k])))]
                           [(nth terms i) (nth terms j) (nth terms k)])))))
     :gradient (mapv (fn [x q r]
                       (let [db (fmul seed r)]
                         (fadd (fmul db (float (- x q))) (fmul db x)))) xs qs rs)}))

(deftest repeated-map-oracle-distinguishes-reassociated-backward-arithmetic
  (let [values (map-spine-values true) seed (float -0.5)
        reassociated (mapv (fn [x q r] (fmul seed (fmul (fadd (float (- x q)) x) r)))
                            (get values 'x) (get values 'q) (get values 'r))]
    (is (not= (bits reassociated) (bits (:gradient (map-spine-oracle values seed)))))
    (is (= #{51200520.0 51200524.0} (set (map double (:losses (map-spine-oracle values seed))))))))

(defn- run-resident-producer! [target source active values-for oracle]
  (let [owner (checked-producer source active)
        initial (values-for false)
        seed (float -0.5)
        captured (atom nil)
        instantiate link-module/instantiate!]
    (with-redefs [link-module/instantiate!
                  (fn [module options]
                    (let [runtime (instantiate module options)] (reset! captured runtime) runtime))]
      (let [prepared (device/prepare-sum-owner
                      owner (mapv #(get initial (:id %)) (get-in owner [:interface :inputs])) seed
                      {:target target :schedule {:precision :f32-scalar}})]
        (try
          (doseq [changed? [false true]]
            (let [values (if changed? (values-for true) initial)
                  before (update-vals values vec)
                  expected (oracle values seed)
                  primal ((:primal! prepared) values)]
              (try
                ;; Download waits for the primal before mutating its host input buffers.
                (is (contains? (set (map #(bits [%]) (:losses expected)))
                               (bits (download @captured (:lease primal) (:output-id primal) 1))))
                (is (= before (update-vals values vec)))
                (doseq [buffer (vals values)] (java.util.Arrays/fill ^floats buffer (float 99.0)))
                (let [transpose ((:transpose! prepared) (:dependencies primal))]
                  (try
                    (is (= (bits (:gradient expected))
                           (bits (download @captured (:lease transpose) (:output-id transpose)
                                           (count (:gradient expected))))))
                    (finally (.close ^java.io.Closeable (:lease transpose)))))
                (is (= :link-module-runtime-lease-active
                       (reason #((:primal! prepared) (values-for changed?)))))
                (finally (.close ^java.io.Closeable (:lease primal))))
              (is (= :link-module-runtime-lease-released
                     (reason #((:transpose! prepared) (:dependencies primal)))))))
          (finally ((:close! prepared))))))))

(defn- run-resident-producers! [target]
  (doseq [active '[A W]]
    (run-resident-producer! target (contract-loss-source :float '[W A]) active
                            #(contract-values :float %) #(contract-loss-oracle active %1 %2)))
  (run-resident-producer! target (map-spine-loss-source) 'x map-spine-values map-spine-oracle))

(deftest resident-producer-loss-runs-on-opencl
  (if @ocl-probe/opencl-available?
    (run-resident-producers! :ocl:0)
    (ocl-probe/opencl-skip! "canonical producer ordered-loss device composition")))

(deftest resident-producer-loss-runs-on-level-zero
  (if @ze-probe/gpu-available?
    (run-resident-producers! :ze:0)
    (ze-probe/gpu-skip! "canonical producer ordered-loss device composition")))

(deftest public-array-boundary-is-an-exact-id-bijection
  (let [boundary (device-private-fn 'public-array-boundary!)
        descriptors [{:id 'xs} {:id 'weights}]
        layout {:public-inputs [{:symbol 'weights :value :weight-value}
                                {:symbol 'xs :value :xs-value}]
                :buffer-bindings {'xs :xs-value 'weights :weight-value}}]
    (is (= {:ids '[xs weights] :values [:xs-value :weight-value]}
           (boundary descriptors layout)))
    (doseq [[declared malformed]
            [[descriptors (assoc layout :public-inputs [{:symbol 'xs :value :xs-value}])]
             [[{:id 'xs} {:id 'xs}] layout]
             [descriptors (assoc-in layout [:public-inputs 0 :value] :wrong-value)]]]
      (is (= :normalized-device-array-boundary
             (reason #(boundary declared malformed)))))))

(deftest reordered-map-sum-source-binds-public-arrays-by-id
  (let [owner (reordered-owner)
        selected ((device-private-fn 'sum-interface!) owner)
        xs (float-array [1.0 2.0 3.0])
        weights (float-array [4.0 5.0 6.0])
        compilations (atom [])
        calls (atom [])
        compile equation-first/compile-typed
        runtime (reify java.io.Closeable (close [_]))]
    (is (identical? (first (get-in owner [:interface :tangent-inputs])) (:tangent selected)))
    (is (identical? (get-in owner [:interface :cotangent-input]) (:seed selected)))
    (is (identical? (first (get-in owner [:interface :cotangent-results])) (:gradient selected)))
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  equation-first/compile-typed
                  (fn [& args]
                    (let [compiled (apply compile args)]
                      (swap! compilations conj compiled)
                      compiled))
                  link-module/instantiate! (fn [_ _] runtime)
                  link-module/run-and-lease!
                  (fn [_ entry request]
                    (swap! calls conj [entry request])
                    (reify java.io.Closeable (close [_])))]
      (let [prepared (device/prepare-sum-owner
                      owner (arguments owner xs weights) (float 1.0)
                      {:target :ocl:normalized-sum :schedule {:precision :f32-scalar}})
            ports (:ports prepared)
            canonical-ids (mapv :id (filterv #(seq (get-in % [:value :shape]))
                                             (get-in owner [:interface :inputs])))
            primal-parameters
            (get-in (first @compilations) [:semantic :attributes :invocation-plan :parameters])
            compiled-array-ids
            (mapv :symbol (filterv #(seq (get-in % [:value :shape])) primal-parameters))
            primal ((:primal! prepared) {'xs xs 'weights weights})
            values-by-id (into {} (map (juxt :id :value)) (:inputs ports))
            expected {(get values-by-id 'xs) xs (get values-by-id 'weights) weights}]
        (is (= '[weights xs] canonical-ids))
        (is (not= canonical-ids compiled-array-ids))
        (is (= canonical-ids (mapv :id (:inputs ports))))
        (is (= [:primal {:inputs expected}] (first @calls)))
        (.close ^java.io.Closeable (:lease primal))
        ((:close! prepared))))))

(deftest fixed-seed-map-sum-uses-exact-array-and-scalar-boundaries
  (let [owner (owner)
        xs (float-array [1.5 -2.0 0.5])
        weights (float-array [2.0 0.5 -4.0])
        captured-module (atom nil)
        calls (atom [])
        runtime (reify java.io.Closeable (close [_]))
        leases (atom [])]
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate!
                  (fn [module _] (reset! captured-module module) runtime)
                  link-module/run-and-lease!
                  (fn [_ entry request]
                    (swap! calls conj [entry request])
                    (let [lease (reify java.io.Closeable (close [_]))]
                      (swap! leases conj lease)
                      lease))]
      (let [prepared (device/prepare-sum-owner
                      owner (arguments owner xs weights) (float -0.5)
                      {:target :ocl:normalized-sum :schedule {:precision :f32-scalar}})
            ports (:ports prepared)
            entries (into {} (map (juxt :name identity)) (:entries @captured-module))
            instances (into {} (map (juxt :id identity)) (get-in @captured-module [:spine :instances]))
            transpose-instance (get instances (first (get-in entries [:transpose :instances])))
            seed-id (get-in ports [:fixed-seed :id])
            seed-value (get-in transpose-instance [:call :scalar-values seed-id])
            input-ids (mapv :id (:inputs ports))]
        (is (= '[xs weights] input-ids))
        (is (= [:raster.ad.normalized-device/primal
                :raster.ad.normalized-device/transpose]
               (mapv :id (get-in @captured-module [:spine :instances]))))
        (is (= (mapv :value (:inputs ports)) (get-in entries [:primal :interface :inputs])))
        (is (= [] (get-in entries [:transpose :interface :inputs])))
        (is (= {:type :float :value (float -0.5)} seed-value
               (get-in ports [:fixed-seed :value])))
        (is (= :resident-scalar-buffer
               (get-in @captured-module
                       [:spine :values (:primal-output ports) :abstract :representation :kind])))
        (is (every? (comp nil? :source val) (get-in @captured-module [:spine :nodes])))
        (let [primal ((:primal! prepared) {'weights weights 'xs xs})
              transpose ((:transpose! prepared) (:dependencies primal))
              expected-inputs
              (into {} (map (fn [{:keys [id value]}]
                              [value (get {'xs xs 'weights weights} id)]))
                    (:inputs ports))]
          (is (= [:primal {:inputs expected-inputs}]
                 (first @calls)))
          (is (= [:transpose {:dependencies (:dependencies primal)}] (second @calls)))
          (.close ^java.io.Closeable (:lease transpose))
          (.close ^java.io.Closeable (:lease primal)))
        (is (= :normalized-device-inputs
               (reason #((:primal! prepared) {'xs xs}))))
        ((:close! prepared))))
    (is (= :invocation-materialization-scalar
           (reason #(with-redefs [hardware/descriptor-for synthetic-descriptor]
                      (device/prepare-sum-owner
                       owner (arguments owner xs weights)
                       {:type :double :value -0.5}
                       {:target :ocl:normalized-sum
                        :schedule {:precision :f32-scalar}})))))))

(deftest zero-residual-sum-forwards-the-empty-dependency-contract
  (let [owner (zero-residual-owner)
        xs (float-array [1.0 2.0])
        weights (float-array [3.0 4.0])
        calls (atom [])
        runtime (reify java.io.Closeable (close [_]))]
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [_ _] runtime)
                  link-module/run-and-lease!
                  (fn [_ entry request]
                    (swap! calls conj [entry request])
                    (when (and (= :transpose entry) (seq (:dependencies request)))
                      (throw (ex-info "entry dependencies differ from its exact frontiers"
                                      {:reason :link-module-runtime-dependencies})))
                    (reify java.io.Closeable (close [_])))]
      (let [prepared (device/prepare-sum-owner
                      owner (arguments owner xs weights) (float 2.0)
                      {:target :ocl:normalized-sum :schedule {:precision :f32-scalar}})
            primal ((:primal! prepared) {'xs xs 'weights weights})]
        (is (= {} (:dependencies primal)))
        (let [transpose ((:transpose! prepared) {})]
          (is (= [:transpose {:dependencies {}}] (last @calls)))
          (.close ^java.io.Closeable (:lease transpose)))
        (let [extra {:not-a-frontier (:lease primal)}
              refused (reason #((:transpose! prepared) extra))]
          (is (= :link-module-runtime-dependencies refused))
          (is (= [:transpose {:dependencies extra}] (last @calls)))
          (is (= 3 (count @calls))))
        (.close ^java.io.Closeable (:lease primal))
        ((:close! prepared)))))
  (is (= :normalized-device-family
         (reason #(device/prepare-owner (owner) [] {:target :ocl:normalized-sum})))))

(deftest stale-map-sum-authority-refuses-before-compilation
  (let [owner (owner)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
        current (revisions)
        compilations (atom 0)
        compile equation-first/compile-typed]
    (with-redefs-fn {revisions (constantly (update current :compiler-revision inc))
                     #'equation-first/compile-typed
                     (fn [& args] (swap! compilations inc) (apply compile args))}
      (fn []
        (is (= :ad-normalized-stale-rules
               (reason #(device/prepare-sum-owner
                         owner (arguments owner (float-array 0) (float-array 0))
                         (float 1.0) {:target :ocl:normalized-sum}))))
        (is (zero? @compilations))))))

(deftest multiple-active-map-sum-refuses-before-device-compilation-or-instantiation
  (let [owner (multi-active-owner)
        xs (float-array [1.0])
        weights (float-array [2.0])
        compilations (atom 0)
        instantiations (atom 0)]
    (with-redefs [equation-first/compile-typed
                  (fn [& _] (swap! compilations inc) (throw (Exception. "compile observed")))
                  link-module/instantiate!
                  (fn [& _] (swap! instantiations inc) (throw (Exception. "instantiate observed")))]
      (is (= :normalized-device-multi-active
             (reason #(device/prepare-sum-owner
                       owner (arguments owner xs weights) (float 1.0)
                       {:target :ocl:normalized-sum})))))
    (is (zero? @compilations))
    (is (zero? @instantiations))))

(defn- run-device-case! [target]
  (let [owner (owner)
        initial-xs-values [1.5 -2.0 0.5]
        initial-weight-values [2.0 0.5 -4.0]
        changed-xs-values [2.0 1.0 -1.0]
        changed-weight-values [0.25 -2.0 0.5]
        seed (float -0.5)
        initial-xs (float-array initial-xs-values)
        initial-weights (float-array initial-weight-values)
        initial-xs-snapshot (aclone initial-xs)
        initial-weights-snapshot (aclone initial-weights)
        captured (atom nil)
        instantiate link-module/instantiate!]
    (with-redefs [link-module/instantiate!
                  (fn [module options]
                    (let [runtime (instantiate module options)]
                      (reset! captured runtime)
                      runtime))]
      (let [prepared (device/prepare-sum-owner
                      owner (arguments owner initial-xs initial-weights) seed
                      {:target target :schedule {:precision :f32-scalar}})]
        (try
          (let [primal ((:primal! prepared) {'weights initial-weights 'xs initial-xs})]
            (try
              (is (= (bits initial-xs-snapshot) (bits initial-xs)))
              (is (= (bits initial-weights-snapshot) (bits initial-weights)))
              (is (= (bits [(loss-oracle initial-xs-values initial-weight-values)])
                     (bits (download @captured (:lease primal) (:output-id primal) 1))))
              (dotimes [index 3]
                (aset-float initial-xs index (float 99.0))
                (aset-float initial-weights index (float -99.0)))
              (let [transpose ((:transpose! prepared) (:dependencies primal))]
                (try
                  (is (= (bits (gradient-oracle initial-xs-values initial-weight-values seed))
                         (bits (download @captured (:lease transpose)
                                         (:output-id transpose) 3))))
                  (finally (.close ^java.io.Closeable (:lease transpose)))))
              (is (= :link-module-runtime-lease-active
                     (reason #((:primal! prepared)
                               {'xs (float-array changed-xs-values)
                                'weights (float-array changed-weight-values)}))))
              (finally
                (.close ^java.io.Closeable (:lease primal))))
            (is (= :link-module-runtime-lease-released
                   (reason #((:transpose! prepared) (:dependencies primal))))))
          (let [changed-xs (float-array changed-xs-values)
                changed-weights (float-array changed-weight-values)
                primal ((:primal! prepared) {'xs changed-xs 'weights changed-weights})]
            (try
              (is (= (bits [(loss-oracle changed-xs-values changed-weight-values)])
                     (bits (download @captured (:lease primal) (:output-id primal) 1))))
              (let [transpose ((:transpose! prepared) (:dependencies primal))]
                (try
                  (is (= (bits (gradient-oracle changed-xs-values changed-weight-values seed))
                         (bits (download @captured (:lease transpose)
                                         (:output-id transpose) 3))))
                  (finally (.close ^java.io.Closeable (:lease transpose)))))
              (is (= changed-xs-values (vec changed-xs)))
              (is (= changed-weight-values (vec changed-weights)))
              (finally
                (.close ^java.io.Closeable (:lease primal)))))
          (finally
            ((:close! prepared))))))))

(deftest fixed-seed-map-sum-runs-on-opencl
  (if @ocl-probe/opencl-available?
    (run-device-case! :ocl:0)
    (ocl-probe/opencl-skip! "normalized fixed-seed map-sum device owner")))

(deftest fixed-seed-map-sum-runs-on-level-zero
  (if @ze-probe/gpu-available?
    (run-device-case! :ze:0)
    (ze-probe/gpu-skip! "normalized fixed-seed map-sum device owner")))
