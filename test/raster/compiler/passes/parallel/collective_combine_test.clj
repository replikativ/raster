(ns raster.compiler.passes.parallel.collective-combine-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.collective-combine :as combine]))

(defn- algebra [operator identity dtype]
  (scan/certify-reassociation
   {:acc 'acc :init identity :lambda (list operator 'acc 'element)} dtype))

(defn- target-options [dialect]
  (let [device (keyword (str (name dialect) ":analytic"))]
    {:target-device device :target-dialect dialect
     :target-descriptor {:device-id device :device-type :gpu
                         :backend (case dialect :cuda :cuda :hip :hip :ocl)
                         :subgroup-dialect (case dialect :opencl-intel :intel-opencl dialect)
                         :subgroup-size 32 :max-workgroup-size 256}}))

(deftest collective-combines-are-ordinary-typed-maps
  (doseq [dtype [:float :double]
          [operator identity] [['+ 0.0] ['* 1.0]]]
    (let [certificate (algebra operator identity dtype)
          algorithm (combine/algorithm certificate 17)
          equation (first (soac/equations algorithm))
          parts (soac/lambda-parts (:lambda (soac/operation-parts equation)))]
      (is (= '[left right] (:inputs (soac/facts algorithm))))
      (is (= '[result] (soac/outputs algorithm)))
      (is (= [17] (get-in (soac/facts algorithm) [:values 'result :shape])))
      (is (= [{:id 'combined :dtype dtype :init (list (:combine certificate) 'lhs 'rhs)}]
             (:locals parts)))
      (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
        (let [emitted (combine/emit certificate 17 (target-options dialect))
              kernel (get-in emitted [:graph :nodes 0 :operation])]
          (is (= emitted (combine/validate! certificate 17 emitted)))
          (is (= :kernel-body (artifact/emission-route kernel)))
          (is (= #{'left 'right 'result 17} (set (:arguments kernel)))))))))

(deftest collective-combine-rejects-an-unrelated-valid-algorithm
  (let [sum (algebra '+ 0.0 :float)
        product (algebra '* 1.0 :float)
        emitted (combine/emit sum 17 (target-options :opencl-portable))]
    (doseq [[certificate elements] [[product 17] [sum 18]]]
      (is (= :collective-combine-algorithm
             (:reason (ex-data (try (combine/validate! certificate elements emitted)
                                   (catch clojure.lang.ExceptionInfo e e)))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (combine/algorithm (assoc sum :combine '*) 17)))
    (let [product-emitted (combine/emit product 17 (target-options :opencl-portable))
          claimed-sum (-> product-emitted
                          (assoc :algorithm (:algorithm emitted))
                          (assoc-in [:body :equations 0 :algorithm] (:algorithm emitted)))]
      (is (= :collective-combine-schedule
             (:reason (ex-data (try (combine/validate! sum 17 claimed-sum)
                                   (catch clojure.lang.ExceptionInfo e e)))))
          "a valid different scalar schedule cannot inherit the requested monoid label"))))

(deftest collective-combine-rejects-inconsistent-target-snapshots
  (let [algebra (algebra '+ 0.0 :float)
        options (target-options :cuda)]
    (doseq [bad [{} (assoc options :target-device :cpu:0)
                 (assoc options :target-dialect :hip)
                 (assoc-in options [:target-descriptor :device-type] :cpu)]]
      (is (= :collective-combine-target
             (:reason (ex-data (try (combine/emit algebra 17 bad)
                                   (catch clojure.lang.ExceptionInfo e e)))))))
    (let [emitted (combine/emit algebra 17 options)]
      (is (= :collective-combine-target
             (:reason (ex-data
                       (try (combine/validate!
                             algebra 17 (assoc-in emitted [:attributes :collective-target]
                                                  (target-options :hip)))
                            (catch clojure.lang.ExceptionInfo e e)))))))))

(deftest collective-combine-rejects-invalid-extents
  (doseq [elements [0 -1 1.5 (inc (bigint Long/MAX_VALUE))]]
    (is (= :collective-combine-extent
           (:reason (ex-data (try (combine/algorithm (algebra '+ 0.0 :float) elements)
                                 (catch clojure.lang.ExceptionInfo e e))))))))

(defn- local-request [device shape]
  {:id :local-combine
   :nodes (into {} (map (fn [role]
                         [role (link/node {:id [role :storage] :device device :dtype :float
                                           :shape shape})])) [:left :right :result])})

(deftest generated-combines-bind-to-the-ordinary-link-plan-boundary
  (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]
          shape [[17] [2 3]]]
    (let [options (target-options dialect)
          request (local-request (:target-device options) shape)
          elements (reduce * shape)
          cert (algebra '+ 0.0 :float)
          linked (combine/bind-local cert elements options request)
          plan (:link-plan linked)]
      (is (= linked (combine/validate-local! cert elements request linked)))
      (is (= :graph-link-instance
             (when (link/graph-link-instance? (first (:instances plan))) :graph-link-instance)))
      (is (= [:result :storage] (first (link/output-value-ids plan))))
      (is (= shape (get-in plan [:values [:result :storage] :abstract :shape])))
      (is (= #{[:left :storage] [:right :storage]} (:requires (link/initialization-contract plan))))
      (is (= #{[:result :storage]} (:produces (link/initialization-contract plan)))))))

(deftest local-binding-rejects-storage-and-interface-forgeries
  (let [options (target-options :opencl-portable)
        cert (algebra '+ 0.0 :float)
        request (local-request (:target-device options) [17])
        linked (combine/bind-local cert 17 options request)]
    (doseq [bad [(assoc request :unknown true)
                 (assoc-in request [:nodes :result] (get-in request [:nodes :left]))
                 (assoc-in request [:nodes :result :source] (float-array 17))
                 (assoc-in request [:nodes :left]
                           (link/node {:id :wrong-width :device (:target-device options)
                                       :dtype :float :shape [18]}))]]
      (is (thrown? clojure.lang.ExceptionInfo (combine/bind-local cert 17 options bad))))
    (let [plan (:link-plan linked)
          swapped (assoc-in plan [:instances 0 :bindings]
                            {'left [:right :storage] 'right [:left :storage]
                             'result [:result :storage]})]
      (is (= swapped (link/validate! swapped)) "generic memory validation cannot know semantic roles")
      (is (= :collective-combine-local-plan
             (:reason (ex-data (try (combine/validate-local! cert 17 request
                                                             (assoc linked :link-plan swapped))
                                   (catch clojure.lang.ExceptionInfo e e)))))))))
