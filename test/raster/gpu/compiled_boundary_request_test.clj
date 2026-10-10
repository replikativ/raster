(ns raster.gpu.compiled-boundary-request-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.compiled :as compiled]))

(defn- request [options]
  (#'compiled/boundary-request (:donate options) (:constants options)
                              (:outputs options) (:taps options) (:roles options)))

(def ^:private values
  {'x {:node :x-node :role :input :shape [4] :dtype :float}
   'out {:node :out-node :role :output :shape [4] :dtype :float}
   'result {:node :out-node :role :output :shape [4] :dtype :float}})

(defn- equation-projection [options semantic-outputs]
  (#'compiled/project-equation-first-boundary
   {:nodes (into {} (map (fn [[_ {:keys [node role shape dtype]}]]
                          [node {:role role :view {:shape shape :dtype dtype}}])) values)
    :instances [{:roles {}}]
    :outputs (mapv second semantic-outputs)
    :attributes {:public-buffer-bindings {'x :x-node 'out :out-node}
                 :public-buffer-roles {'x :input 'out :output}
                 :compiler-buffer-bindings {'x :x-node 'out :out-node 'result :out-node}
                 :semantic-outputs semantic-outputs}}
   {:semantic {:attributes {:invocation-plan {:parameters [{:symbol 'x} {:symbol 'out}]}}}}
   [:x-initial :out-initial] options))

(deftest caller-intent-retains-order-duplicates-and-role-precedence
  (let [options {:donate '[x x] :constants '[out out]
                 :outputs '(out out) :taps '[out x]
                 :roles {'x :input 'out :output}}
        r (request options)]
    (doseq [key [:donate :constants :outputs :taps :roles]]
      (is (identical? (get options key) (get r key))))
    (is (= #{'x} (:donate-set r)))
    (is (= {'x :input 'out :output} (:effective-roles r)))
    (is (= [{:key :x' :sym 'x :from :donated}
            {:key :x' :sym 'x :from :donated}] (vec (:donated-entries r))))
    (is (= [:out :out] (mapv :key (:output-entries r))))
    (is (= [:out :x] (mapv :key (:tap-entries r))))
    (is (= {:x :x'} (#'compiled/donated-output-keys r))))
  (testing "explicit roles do not override donation/constant conflict admission"
    (let [error (try (request {:donate '[x] :constants '[x] :roles {'x :input}})
                     (catch clojure.lang.ExceptionInfo e e))]
      (is (= #{'x} (:conflict (ex-data error)))))))

(deftest adapters-share-intent-without-erasing-their-result-contracts
  (let [options {:donate '[x x] :outputs '[out out] :taps '[out x]
                 :roles {'x :input}}
        r (request options)
        lowering {:certificate {:values (assoc-in values ['x :role] :input)}}
        resident-input (#'compiled/build-in-tree lowering
                        {:all-params '[x out] :array-params '[x out]}
                        [:x-initial :out-initial] r)
        resident-output (#'compiled/build-out-tree lowering r 'result)
        equation (equation-projection options [['result :out-node]])
        projection (:projection equation)]
    (is (= resident-input (:in-tree projection)))
    (is (= [:x' :x' :out :out :result :out :x] (mapv :key resident-output)))
    (is (= [:donated :donated :output :output :result :tap :tap]
           (mapv :from resident-output)))
    (is (= [:x' :out] (mapv :key (:out-tree projection)))
        "equation-first retains its physical-node deduplication")
    (is (= [:donated :output] (mapv :from (:out-tree projection))))
    (is (= (#'compiled/donated-output-keys r) (:donated projection)))
    (is (= [:x-node :out-node] (get-in equation [:plan :outputs])))
    (is (= :input (get-in equation [:plan :nodes :x-node :role])))))

(deftest void-and-default-boundaries-remain-adapter-owned
  (let [r (request {})
        lowering {:certificate {:values values}}]
    (is (empty? (:effective-roles r)))
    (is (empty? (#'compiled/build-out-tree lowering r nil)))
    (is (empty? (get-in (equation-projection {} []) [:projection :out-tree])))
    (is (= [:result] (mapv :key (#'compiled/build-out-tree lowering r 'result))))
    (is (= [:result] (mapv :key (get-in (equation-projection {} [['result :out-node]])
                                      [:projection :out-tree]))))))

(deftest equation-role-rejection-still-precedes-shared-conflict-admission
  (let [error (try (equation-projection {:donate '[missing] :constants '[missing]} [])
                   (catch clojure.lang.ExceptionInfo e e))]
    (is (= :compiled-equation-first-role-symbols (:reason (ex-data error))))
    (is (= #{'missing} (:symbols (ex-data error))))))
