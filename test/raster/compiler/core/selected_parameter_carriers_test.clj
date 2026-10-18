(ns raster.compiler.core.selected-parameter-carriers-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]))

(def body-calls (atom 0))

(deftm carrier-probe [values :- (Array double), count :- Long,
                     scale :- Double, reference :- Object] :- Double
  (do (swap! body-calls inc) scale))

(defn- selection []
  (let [generic #'carrier-probe
        method (first (get @(:raster.core/dispatch-table (meta generic)) 4))
        name (types/mangle 'carrier-probe (:tags method))]
    {:operation 'raster.compiler.core.selected-parameter-carriers-test/carrier-probe
     :implementation (symbol (str (:mangled-ns method)) (str name "-impl"))
     :entry method}))

(defn- owned-projection [operation implementation arity]
  (binding [entry/*emission-inputs* (atom {:static [] :values []})]
    (let [projection (entry/selected-parameter-carriers operation implementation arity)]
      {:projection projection :observations @entry/*emission-inputs*})))

(deftest selected-parameters-come-from-the-live-interface-not-logical-results
  (let [{:keys [operation implementation]} (selection)
        before @body-calls
        {:keys [projection observations]} (owned-projection operation implementation 4)
        definition (:definition projection)
        ^java.lang.reflect.Method method (:invocation-method projection)]
    (is (nil? (entry/selected-parameter-carriers operation implementation 4))
        "an optional proof has no authority without its definition owner")
    (is (= [(class (double-array 0)) Long/TYPE Double/TYPE Object]
           (:parameter-carriers projection)))
    (is (= '[values count scale reference] (:parameters projection)))
    (is (= Object (:result-carrier projection))
        "the real interface returns Object despite the logical Double result")
    (is (= 'double (:return-tag definition)))
    (is (= "invk" (.getName method)))
    (is (= (:parameter-carriers projection) (vec (.getParameterTypes method))))
    (is (= #{(:generic-var definition) (:receiver-var definition) (:source-var definition)}
           (set (keys (:definition-proofs observations)))))
    (is (entry/assert-definition-inputs-current! observations))
    (is (= before @body-calls) "projection never invokes the user's method")))

(deftest changed-receivers-sources-and-signatures-cannot-supply-carrier-evidence
  (let [{:keys [operation implementation]} (selection)
        {:keys [projection observations]} (owned-projection operation implementation 4)
        {:keys [receiver-var source-var]} (:definition projection)]
    (doseq [var [receiver-var source-var]]
      (with-redefs-fn {var (fn [& _] ::replacement)}
        #(do
           (is (nil? (:projection (owned-projection operation implementation 4))))
           (is (= :stale-definition-proof-input
                  (try (entry/assert-definition-inputs-current! observations) nil
                       (catch clojure.lang.ExceptionInfo failure
                         (:reason (ex-data failure)))))))))
    (doseq [[operation implementation arity]
            [[operation implementation 3]
             ['carrier-probe implementation 4]
             ['user.helpers/carrier-probe implementation 4]
             [operation 'user.helpers/carrier-probe-impl 4]]]
      (is (nil? (:projection (owned-projection operation implementation arity)))))))

(deftest interface-correspondence-is-checked-without-generating-a-replacement
  (let [{:keys [operation implementation entry]} (selection)
        interface-name (:typed-iface entry)
        interfaces @types/fn-interface-by-name
        interface (get interfaces interface-name)]
    (doseq [replacement [(dissoc interfaces interface-name)
                         (assoc-in interfaces [interface-name :iface-class] Object)
                         (assoc-in interfaces [interface-name :param-tags] ['Object])]]
      (with-redefs [types/fn-interface-by-name (atom replacement)]
        (is (nil? (:projection (owned-projection operation implementation 4))))))
    (is (identical? interface (get @types/fn-interface-by-name interface-name)))))

(deftest mid-projection-redefinition-cannot-attach-new-observations-to-old-evidence
  (let [{:keys [operation implementation]} (selection)
        definition (:definition (:projection (owned-projection operation implementation 4)))
        original-tag->iface-type types/tag->iface-type]
    (doseq [var [(:receiver-var definition) (:source-var definition)]]
      (let [original @var
            changed (atom false)]
        (try
          (with-redefs [types/tag->iface-type
                        (fn [tag]
                          (when (compare-and-set! changed false true)
                            (alter-var-root var (constantly (fn [& _] ::replacement))))
                          (original-tag->iface-type tag))]
            (is (= :stale-definition-proof-input
                   (try (owned-projection operation implementation 4) nil
                        (catch clojure.lang.ExceptionInfo failure
                          (:reason (ex-data failure)))))))
          (finally (alter-var-root var (constantly original))))))))
