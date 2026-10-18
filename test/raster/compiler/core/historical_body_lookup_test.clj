(ns raster.compiler.core.historical-body-lookup-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.numeric :as n]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- remap-float! [source-ns core?]
  (ns-unmap source-ns 'float)
  (if core?
    (binding [*ns* source-ns] (refer 'clojure.core :only '[float]))
    (intern source-ns 'float
            (fn [_] (throw (ArithmeticException. "historical user float"))))))

(defn- with-scalar [core? inspect!]
  (let [source-ns (create-ns (symbol (str "raster.internal.historical-float-" (gensym))))
        generic #'n/* original-generic (meta generic)]
    (try
      (binding [*ns* source-ns]
        (refer 'clojure.core :exclude '[float *])
        (remap-float! source-ns core?)
        (eval '(raster.core/deftm * [x :- Float y :- Float] :- Float
                 (float (clojure.core/* x y)))))
      (let [fixture (ns-resolve source-ns '*)
            method (first (get (core/methods-of fixture) 2))
            name (types/mangle '* (:tags method))
            receiver (ns-resolve source-ns (symbol (str name "-impl")))
            expression (with-meta (list '.invk (symbol (str (ns-name source-ns))
                                                      (str name "-impl")) 'x 'y)
                         {:raster.op/original 'raster.numeric/*})
            proof #(entry/call-with-definition-inputs
                    (fn [] (scalar/checked-selected-primitive-evaluation
                            expression [{:carrier Float/TYPE :presence :present}
                                        {:carrier Float/TYPE :presence :present}])))
            run #(clojure.lang.Reflector/invokeInstanceMethod
                  @receiver "invk" (object-array [(float 2.0) (float 3.0)]))]
        (alter-meta! generic assoc :raster.core/dispatch-table
                     (:raster.core/dispatch-table (meta fixture)))
        (inspect! {:source-ns source-ns :receiver receiver :expression expression
                   :proof proof :run run}))
      (finally
        (reset-meta! generic original-generic)
        (remove-ns (ns-name source-ns))))))

(deftest warm-user-to-core-remapping-cannot-create-a-completion-proof
  (with-scalar false
    (fn [{:keys [source-ns receiver proof run]}]
      (is (nil? (proof)))
      (core/ensure-bytecode-implementation! receiver)
      (is (thrown-with-msg? ArithmeticException #"historical user float" (run)))
      (is (nil? (proof)))
      (remap-float! source-ns true)
      (is (thrown-with-msg? ArithmeticException #"historical user float" (run)))
      (is (nil? (proof))))))

(deftest warm-core-to-user-remapping-does-not-reinterpret-the-old-body
  (with-scalar true
    (fn [{:keys [source-ns receiver proof run]}]
      (is (some? (proof)))
      (core/ensure-bytecode-implementation! receiver)
      (is (some? (proof)))
      (is (= (float 6.0) (run)))
      (remap-float! source-ns false)
      (is (= (float 6.0) (run)))
      (is (nil? (proof))))))

(deftest a-pending-source-query-retains-its-actual-lookup
  (with-scalar false
    (fn [{:keys [source-ns expression proof run]}]
      (is (nil? (proof)))
      (remap-float! source-ns true)
      (let [owner (atom {:static [] :values []})]
        (binding [entry/*emission-inputs* owner]
          (is (some? (scalar/checked-selected-primitive-evaluation
                      expression [{:carrier Float/TYPE :presence :present}
                                  {:carrier Float/TYPE :presence :present}]))))
        (is (contains? (get-in @owner [:definition-proofs #'clojure.core/float :lookups])
                       [source-ns 'float]))
        (remap-float! source-ns false)
        (is (= :stale-definition-proof-input
               (try (entry/assert-definition-inputs-current! @owner)
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (remap-float! source-ns true)
      (is (= (float 6.0) (run))))))

(deftest legacy-emissions-without-a-bare-lookup-edge-abstain
  (with-scalar true
    (fn [{:keys [receiver proof run]}]
      (core/ensure-bytecode-implementation! receiver)
      (let [original (meta receiver)]
        (try
          (is (some? (proof)))
          (alter-meta! receiver assoc-in [:raster.core/bytecode-emission :dependency-snapshots] [])
          (is (= (float 6.0) (run)))
          (is (nil? (proof)))
          (finally (reset-meta! receiver original)))))))

(deftest lexical-callees-mask-lookups-and-each-alias-edge-is-retained
  (let [source-ns (create-ns (symbol (str "raster.internal.historical-alias-" (gensym))))
        owner (atom {:static [] :values []})
        context {:source-ns source-ns :pending-source? true}]
    (try
      (binding [*ns* source-ns]
        (refer 'clojure.core :only '[float] :rename '{float left})
        (refer 'clojure.core :only '[float] :rename '{float right}))
      (intern source-ns '.invk (fn [& _] nil))
      (binding [entry/*emission-inputs* owner]
        (is (nil? (entry/observed-body-operation context '.invk #{})))
        (is (nil? (entry/observed-body-operation context 'if #{})))
        (is (nil? (entry/observed-body-operation context 'left #{'left})))
        (is (empty? (:definition-proofs @owner)))
        (is (= 'clojure.core/float (entry/observed-body-operation context 'left #{})))
        (is (= 'clojure.core/float (entry/observed-body-operation context 'right #{}))))
      (is (= #{[source-ns 'left] [source-ns 'right]}
             (get-in @owner [:definition-proofs #'clojure.core/float :lookups])))
      (let [historical (assoc context :pending-source? false
                             :dependency-snapshots (:values @owner))
            consumer (atom {:static [] :values []})]
        (binding [entry/*emission-inputs* consumer]
          (is (= 'clojure.core/float (entry/observed-body-operation historical 'right #{}))))
        (is (contains? (:definition-proofs @consumer) #'clojure.core/float))
        (ns-unmap source-ns 'right)
        (intern source-ns 'right (fn [_] nil))
        (is (= :stale-definition-proof-input
               (try (entry/assert-definition-inputs-current! @consumer)
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (finally (remove-ns (ns-name source-ns))))))
