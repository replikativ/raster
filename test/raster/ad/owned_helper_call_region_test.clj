(ns raster.ad.owned-helper-call-region-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.backend.jvm.source-owner :as source-owner]))

(deftm response [a :- Double k :- Double x :- Double] :- Double
  (n/* a (n// x (n/+ k x))))
(deftm direct-call [a :- Double k :- Double x :- Double] :- Double
  (response a k x))

(def policy {:unchecked-math false})

(defn- retained-call []
  (let [v (core/resolve-deftm-var #'direct-call {:on-miss :self})
        body (first (core/ensure-walked-body! v))]
    (first (filter #(and (seq? %) (= '.invk (first %))
                         (= 'raster.ad.owned-helper-call-region-test/response
                            (:operation (op/call-description %))))
                   (tree-seq coll? seq body)))))

(defn- capture [expression]
  (let [{:keys [operation implementation-op arguments]} (op/call-description expression)
        hint (:tag (meta implementation-op))
        carrier (if (class? hint) hint (:iface-class (get @types/fn-interface-by-name hint)))
        method (first (.getDeclaredMethods ^Class carrier))]
    (#'rev/capture-selected-helper-emission operation implementation-op (count arguments) method)))

(defn- compile-region [region paired? params]
  (let [{:keys [call helper method source-policy guard]} region
        companion (with-meta (gensym "companion") {:tag (.getDeclaringClass ^java.lang.reflect.Method method)})
        captures (cond-> (mapv (fn [{:keys [class port]}] {:port port :value class}) (:class-ports call))
                   paired? (conj {:port companion :value (:companion helper)}))
        ;; Keep the ORIGINAL receiver binding even when using the companion.
        invocation (if paired?
                     (with-meta (list* (first (:invocation call)) companion (:input-values call))
                       (meta (:invocation call)))
                     (:invocation call))
        execute (source-owner/compile-function
                 params (list 'let* (:bindings call) invocation)
                 (fn [parameters body] (list 'fn parameters body))
                 {:source-policy source-policy :source-ns *ns* :captures captures
                  :reference-bindings (into {} (map (juxt :class :port)) (:class-ports call))
                  :methods [method] :guard guard})]
    (fn [& arguments]
      (guard)
      (let [result (apply execute arguments)]
        (guard)
        (if-not paired? result
          (let [[value pullback] result]
            [value (fn [seed]
                     (guard)
                     (let [gradient (pullback seed)] (guard) gradient))]))))))

(defn- fixture []
  (let [original (retained-call)
        implementation (second original)
        arguments '[a (do (swap! events conj :later) (clojure.core/aget xs 0)) exposure]
        expression (with-meta (list* '.invk implementation arguments) (meta original))
        params ['a (with-meta 'xs {:tag 'doubles}) 'exposure 'events]
        owner (capture expression)
        region (rev/prepare-helper-call-region expression owner [0 2] policy gensym (set params))]
    {:expression expression :owner owner :region region :params params
     :source (binding [*unchecked-math* false]
               (eval (list 'fn params (list 'let* '[prefix 0.3] expression))))}))

(deftest original-input-completion-precedes-the-next-operand
  (let [{:keys [region params source expression]} (fixture)
        bindings (get-in region [:call :bindings])]
    (is (identical? expression (:original region)))
    (is (= (second expression)
           (:operand (completion/canonical-parts (second bindings)))))
    (doseq [[original [_ completed]] (map vector (drop 2 expression) (rest (partition 2 bindings)))]
      (is (identical? original (:operand (completion/canonical-parts completed)))
          "each original operand stays inside its input completion, not a raw ANF snapshot"))
    (doseq [f [source (compile-region region false params) (compile-region region true params)]]
      (let [events (atom [])]
        (is (thrown? NullPointerException (f nil (double-array 0) 1.0 events)))
        (is (= [] @events)))
      (let [events (atom [])]
        (is (thrown? ArrayIndexOutOfBoundsException (f 0.8 (double-array 0) 1.0 events)))
        (is (= [:later] @events))))))

(deftest paired-call-retains-its-transparent-program-and-delayed-guard
  (let [{:keys [region params owner]} (fixture)
        f (compile-region region true params)
        original (compile-region region false params)]
    (is (seq? (get-in region [:helper :program :form])))
    (is (= [0 2] (get-in region [:helper :active-indices])))
    (doseq [[a k x] [[0.8 0.7 1.0] [1.2 0.4 0.2] [-0.2 1.1 2.0]]]
      (let [events (atom []) [value pullback] (f a (double-array [k]) x events)
            expected [(/ x (+ k x)) (/ (* a k) (* (+ k x) (+ k x)))]]
        (is (< (Math/abs (- value (/ (* a x) (+ k x)))) 1e-12))
        (is (= value (original a (double-array [k]) x (atom []))))
        (is (= [:later] @events))
        (doseq [seed [1.0 2.0]]
          (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12)
                                 (map #(* seed %) expected) (pullback seed)))))))
    (let [[_ pullback] (f 0.8 (double-array [0.7]) 1.0 (atom []))
          receiver (get-in owner [:signature :definition :receiver-var])]
      (with-redefs-fn {receiver nil}
        #(do (is (thrown? clojure.lang.ExceptionInfo (f 0.8 (double-array [0.7]) 1.0 (atom []))))
             (is (thrown? clojure.lang.ExceptionInfo (pullback 1.0))))))))

(deftest call-region-rejects-missing-descriptor-and-open-lexical-correspondence
  (let [{:keys [expression owner params]} (fixture)
        missing-hint (with-meta (list* '.invk (with-meta (second expression) nil) (drop 2 expression))
                       (meta expression))]
    (doseq [[call policy scope] [[missing-hint policy (set params)]
                                [expression nil (set params)]
                                [expression policy (conj (set params) (second expression))]]]
      (is (= :ad-unclosed-helper-call-region
             (try (rev/prepare-helper-call-region call owner [0 2] policy gensym scope) :accepted
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
