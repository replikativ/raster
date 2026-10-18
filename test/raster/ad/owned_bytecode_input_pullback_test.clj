(ns raster.ad.owned-bytecode-input-pullback-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.ad.reverse :as reverse]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.source-owner :as source-owner]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]))

(deftm response [a :- Double k :- Double x :- Double] :- Double
  (n/* a (n// x (n/+ k x))))
(deftm factor [a :- Double k :- Double x :- Double] :- Double
  (n/* 2.0 (response a k x)))
(deftm shadow-factor [a :- Double k :- Double x :- Double] :- Double
  (let [amplitude (n/+ a 0.0)] (n/* 2.0 (response amplitude k x))))
(deftm consume-float [x :- Float] :- Float x)
(deftm narrowed-factor [a :- Double k :- Double x :- Double] :- Float
  (consume-float (response a k x)))
(deftm nested-factor [a :- Double k :- Double x :- Double] :- Double
  (n/* 2.0 (response a (n/+ k x) x)))
(deftm shared-input-factor [a :- Double x :- Double] :- Double
  (n/* 2.0 (response a a x)))
(deftm deep-factor [a :- Double k :- Double x :- Double] :- Double
  (n/* 2.0 (n/- (response a k x) 0.3)))
(deftm inactive-nested-factor [a :- Double k :- Double x :- Double] :- Double
  (n/* 2.0 (response a k (n/+ x 1.0))))
(deftm branch-factor [a :- Double k :- Double x :- Double] :- Double
  (n/* 2.0 (response a (if (pos? x) k x) x)))
(deftm shadow-tree-factor [a :- Double k :- Double x :- Double] :- Double
  (let [amplitude (n/+ a 0.0)
        threshold (n/+ k 0.0)]
    (n/* 2.0 (response amplitude (n/+ threshold x) x))))

(use-fixtures :once (fn [run] (response 0.8 0.7 1.0) (run)))

(defn- receiver [v]
  (let [source (core/resolve-deftm-var v)]
    (ns-resolve (.ns ^clojure.lang.Var source) (symbol (str (:name (meta source)) "-impl")))))

(defn- input-owner [v outer-operation index]
  (let [r (receiver v)
        _ (core/ensure-bytecode-implementation! r)
        emission (:raster.core/bytecode-emission (meta r))
        original (first (filter #(= outer-operation (or (:raster.op/original (meta %)) (first %)))
                                (vals (:invocation-catalog emission))))
        context (bytecode/retained-invocation-context emission original)
        ports (get-in context [:inputs index :ports])]
    (assert original "fixture must retain the actual outer call")
    (assert context "fixture must retain the actual emitted input")
    {:emission emission :original original :ports ports :index index}))

(defn- prepare [v outer-operation index active-names]
  (let [{:keys [emission original ports]} (input-owner v outer-operation index)
        names (into {} (map (fn [id] [(name id) id])) (keys ports))]
    (reverse/prepare-bytecode-input-pullback
     emission original index (zipmap (keys ports) (repeat 'double))
     (mapv names active-names))))

(defn- arguments [prepared a k x]
  (let [values {"a" a "amplitude" a "k" k "x" x}]
    (mapv #(get values (name %)) (:params prepared))))

(defn- reference [a k x]
  (let [den (+ k x)]
    [(/ (* a x) den) (/ x den) (- (/ (* a x) (* den den)))]))

(deftest emitted-parent-input-identifies-its-actual-nested-call
  (let [{:keys [emission original index]} (input-owner #'factor 'raster.numeric/* 1)
        parent (bytecode/retained-invocation-context emission original)
        child (bytecode/retained-input-call-context emission original index)]
    (is (some? child))
    (is (identical? (:expression child) (get-in parent [:inputs index :operand])))
    (is (= (dissoc child :source-original)
           (bytecode/retained-invocation-context emission (:source-original child))))
    (is (nil? (bytecode/retained-input-call-context
               emission (with-meta (apply list original) (meta original)) index)))
    (is (nil? (bytecode/retained-input-call-context emission original -1)))))

(deftest owned-compound-input-reuses-the-shared-binding-engine
  (doseq [[v amplitude-name] [[#'factor "a"] [#'shadow-factor "amplitude"]]]
    (let [prepared (prepare v 'raster.numeric/* 1 [amplitude-name "k"])
          f (#'reverse/compile-prepared-value+grad prepared)
          original (get-in prepared [:source-owner :input-region :invoke])]
      (is (= :jvm-bytecode (get-in prepared [:source-owner :backend])))
      (is (seq? (get-in prepared [:program :form])))
      (doseq [[a k x] [[0.8 0.7 1.0] [1.2 0.4 0.2] [-0.2 1.1 2.0]]]
        (let [args (arguments prepared a k x)
              [value pb] (apply f args)
              [expected da dk] (reference a k x)]
          (is (= (Double/doubleToRawLongBits (apply original args))
                 (Double/doubleToRawLongBits value)))
          (is (< (Math/abs (- expected value)) 1e-12))
          (doseq [seed [0.0 1.0 2.0]]
            (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12)
                                   [(* seed da) (* seed dk)] (pb seed)))))
          (is (= [nil nil] (pb nil)))))
      (is (= :unprojected-jvm-source-owner
             (try (source-owner/assert-portable-source! prepared)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest generic-parent-does-not-establish-selected-float-input-ownership
  (let [r (receiver #'narrowed-factor)
        _ (core/ensure-bytecode-implementation! r)
        emission (:raster.core/bytecode-emission (meta r))
        original (first (filter #(= 'raster.ad.owned-bytecode-input-pullback-test/consume-float
                                    (first %))
                                (vals (:invocation-catalog emission))))]
    (is (some? original))
    (is (nil? (bytecode/retained-invocation-context emission original)))
    (is (= :unowned-bytecode-input-region
           (try (reverse/prepare-bytecode-input-pullback emission original 0 {} [])
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest repeated-source-port-accumulates-both-helper-contributions
  (let [prepared (prepare #'shared-input-factor 'raster.numeric/* 1 ["a"])
        f (#'reverse/compile-prepared-value+grad prepared)]
    (doseq [[a x] [[0.8 1.0] [1.2 0.2] [0.3 2.0]]]
      (let [[value pb] (apply f (arguments prepared a nil x))
            [expected da dk] (reference a a x)]
        (is (< (Math/abs (- value expected)) 1e-12))
        (is (< (Math/abs (- (first (pb 1.0)) (+ da dk))) 1e-12))
        (is (= [nil] (pb nil)))))))

(deftest input-owner-fences-forward-and-delayed-pullback
  (doseq [v [#'factor #'deep-factor]]
   (let [prepared (prepare v 'raster.numeric/* 1 ["a" "k"])
        f (#'reverse/compile-prepared-value+grad prepared)
        args (arguments prepared 0.8 0.7 1.0)
        [_ pb] (apply f args)]
    (with-redefs-fn {(receiver #'response) nil}
      #(do (is (thrown? clojure.lang.ExceptionInfo (apply f args)))
           (is (thrown? clojure.lang.ExceptionInfo (pb 1.0)))))
    (is (= 2 (count (pb 1.0)))))))

(deftest nested-typed-inputs-compose-through-the-shared-engine
  (doseq [[v k-transform x-transform offset]
          [[#'nested-factor + identity 0.0]
           [#'deep-factor (fn [k _] k) identity 0.3]
           [#'inactive-nested-factor (fn [k _] k) inc 0.0]]]
    (let [prepared (prepare v 'raster.numeric/* 1 ["a" "k"])
          f (#'reverse/compile-prepared-value+grad prepared)
          original (get-in prepared [:source-owner :input-region :invoke])]
      (is (= 2 (count (:helpers prepared))))
      (doseq [[a k x] [[0.8 0.7 1.0] [1.2 0.4 0.2] [-0.2 1.1 2.0]]]
        (let [args (arguments prepared a k x)
              [value pb] (apply f args)
              [expected da dk] (reference a (k-transform k x) (x-transform x))]
          (is (= (Double/doubleToRawLongBits (apply original args))
                 (Double/doubleToRawLongBits value)))
          (is (< (Math/abs (- value (- expected offset))) 1e-12))
          (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12) [da dk] (pb 1.0)))))))))

(deftest unclosed-branch-operands-do-not-gain-early-anf-permission
  (is (= :ad-unclosed-bytecode-input-region
         (try (prepare #'branch-factor 'raster.numeric/* 1 ["a" "k"])
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))

(deftest earlier-object-input-completes-before-nested-later-call
  (let [prepared (prepare #'shadow-tree-factor 'raster.numeric/* 1 ["amplitude" "threshold"])
        f (#'reverse/compile-prepared-value+grad prepared)
        region (get-in prepared [:source-owner :input-region])
        original (:invoke region)
        ports (:physical-ports region)
        by-name (into {} (map (fn [[id facts]] [(name id) facts])) ports)
        run (fn [call amplitude threshold]
              (apply call (mapv #({"amplitude" amplitude "threshold" threshold "x" 1.0} (name %))
                                (:params prepared))))]
    ;; These are actual Object slots of the owning emitter, not fabricated
    ;; logical Double hints on new primitive parameters. Calling the outlined
    ;; region with invalid carriers tests its original completion boundaries.
    (is (= :ref (:type (get by-name "amplitude"))))
    (is (= :ref (:type (get by-name "threshold"))))
    (doseq [call [original f]]
      (is (thrown? NullPointerException (run call nil "later-invalid")))
      (is (thrown? ClassCastException (run call "earlier-invalid" nil)))
      (is (thrown? ClassCastException (run call 0.8 "later-invalid"))))))

(deftest recursive-derivation-cannot-certify-itself-with-a-later-epoch
  ;; Change only this test's observed epoch, not a real registry or cache.
  ;; Recording the outer capture matters: a helper's own guard could also
  ;; reject the mutation and otherwise mask a late snapshot by this producer.
  (prepare #'factor 'raster.numeric/* 1 ["a" "k"])
  (let [before (normalized/revisions)
        current (atom before)
        captures (atom [])
        derive @#'reverse/derive-flat-product-program
        execution-guard normalized/execution-guard]
    (with-redefs-fn
      {#'normalized/revisions #(deref current)
       #'normalized/execution-guard
       (fn [source-ns expected projections]
         (swap! captures conj expected)
         (execution-guard source-ns expected projections))
       #'reverse/derive-flat-product-program
       (fn [& args]
         (let [program (apply derive args)]
           (swap! current update :template-revision inc)
           program))}
      #(is (= :ad-normalized-stale-rules
              (try (prepare #'factor 'raster.numeric/* 1 ["a" "k"])
                   (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (not= before @current))
    (is (= before (last @captures)))))
