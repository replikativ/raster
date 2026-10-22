(ns raster.compiler.core.prepared-program-import-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.passes.scalar.inline :as inline]))

(deftm primal [x :- Double unused :- Double] :- Double (numeric/* x x))
(deftm gradient-root [x :- Double unused :- Double] :- Double
  (let [pair ((reverse/value+grad (var primal) :wrt [0]) x unused)]
    (clojure.core/nth pair 1)))
(deftm grad-root [x :- Double unused :- Double] :- Double
  (let [gradients ((reverse/grad (var primal) :wrt [0]) x unused)]
    (clojure.core/nth gradients 0)))

(defn- normalized-root [root]
  (let [source (core/resolve-deftm-var root)
        _ (core/prepare-deftm-frontend! source)
        frontend (entry/original-source-frontend source)]
    (assoc (transport/normalize-frontend-bindings frontend (get-in frontend [:source-entry :params]))
           :environment (zipmap (get-in frontend [:source-entry :params])
                                (get-in frontend [:source-entry :tags])))))

(defn- expand
  ([consumer] (expand #'gradient-root consumer inline/inline-ad-applications))
  ([root consumer construct]
  (entry/call-with-definition-inputs
   (fn []
     (let [{:keys [walked-body source-normalization environment]} (normalized-root root)]
       (binding [transport/*prepared-program-consumer* consumer]
         (assoc (transport/with-source-construction-history
                 source-normalization
                 #(construct (first walked-body) environment))
                :observations @entry/*emission-inputs*)))))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest direct-application-is-consumed-before-raw-bindings-are-appended
  (let [seen (atom [])
        imported (expand (fn [event placement]
                           (swap! seen conj [event placement])
                           '[module_loss module_gradient nil]))
        [event placement] (last @seen)
        source (:form imported)
        projected-binders (set (map first (get-in event [:projection :bindings])))
        actual-symbols (set (filter symbol? (tree-seq coll? seq source)))]
    (is (some? (reverse/assert-prepared-program-source! (:prepared event))))
    (is (= :binding (:kind placement)))
    (is (some? (:caller-origin event)))
    (is (= [false false true] (mapv nil? (get-in event [:projection :elements]))))
    (is (seq (get-in event [:projection :bindings])))
    (is (not-any? actual-symbols projected-binders)
        "no part of the raw generated program enters the outer source passes")
    (is (contains? actual-symbols 'module_gradient))
    (is (some #(and (= :prepared-reverse-ad (first %))
                    (identical? (:occurrence event) (nth % 2 nil)))
              (:unclosed-source-imports (:observations imported)))
        "capture is not execution or permission to close the import marker")))

(deftest default-expansion-and-ownerless-refusal
  (let [projection {:bindings [['inside 1]] :elements '[loss gradient]}
        calls (atom 0)]
    (is (identical? projection
                    (transport/consume-prepared-program-projection
                     projection {:kind :binding :destination 'result})))
    (binding [transport/*prepared-program-consumer* (fn [& _] (swap! calls inc))]
      (is (= :unclosed-prepared-program-placement
             (reason #(transport/consume-prepared-program-projection
                       projection {:kind :binding :destination 'result})))))
    (is (zero? @calls))))

(deftest grad-mode-keeps-its-own-ordered-slots-without-a-loss-port
  (let [seen (atom nil)
        imported (expand #'grad-root
                         (fn [event placement]
                           (reset! seen [event placement])
                           '[module_gradient nil])
                         inline/inline-ad-applications)]
    (is (= :grad (:mode (first @seen))))
    (is (= [false true] (mapv nil? (get-in (first @seen) [:projection :elements]))))
    (is (some #{'module_gradient} (tree-seq coll? seq (:form imported))))))

(deftest a-captured-prepared-call-uses-the-same-early-consumer
  (let [scope (create-ns (gensym "raster.prepared-import.captured."))
        callable (reverse/value+grad #'primal :wrt [0])
        seen (atom [])]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (intern scope 'captured callable)
        (eval '(raster.core/deftm root [x :- Double unused :- Double] :- Double
                 (let [pair (captured x unused)] (clojure.core/nth pair 1)))))
      (let [imported
            (expand (ns-resolve scope 'root)
                    (fn [event placement]
                      (swap! seen conj [event placement])
                      '[module_loss module_gradient nil])
                    (fn [source environment]
                      (binding [inline/*param-env* environment]
                        (inline/inline-deftm-calls source 1))))]
        (is (seq @seen))
        (is (identical? (:prepared (reverse/captured-prepared-callable callable (meta callable)))
                        (:prepared (first (last @seen)))))
        (is (= :binding (:kind (second (last @seen)))))
        (is (some #{'module_gradient} (tree-seq coll? seq (:form imported))))
        (let [event (first (last @seen))
              symbols (set (filter symbol? (tree-seq coll? seq (:form imported))))]
          (is (not-any? symbols (map first (get-in event [:projection :bindings]))))
          (is (some #(and (= :prepared-reverse-ad (first %))
                          (identical? (:occurrence event) (nth % 2 nil)))
                    (:unclosed-source-imports (:observations imported))))))
      (finally (remove-ns (ns-name scope))))))

(deftest a-consumer-cannot-drop-or-replace-inactive-slots-or-return-expressions
  (doseq [ports [nil '[loss gradient] '[loss gradient other]
                 '[loss nil nil] '[loss (identity gradient) nil]]]
    (is (= :unclosed-prepared-program-outputs
           (reason #(expand (fn [& _] ports)))))))

(deftest module-output-ports-cannot-capture-source-or-collapse-results
  (doseq [ports (list '[same same nil] '[loss x nil] '[loss unused nil]
                     [(with-meta 'loss {:raster.type/tag 'double}) 'gradient nil]
                     '[qualified/loss gradient nil])]
    (is (= :unclosed-prepared-program-outputs
           (reason #(expand (fn [& _] ports))))))
  (doseq [select-port [(fn [_ placement] (:destination placement))
                      (fn [event _] (ffirst (get-in event [:projection :bindings])))]]
    (is (= :unclosed-prepared-program-outputs
           (reason #(expand (fn [event placement]
                              ['module_loss (select-port event placement) nil])))))))

(deftest an-exact-projection-cannot-be-consumed-twice
  (let [calls (atom 0)]
    (is (= :unclosed-prepared-program-placement
           (reason #(expand (fn [event placement]
                              (swap! calls inc)
                              (transport/consume-prepared-program-projection
                               (:projection event) placement))))))
    (is (= 1 @calls))))

(deftest a-copied-projection-does-not-borrow-the-original-live-event
  (let [calls (atom 0)]
    (is (= :unclosed-prepared-program-placement
           (reason #(expand (fn [event placement]
                              (swap! calls inc)
                              (transport/consume-prepared-program-projection
                               (into {} (:projection event)) placement))))))
    (is (= 1 @calls))))

(deftest copied-projections-refuse-before-the-first-consumer-call
  (let [project reverse/project-prepared-value+grad-call
        calls (atom 0)]
    (with-redefs [reverse/project-prepared-value+grad-call
                  (fn [& arguments]
                    (into {} (apply project arguments)))]
      (is (= :unclosed-prepared-program-placement
             (reason #(expand (fn [& _] (swap! calls inc)))))))
    (is (zero? @calls))))

(deftest nested-live-placement-refuses-before-the-consumer
  (let [project reverse/project-prepared-value+grad-call
        calls (atom 0)]
    (with-redefs [reverse/project-prepared-value+grad-call
                  (fn [& arguments]
                    (transport/consume-prepared-program-projection
                     (apply project arguments)
                     {:kind :nested :destination nil :reserved-symbols #{}}))]
      (is (= :unclosed-prepared-program-placement
             (reason #(expand (fn [& _] (swap! calls inc)))))))
    (is (zero? @calls))))
