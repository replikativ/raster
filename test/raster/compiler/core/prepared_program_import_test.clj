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
(deftm ordered-root [x :- Double unused :- Double] :- Double
  (let [before (numeric/+ x unused)
        pair ((reverse/value+grad (var primal) :wrt [0]) before unused)
        after (numeric/- (clojure.core/nth pair 1) before)]
    after))
(deftm two-import-root [x :- Double unused :- Double] :- Double
  (let [first-pair ((reverse/value+grad (var primal) :wrt [0]) x unused)
        second-pair ((reverse/value+grad (var primal) :wrt [0]) unused x)]
    (numeric/+ (clojure.core/nth first-pair 1) (clojure.core/nth second-pair 1))))

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
                :normalization source-normalization
                :observations @entry/*emission-inputs*)))))))

(defn- construction-view [imported]
  (entry/call-with-definition-inputs
   #(transport/source-construction-view (:source-construction imported)
                                        (:normalization imported) (:form imported))))

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
    (is (= 2 (count (get-in event [:argument-plan :object-bindings]))))
    (is (= 2 (count (get-in event [:argument-plan :input-bindings]))))
    (is (= [Double/TYPE Double/TYPE]
           (mapv :parameter-class (get-in event [:argument-plan :slots]))))
    (is (some? (get-in event [:argument-plan :boxed-entry-bridge])))
    (is (= :binding (:kind placement)))
    (let [{:keys [scope index prefix]} (:binding-site placement)
          cut (first (:prepared-program-cuts (construction-view imported)))]
      (is (= 0 index))
      (is (= [] prefix))
      (is (identical? (:destination placement) (nth (second scope) (* 2 index))))
      (is (identical? (:caller event) (nth (second scope) (inc (* 2 index)))))
      (is (identical? event (:event cut)))
      (is (identical? placement (:placement cut)))
      (is (= '[module_loss module_gradient nil] (:ports cut))))
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

(deftest canonical-import-inserts-only-its-pending-spine-at-the-owned-cut
  (let [seen (atom nil)
        bindings '[ [module_loss 1.0] [module_gradient 2.0]]
        imported (expand #'ordered-root
                         (fn [event placement]
                           (let [import {:event event :status :pending}]
                             (reset! seen [event placement import])
                             {:ports '[module_loss module_gradient nil]
                              :bindings bindings :import import}))
                         inline/inline-ad-applications)
        [event placement import] @seen
        cut (first (:prepared-program-cuts (construction-view imported)))
        source-bindings (vec (partition 2 (second (:form imported))))
        ids (mapv first source-bindings)
        loss-index (.indexOf ids 'module_loss)
        raw-ids (set (map first (get-in event [:projection :bindings])))]
    (is (pos? loss-index) "the already-constructed prefix remains before the module")
    (is (= bindings (subvec source-bindings loss-index (+ loss-index 2))))
    (is (< (+ loss-index 1) (dec (count source-bindings)))
        "the caller continuation remains after the module")
    (is (not-any? raw-ids ids))
    (is (identical? placement (:placement cut)))
    (is (identical? bindings (:canonical-bindings cut)))
    (is (identical? import (:import cut)))
    (is (= :pending (get-in cut [:import :status])))
    (is (some #(and (= :prepared-reverse-ad (first %))
                    (identical? (:occurrence event) (nth % 2 nil)))
              (get-in imported [:observations :unclosed-source-imports]))
        "construction never grants execution or clears the source import")))

(deftest canonical-import-refuses-incomplete-capturing-or-preclosed-spines
  (doseq [change [(fn [r _] (assoc r :bindings '([module_loss 1.0])))
                  (fn [r _] (assoc r :bindings '[[]]))
                  (fn [r _] (assoc r :bindings '[[module_loss 1.0] [module_loss 2.0]]))
                  (fn [r _] (assoc r :bindings '[[module_loss 1.0]]))
                  (fn [r _] (update r :bindings conj ['foreign/name 0]))
                  (fn [r _] (update r :bindings conj [(with-meta 'annotated {:tag 'double}) 0]))
                  (fn [r placement] (update r :bindings conj [(:destination placement) 0]))
                  (fn [r placement]
                    (update r :bindings conj [(first (second (get-in placement [:binding-site :scope]))) 0]))
                  (fn [r _] (assoc-in r [:import :event] {}))
                  (fn [r _] (assoc-in r [:import :status] :closed))
                  (fn [r _] (assoc-in r [:import :admitted?] true))
                  (fn [r _] (assoc-in r [:bindings 0 1] 'module_gradient))
                  (fn [r _] (assoc-in r [:bindings 0 1] 'module_loss))
                  (fn [r _] (assoc-in r [:bindings 0 1] 'unknown_input))
                  (fn [r _] (assoc r :bindings '[[module_loss count]
                                                              [count 1.0]
                                                              [module_gradient module_loss]]))
                  (fn [r placement]
                    (assoc-in r [:bindings 0 1] (:destination placement)))
                  (fn [r placement]
                    (let [{:keys [scope index]} (:binding-site placement)]
                      (assoc-in r [:bindings 0 1] (nth (second scope) (* 2 (inc index))))))
                  (fn [r _] (assoc r :extra true))]]
    (is (= :unclosed-prepared-program-bindings
           (reason #(expand #'ordered-root
                            (fn [event placement]
                              (change {:ports '[module_loss module_gradient nil]
                                       :bindings '[[module_loss 1.0] [module_gradient 2.0]]
                                       :import {:event event :status :pending}}
                                      placement))
                            inline/inline-ad-applications))))))

(deftest canonical-initializers-see-live-parameters-prefix-and-earlier-module-bindings
  (let [seen (atom nil)]
    (expand #'ordered-root
            (fn [event placement]
              (reset! seen placement)
              (let [prior (ffirst (get-in placement [:binding-site :prefix]))]
                {:ports '[module_loss module_gradient nil]
                 :bindings [['module_loss prior]
                            ['module_gradient '(clojure.core/+ module_loss x)]]
                 :import {:event event :status :pending :admitted? false}}))
            inline/inline-ad-applications)
    (is (contains? (get-in @seen [:binding-site :live-symbols]) 'x))
    (is (contains? (get-in @seen [:binding-site :live-symbols])
                   (ffirst (get-in @seen [:binding-site :prefix]))))))

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

(deftest prepared-cuts-retain-their-position-and-already-constructed-prefix
  (let [seen (atom nil)
        imported (expand #'ordered-root
                         (fn [event placement]
                           (reset! seen [event placement])
                           '[module_loss module_gradient nil])
                         inline/inline-ad-applications)
        [event placement] @seen
        {:keys [scope index prefix]} (:binding-site placement)]
    (is (= 1 index))
    (is (= 1 (count prefix)))
    (is (identical? (first (first prefix)) (first (second scope))))
    (is (identical? (second (first prefix)) (second (second scope))))
    (is (identical? (:caller event) (nth (second scope) (inc (* 2 index)))))
    (is (= 1 (count (:prepared-program-cuts (construction-view imported)))))))

(deftest multiple-module-cuts-have-distinct-ordered-sites-and-ports
  (let [counter (atom 0)
        imported (expand #'two-import-root
                         (fn [_ _]
                           (let [ordinal (swap! counter inc)]
                             [(symbol (str "module_loss_" ordinal))
                              (symbol (str "module_gradient_" ordinal)) nil]))
                         inline/inline-ad-applications)
        cuts (:prepared-program-cuts (construction-view imported))]
    (is (= [0 1] (mapv #(get-in % [:placement :binding-site :index]) cuts)))
    (is (= [0 1] (mapv #(count (get-in % [:placement :binding-site :prefix])) cuts)))
    (is (= 4 (count (set (remove nil? (mapcat :ports cuts))))))
    (is (not (identical? (get-in (first cuts) [:event :occurrence])
                         (get-in (second cuts) [:event :occurrence]))))))

(deftest invalid-binding-sites-refuse-before-the-consumer-is-called
  (doseq [change [(fn [placement] (dissoc placement :binding-site))
                  #(assoc-in % [:binding-site :index] -1)
                  #(assoc-in % [:binding-site :index] 1000)
                  #(assoc-in % [:binding-site :prefix] :not-bindings)
                  #(assoc-in % [:binding-site :prefix] [['bad]])
                  #(update-in % [:binding-site :scope]
                              (fn [[head bindings & body]]
                                (list* head (into bindings ['forged-sibling 1.0]) body)))
                  #(update-in % [:binding-site :scope]
                              (fn [[head bindings & body]]
                                (list* head (assoc bindings 1 (apply list (second bindings))) body)))]]
    (let [consume transport/consume-prepared-program-projection
          calls (atom 0)]
      (with-redefs [transport/consume-prepared-program-projection
                    (fn [projection placement] (consume projection (change placement)))]
        (is (= :unclosed-prepared-program-placement
               (reason #(expand (fn [& _] (swap! calls inc) '[loss gradient nil]))))))
      (is (zero? @calls)))))

(deftest a-later-module-cannot-reuse-an-earlier-modules-ports
  (let [counter (atom 0)]
    (is (= :unclosed-prepared-program-outputs
           (reason #(expand #'two-import-root
                            (fn [& _]
                              (if (= 1 (swap! counter inc))
                                '[first_loss first_gradient nil]
                                '[second_loss first_gradient nil]))
                            inline/inline-ad-applications))))
    (is (= 2 @counter))))

(deftest well-formed-but-replaced-prefixes-refuse-before-consumption
  (doseq [prefix [[] [['forged 1.0]]]]
    (let [consume transport/consume-prepared-program-projection
          calls (atom 0)]
      (with-redefs [transport/consume-prepared-program-projection
                    (fn [projection placement]
                      (consume projection (assoc-in placement [:binding-site :prefix] prefix)))]
        (is (= :unclosed-prepared-program-placement
               (reason #(expand #'ordered-root
                                (fn [& _] (swap! calls inc) '[loss gradient nil])
                                inline/inline-ad-applications)))))
      (is (zero? @calls)))))

(deftest ports-cannot-capture-later-binders-introduced-by-this-inliner-pass
  (let [candidate (atom nil)]
    (is (= :unclosed-prepared-program-outputs
           (reason
            #(expand #'gradient-root
                     (fn [_ placement]
                       (let [ids (take-nth 2 (second (get-in placement [:binding-site :scope])))
                             fresh-id (first (remove (:reserved-symbols placement) ids))]
                         (reset! candidate fresh-id)
                         ['module_loss (some-> fresh-id name symbol) nil]))
                     (fn [source environment]
                       (binding [inline/*param-env* environment]
                         (inline/inline-deftm-calls source 1)))))))
    (is (symbol? @candidate) "exercise an actual new binder, not a malformed nil port")))

(deftest ordinary-expansion-does-not-record-a-module-cut
  (let [view (construction-view (expand nil))]
    (is (empty? (:prepared-program-cuts view)))
    (is (empty? (:prepared-program-scopes view)))))

(deftest module-cut-and-scope-records-are-not-generic-source-edges
  (let [imported (expand (fn [& _] '[loss gradient nil]))
        view (construction-view imported)
        scopes (mapv :scope (:prepared-program-scopes view))]
    (is (= 1 (count (:prepared-program-cuts view))))
    (is (seq scopes) "inliner passes may register more than one scope snapshot")
    (is (every? (fn [cut]
                  (some #(identical? % (get-in cut [:placement :binding-site :scope])) scopes))
                (:prepared-program-cuts view))
        "each cut belongs to an actually registered scope")
    (is (not-any? (fn [edge] (some #(identical? % (:constructed edge)) scopes))
                  (:type-elaborations view)))
    (entry/call-with-definition-inputs
     #(do
        (is (nil? (transport/construction-occurrence (:source-construction imported) nil)))
        (is (nil? (transport/prepared-program-binding-occurrence
                   (:source-construction imported) nil)))))))

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
