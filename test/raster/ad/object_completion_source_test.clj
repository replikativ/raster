(ns raster.ad.object-completion-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.core.value-completion :as completion]))

(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))

(defn- source-function [params tags body]
  ;; Generated functions with retained deftm bodies are a supported public AD
  ;; input. No generated-product owner, selected-call proof or policy is added.
  (with-meta (eval (list 'fn params (completion/project-to-jvm-source body)))
    {:raster.core/deftm true :raster.core/deftm-params params
     :raster.core/deftm-tags tags :raster.core/deftm-walked-body [body]
     :ns (the-ns 'raster.ad.object-completion-source-test)}))

(deftest ordinary-public-ad-consumes-object-input-completion
  (doseq [[tag cast] [['double double] ['float float]]]
    (let [x (typed 'x tag) materialized (typed 'materialized tag)
          result (typed 'result tag)
          term (completion/complete-value completion/object-input-contract x)
          source (source-function [x] [tag]
                                  (list 'let* [materialized term
                                               result (list 'raster.numeric/* materialized materialized)] result))
          paired (reverse/value+grad source)
          directional (jvp/jvp source)]
      (doseq [input (map cast [-2.0 0.0 3.0])]
        (is (= [(cast (* input input)) (cast (* 2.0 input))] (paired input)))
        (is (= [(cast (* input input)) (cast (* 4.0 input))]
               (directional input (cast 2.0)))))
      (is (some #(= term %) (tree-seq coll? seq (:raster.core/deftm-walked-body (meta paired))))
          "the retained derivative keeps canonical completion for AD re-entry")
      (is (some #(= term %) (tree-seq coll? seq (:raster.core/deftm-walked-body (meta directional))))))))

(deftest ordinary-object-completion-retains-primal-reference-and-absence
  (let [x (typed 'x 'double) materialized (typed 'materialized 'double)
        source (source-function [x] ['double]
                                (list 'let* [materialized (completion/complete-value
                                                         completion/object-input-contract x)] materialized))
        paired (reverse/value+grad source)
        directional (jvp/jvp source)
        input (Double. 5.125)]
    (is (identical? input (first (paired input))))
    (is (identical? input (first (directional input 2.0))))
    ;; Completion itself does not invent an unbox, a presence guard or zero.
    (is (= [nil 1.0] (paired nil)))
    (is (= [nil 2.0] (directional nil 2.0)))))

(deftest object-completion-reenters-reverse-differentiation
  (let [x (typed 'x 'double) materialized (typed 'materialized 'double)
        result (typed 'result 'double)
        source (source-function [x] ['double]
                                (list 'let* [materialized (completion/complete-value
                                                         completion/object-input-contract x)
                                             result '(raster.numeric/* materialized materialized)] result))
        twice (reverse/grad (reverse/grad source))
        directional (jvp/hvp source)]
    (doseq [input [-2.0 0.0 3.0]]
      (is (= 2.0 (twice input)))
      (is (= [[(* 2.0 input)] [4.0]] (directional input 2.0))))))

(deftest ordinary-completion-keeps-source-effects-once
  (let [x (typed 'x 'double) out (typed 'out 'double)
        source (source-function [x 'events] ['double nil]
                                (list 'let* ['tick '(clojure.core/swap! events clojure.core/inc)
                                             out (completion/complete-value completion/object-input-contract x)] out))
        paired (reverse/value+grad source :wrt [0])
        directional (jvp/jvp source :wrt [0])]
    (doseq [[f args expected] [[paired [5.125] [5.125 1.0 nil]]
                               [directional [5.125] [5.125 2.0]]]]
      (let [events (atom 0)
            result (if (= f paired)
                     (f (first args) events)
                     (f (first args) events 2.0))]
        (is (= expected result))
        (is (= 1 @events))))))

(deftest ordinary-and-completed-aliases-mask-stale-tangents
  (doseq [complete [identity #(completion/complete-value completion/object-input-contract %)]]
    (let [masked (#'jvp/jvp-fold ['saved 'x 'saved (complete 'y)] {'x 'dx})
          self (#'jvp/jvp-fold ['saved (complete 'saved)] {'saved 'incoming})]
      (is (not (contains? (:tenv masked) 'saved)))
      (is (= 'incoming (get-in self [:tenv 'saved]))))))

(deftest completed-source-derivatives-compile-without-the-held-product-rewrite
  (doseq [[tag dtype cast] [['double :double double] ['float :float float]]]
    (let [x (typed 'x tag) out (typed 'out tag) result (typed 'result tag)
          source (source-function [x] [tag]
                                  (list 'let* [out (completion/complete-value completion/object-input-contract x)
                                               result '(raster.numeric/* out out)] result))
          owner (the-ns 'raster.ad.object-completion-source-test)
          publish (fn [f]
                    ;; compile-aot takes a Var: it publishes artifact metadata
                    ;; there. Keep that public contract in this test too.
                    (let [v (intern owner (gensym "completion_derivative__") f)]
                      (alter-meta! v merge (meta f) {:private true})
                      v))
          paired-var (publish (reverse/value+grad source))
          directional-var (publish (jvp/jvp source))]
      (try
        (let [paired (pipeline/compile-aot paired-var :target :jvm :dtype dtype)
              directional (pipeline/compile-aot directional-var :target :jvm :dtype dtype)]
          (doseq [input (map cast [-2.0 0.0 3.0])]
            (is (= [(cast (* input input)) (cast (* 2.0 input))] (vec (paired input))))
            (is (= [(cast (* input input)) (cast (* 4.0 input))]
                   (vec (directional input (cast 2.0)))))))
        (finally
          (doseq [v [paired-var directional-var]]
            (ns-unmap owner (:name (meta v)))))))))

(deftest ordinary-completion-records-do-not-grant-owned-conversion-rules
  (let [x (typed 'x 'double) out (typed 'out 'double)
        forward (fn [term]
                  (with-bindings {#'reverse/*generated-product-context* nil
                                  #'reverse/*source-memory-context*
                                  {:values [x] :value-facts {} :replayed {} :unprojectable {} :aliases {}}}
                    (#'reverse/forward-pass [out term] [x])))]
    (let [{:keys [records fwd-bindings value-facts]}
          (forward (completion/complete-value completion/object-input-contract x))]
      (is (= :alias (:type (first records))))
      (is (nil? (get-in value-facts [out :alias])))
      (is (= (completion/complete-value completion/object-input-contract x) (second fwd-bindings))))
    (is (= :ad-unclosed-completion-transpose
           (try (forward (completion/complete-value completion/object-input-contract
                                                    '(raster.numeric/* x x)))
                :accepted
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (doseq [contract [(completion/reference-check-contract Double)
                      (completion/primitive-input-contract Float/TYPE {:unchecked-math false})]]
      (is (not (#'reverse/structural-completion? (completion/complete-value contract x)))))))

(deftest jvp-refuses-unclosed-and-malformed-completions-even-when-inactive
  (doseq [term [(completion/complete-value completion/object-input-contract '(raster.numeric/* x x))
                (completion/complete-value (completion/reference-check-contract Double) 'x)
                (completion/complete-value
                 (completion/primitive-input-contract Float/TYPE {:unchecked-math false}) 'x)
                (list completion/completion-symbol {:source-context :unknown} 'x)
                (list completion/completion-symbol completion/object-input-contract 'x :extra)]
          tangents [{} {'x 'dx}]]
    (is (= :jvp-unclosed-value-completion
           (try (#'jvp/jvp-fold ['out term] tangents) :accepted
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
