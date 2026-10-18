(ns raster.ad.completion-identity-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.value-completion :as completion]
            [raster.analysis.memory :as memory]))

(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))

(defn- forward [input output contract]
  (with-bindings {#'reverse/*source-memory-context*
                  {:values [input] :value-facts {} :replayed {} :unprojectable {} :aliases {}}
                  #'reverse/*generated-product-context* {:source-policy {:unchecked-math false}}}
    (#'reverse/forward-pass
     [output (completion/complete-value contract input)] [input])))

(deftest object-completion-is-a-tangent-identity-not-an-exact-primal-alias
  (let [input (typed 'input 'double) output (typed 'output 'double)
        {:keys [records value-facts aliases fwd-bindings]}
        (forward input output completion/object-input-contract)]
    (is (= :alias (:type (first records))) "the existing transpose transports the tangent")
    (is (nil? (get-in value-facts [output :alias]))
        "boxing a primitive does not preserve its physical value identity")
    (is (= output (#'reverse/exact-replay-root output value-facts)))
    (is (contains? (memory/transitive-closure #{output} aliases) input)
        "positive may-alias information is kept, never interpreted as freshness")
    (is (= (completion/complete-value completion/object-input-contract input)
           (second fwd-bindings)) "the original primal completion is not erased")))

(deftest object-completion-keeps-borrowed-array-aliases
  (let [input (typed 'input 'doubles) output (typed 'output 'doubles)
        {:keys [value-facts aliases]} (forward input output completion/object-input-contract)]
    (is (nil? (get-in value-facts [output :alias])))
    (is (contains? (memory/transitive-closure #{output} aliases) input))
    (is (contains? (memory/transitive-closure #{input} aliases) output))))

(deftest reference-check-target-does-not-prove-a-reference-operand
  (let [input (typed 'input 'double) output (typed 'output 'double)
        contract (completion/reference-check-contract Double)
        {:keys [value-facts aliases fwd-bindings]} (forward input output contract)]
    (is (nil? (get-in value-facts [output :alias]))
        "Class.cast can box a primitive input before checking its reference carrier")
    (is (= output (#'reverse/exact-replay-root output value-facts)))
    (is (contains? (memory/transitive-closure #{output} aliases) input))
    (is (= (completion/complete-value contract input) (second fwd-bindings)))))

(deftest completed-primal-values-keep-their-own-reference-and-trap-behavior
  (doseq [contract [completion/object-input-contract
                    (completion/reference-check-contract Double)]]
    (let [input (typed 'input 'double) output (typed 'output 'double)
          {:keys [fwd-bindings]} (forward input output contract)
          parameter (with-meta input {:tag 'double})
          project #(completion/project-to-jvm-source % {:unchecked-math false}
                                                     {Double 'reference_class})
          ;; Independent source consumes the original primitive in Object context.
          reference (list '. (list 'do 'reference_class) (list 'cast input))
          compile-body (fn [body]
                         (binding [*unchecked-math* false]
                           (eval (list 'let* ['reference_class Double]
                                       (list 'fn* [parameter] body)))))
          oracle (compile-body (list 'let* [output reference]
                                     [(list 'identical? output output)
                                      (list 'identical? input output) output]))
          execute (compile-body (project (list 'let* fwd-bindings
                                                [(list 'identical? output output)
                                                 (list 'identical? input output) output])))]
      (doseq [value [5.125 1000.0 -0.0]]
        (is (= [true false value] (oracle value)))
        (is (= (oracle value) (execute value))))))
  (let [contract (completion/reference-check-contract String)
        {:keys [fwd-bindings]} (forward 'input 'output contract)
        body (completion/project-to-jvm-source (list 'let* fwd-bindings 'output)
                                               {:unchecked-math false} {String 'reference_class})
        execute (eval (list 'let* ['reference_class String] (list 'fn* '[input] body)))]
    (is (nil? (execute nil)))
    (is (thrown? ClassCastException (execute 5.125)))
    (is (= "kept" (execute "kept")))))

(deftest completed-borrowed-storage-still-invalidates-a-replayed-read
  (let [input (typed 'input 'doubles) view (typed 'view 'doubles)
        h (typed 'h 'double) result (typed 'result 'double)
        acc (typed 'acc 'double) index (typed 'index 'long)
        reduction (list 'raster.par/reduce acc 0.0 index 2
                        (list 'raster.numeric/+ acc
                              (list 'raster.numeric/* h (list 'clojure.core/aget view index))))]
    (doseq [owner [nil {:source-policy {:unchecked-math false}}]]
     (with-bindings {#'reverse/*gensym-counter* (atom 0)
                    #'reverse/*source-memory-context*
                    {:values [input h] :value-facts {} :replayed {} :unprojectable {} :aliases {}}
                    #'reverse/*generated-product-context* owner}
      (is (= :replayed-array-overwritten
             (try (#'reverse/forward-pass
                   [view (completion/complete-value completion/object-input-contract input)
                    result reduction 'write (list 'clojure.core/aset input 0 9.0)] [h])
                  :incorrectly-accepted
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))
