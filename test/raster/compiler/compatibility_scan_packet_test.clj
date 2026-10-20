(ns raster.compiler.compatibility-scan-packet-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.par]
            [raster.numeric]
            [raster.compiler.ir.abstract-value :as value]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- options [dtype]
  {:dtype dtype :array-types {'input dtype 'out dtype}
   :scalar-types {'n :long}
   :values {'input (value/tensor {:dtype dtype :shape [33]})
            'out (value/tensor {:dtype dtype :shape [34]})}})

(defn- scan-form [mode dtype bound]
  (list (if (= mode :inclusive) 'raster.par/scan 'raster.par/scan-exclusive)
        'out 'acc 0.0 'i bound (if (= dtype :float) 'float 'double)
        '(+ acc (clojure.core/aget input i))))

(defn- expected-refusal [refusal scan]
  (assoc refusal :stage :typed-admission :sym 'result :source scan
                 :op (first scan) :target-dialect :segop :device :cpu:0
                 :dtype :double :fallback :none))

(defn- reject-legacy-scans [f]
  (let [original soac/par-form->soac]
    (with-redefs [soac/par-form->soac
                  (fn [& arguments]
                    (when (contains? '#{raster.par/scan raster.par/scan-exclusive}
                                     (first (second arguments)))
                      (throw (AssertionError. "admitted scan reached legacy semantic inference")))
                    (apply original arguments))]
      (f))))

(deftest typed-scan-refusals-do-not-retry-legacy-conversion
  (reject-legacy-scans
   (fn []
     (doseq [mode [:inclusive :exclusive]]
       (let [scan (apply list (assoc (vec (scan-form mode :double 'n))
                                    7 '(Math/tanh (+ (* weight acc) (clojure.core/aget input i)))))
             source (list 'let* ['before '(swap! events conj :before)
                                'result scan 'after '(swap! events conj :after)] 'result)
             expected (:declined (route/attempt (list 'let* ['result scan] 'result)
                                                :double {'input :double 'out :double}
                                                {:scalar-types {'n :long 'weight :double}}))
             result (lower/segop-lower-pass source
                                           (assoc (options :double) :scalar-types
                                                  {'n :long 'weight :double}))
             packet (:form result)
             declines (get-in result [:stats :segops-declined])]
         (is (= :scan-not-associative (:reason expected)))
         (is (= [(expected-refusal expected scan)] declines))
         (is (empty? (:equations packet)))
         (is (zero? (get-in result [:stats :kernel-graphs-lowered])))
         (is (= source (:source packet)))
         ;; Zero iterations exercise the unchanged primal/effect path without needing
         ;; the unsupported recurrence's scheduling or derivative semantics.
         (let [events (atom []) out (double-array 1)
               execute (eval (list 'fn ['input 'out 'n 'weight 'events] (:source packet)))]
           (is (identical? out (execute (double-array 0) out 0 0.5 events)))
           (is (= [:before :after] @events))))))))

(deftest scan-decline-boundary-preserves-options-and-fail-fast-exceptions
  (doseq [mode [:inclusive :exclusive]
          source-ns [nil (the-ns 'raster.compiler.compatibility-scan-packet-test)]]
    (let [scan (scan-form mode :double 'n)
          source (list 'let* ['result scan] 'result)
          opts (cond-> (options :double) source-ns (assoc :source-ns source-ns))
          refusal {:reason :typed-soac-unbound-scalar :details {:value 'n}}
          observed (atom nil)]
      (reject-legacy-scans
       (fn []
         (with-redefs [route/attempt (fn [form dt arrays options]
                                      (reset! observed [form dt arrays options])
                                      {:declined refusal})]
           (let [result (lower/segop-lower-pass source opts)]
             (is (= [(expected-refusal refusal scan)]
                    (get-in result [:stats :segops-declined])))
             (is (= source (get-in result [:form :source])))))))
      (is (= [source :double (:array-types opts)
              {:scalar-types (:scalar-types opts) :source-ns source-ns :values (:values opts)
               :resident-reductions? false}] @observed))
      (doseq [failure [(ex-info "compiler invariant" {:reason :raster/bug})
                       (NullPointerException. "unexpected analysis failure")]]
        (with-redefs [route/attempt (fn [& _] (throw failure))]
          (is (identical? failure
                          (try (lower/segop-lower-pass source opts) nil
                               (catch Throwable actual actual)))))))))

(deftest partial-literal-scans-keep-padded-storage-and-concrete-elements
  (reject-legacy-scans
   (fn []
     (doseq [mode [:inclusive :exclusive]
             element ['(clojure.core/aget input i)
                      '(.invk raster.numeric/_star__m_double_double-impl
                              (clojure.core/aget input i) (clojure.core/aget input i))]]
       (let [scan (apply list (assoc (vec (scan-form mode :double 2)) 7 (list '+ 'acc element)))
             source (list 'let* ['result scan] 'result)
             opts (-> (options :double)
                      (assoc :scalar-types {})
                      (assoc-in [:values 'input :shape] [4])
                      (assoc-in [:values 'out :shape] [6]))
             expected (route/attempt source :double (:array-types opts)
                                     {:scalar-types {} :values (:values opts)})
             result (lower/segop-lower-pass source opts)
             packet (:form result)]
         (if-let [refusal (:declined expected)]
           (do (is (= [(expected-refusal refusal scan)]
                      (get-in result [:stats :segops-declined])))
               (is (empty? (:equations packet))))
           (do (is (some? (:program expected)))
               (is (= 1 (get-in result [:stats :typed-soac-reused])))
               (is (= [4] (get-in packet [:values 'input :shape])))
               (is (= [6] (get-in packet [:values 'out :shape])))))
         (doseq [form [source (:source packet)]]
           (let [input (double-array [2.0 3.0 100.0 200.0])
                 out (double-array (repeat 6 -9.0))
                 execute (eval (list 'fn ['input 'out] form))
                 prefixes (if (= element '(clojure.core/aget input i)) [0.0 2.0 5.0] [0.0 4.0 13.0])
                 expected-prefix (if (= mode :inclusive) (rest prefixes) prefixes)]
             (is (identical? out (execute input out)))
             (is (= (vec (concat expected-prefix (repeat (- 6 (count expected-prefix)) -9.0)))
                    (vec out)))
             (is (= [2.0 3.0 100.0 200.0] (vec input))))))))))

(deftest non-scan-compatibility-fallback-is-unchanged
  (let [calls (atom []) original soac/par-form->soac
        scatter '(raster.par/scatter! out indices input n)]
    (with-redefs [soac/par-form->soac (fn [& args]
                                      (swap! calls conj (second args))
                                      (apply original args))]
      (let [result (lower/segop-lower-pass (list 'let* ['result scatter] 'result) (options :double))]
        (is (= [scatter] @calls))
        (is (some #(= :soac (:stage %)) (get-in result [:stats :segops-declined])))))))

(deftest scans-import-complete-typed-packets-and-caller-capacities
  (reject-legacy-scans
   (fn []
     (doseq [mode [:inclusive :exclusive] dtype [:float :double]
             bound ['n '(clojure.core/+ n 1)]]
       (testing [mode dtype bound]
         (let [result (lower/segop-lower-pass
                       (list 'let* ['result (scan-form mode dtype bound)] 'result) (options dtype))
               packet (:form result)
               equations (:equations packet)
               scan (last equations)
               graph (get-in scan [:attributes :kernel-graph])]
           (is (= (if (symbol? bound) 1 2) (count equations)))
           (is (= 1 (get-in result [:stats :typed-soac-reused])))
           (is (= 1 (get-in result [:stats :kernel-graphs-lowered])))
           (is (every? :algorithm equations))
           (is (every? :physical-algorithm equations))
           (is (identical? packet (program/validate! packet)))
           (is (= [33] (get-in packet [:values 'input :shape])))
           (is (= [34] (get-in packet [:values 'out :shape])))
           (is (seq (:nodes graph)))
           (is (= ['out] (mapv :id (:outputs graph))))
           (is (= ['input] (mapv :id (:inputs graph))))
           (is (= [:long] (mapv :dtype (:scalars graph)))
               "shape-only extents retain their integral scalar ABI")
           (is (= mode (get-in graph [:attributes :scan-mode])))
           (doseq [equation equations]
             (is (= (:operands equation) (:inputs (dialect/facts (:algorithm equation)))))
             (is (= (:results equation) (dialect/outputs (:algorithm equation))))
             (is (= equation (program/equation-for-binding
                              packet (second (:site equation)) (:source equation)))))))))))

(deftest scan-host-prefixes-preserve-earlier-effects-and-result-identity
  (reject-legacy-scans
   (fn []
     (doseq [mode [:inclusive :exclusive] dtype [:float :double]]
       (let [scan (scan-form mode dtype '(clojure.core/alength input))
             source (list 'let* [] '(swap! events conj :before) scan)
             packet (:form (lower/segop-lower-pass source (dissoc (options dtype) :values)))
             original (eval (list 'fn ['input 'out 'events] source))
             transformed (eval (list 'fn ['input 'out 'events] (:source packet)))
             primary (last (:equations packet))]
         (is (= [:body 1] (get-in primary [:provenance :compatibility-source-site])))
         (is (= '(swap! events conj :before) (nth (second (:source packet)) 1)))
         (doseq [function [original transformed] n [0 1 17 33] changed? [false true]]
           (let [xs (mapv #(if changed? (* 0.25 (- (mod % 7) 3)) 1.0) (range n))
                 input (if (= dtype :float) (float-array xs) (double-array xs))
                 out (if (= dtype :float) (float-array (inc n)) (double-array (inc n)))
                 events (atom [])
                 prefixes (reductions + 0.0 xs)
                 expected (if (= mode :inclusive) (rest prefixes) prefixes)
                 result (function input out events)]
             (is (identical? out result))
             (is (= [:before] @events))
             (is (= (vec expected) (mapv double (take (count expected) out)))))))))))

(deftest repeated-scans-freshen-prefixes-and-retain-exact-source-sites
  (let [scan (scan-form :inclusive :double '(clojure.core/alength input))
        source (list 'let* ['rstr_extent_0 7] scan scan)
        packet (:form (lower/segop-lower-pass source (dissoc (options :double) :values)))
        scans (remove #(get-in % [:attributes :host-only]) (:equations packet))
        prefixes (filter #(get-in % [:attributes :host-only]) (:equations packet))
        ids (mapcat #(dialect/outputs (:physical-algorithm %)) prefixes)]
    (is (= 2 (count scans)))
    (is (= [[:body 0] [:body 1]]
           (mapv #(get-in % [:provenance :compatibility-source-site]) scans)))
    (is (= 1 (count (distinct ids)))
        "the retained backing extent is reused rather than inferred twice")
    (is (= (mapv :operands scans) [(-> scans first :operands) (-> scans first :operands)]))
    (is (not-any? #{'rstr_extent_0} ids))
    (doseq [equation scans]
      (is (= equation (program/equation-for-binding
                       packet (second (:site equation)) (:source equation)))))
    (is (identical? packet (program/validate! packet)))))

(deftest repeated-compound-scans-retain-shared-backing-and-distinct-capacities
  (let [opts (-> (options :double)
                 (assoc-in [:array-types 'second-out] :double)
                 (assoc-in [:values 'second-out] (value/tensor {:dtype :double :shape [40]})))
        first-scan (scan-form :inclusive :double '(clojure.core/+ n 1))
        second-scan (apply list (assoc (vec (scan-form :exclusive :double '(clojure.core/+ n 2)))
                                      1 'second-out))
        packet (:form (lower/segop-lower-pass
                       (list 'let* ['first-result first-scan 'second-result second-scan]
                             'second-result) opts))
        scans (remove #(get-in % [:attributes :host-only]) (:equations packet))
        prefixes (filter #(get-in % [:attributes :host-only]) (:equations packet))]
    (is (= 2 (count scans)))
    (is (= 2 (count prefixes)))
    (is (= 2 (count (distinct (mapcat :results prefixes)))))
    (is (= [33] (get-in packet [:values 'input :shape])))
    (is (= [34] (get-in packet [:values 'out :shape])))
    (is (= [40] (get-in packet [:values 'second-out :shape])))
    (is (= [['out] ['second-out]]
           (mapv #(mapv :id (get-in % [:attributes :kernel-graph :outputs])) scans)))
    (is (identical? packet (program/validate! packet)))))

(deftest dynamic-empty-scans-keep-logical-storage-separate-from-physical-launch
  (doseq [mode [:inclusive :exclusive] dtype [:float :double]]
    (let [packet (:form (lower/segop-lower-pass
                         (list 'let* ['result (scan-form mode dtype 'n)] 'result)
                         (dissoc (options dtype) :values)))
          source (get-in (last (:equations packet)) [:attributes :kernel-graph])
          emitted (emit/generate-scan-kernel-graph source)
          intra-body (get-in emitted [:nodes 0 :operation :attributes :kernel-body])
          totals-id (:id (first (:temporaries source)))
          total-accesses (filter #(or (= totals-id (:buffer %))
                                     (= 'scan-block-total (get-in % [:result :id])))
                                 (:operations intra-body))]
      (is (= 3 (count (:nodes source))) "runtime extent retains the three-stage schedule")
      (is (= 0 (launch/resolve-expression (constantly 0)
                                          (:elements (first (:temporaries source)))))
          "empty input has no phantom logical block total")
      (is (= 2 (count total-accesses)))
      (is (every? #(= :scan-nonempty-total (:predicate %)) total-accesses)
          "negative scratch index and zero-capacity totals write are both guarded")
      (doseq [n [0 1 17 1025] node (:nodes emitted)]
        (let [geometry (launch/realize (get-in node [:operation :launch]) (constantly n))]
          (is (every? pos? (:group-count geometry)))))
      (is (= :source-grid
             (try
               (emit/generate-scan-kernel-graph
                (assoc-in source [:nodes 0 :operation :grid :num-blocks] 0))
               nil
               (catch clojure.lang.ExceptionInfo failure (:missing-rule (ex-data failure)))))))))

(deftest mixed-unsupported-source-does-not-demote-an-admitted-scan
  (let [source (list 'let* ['unsupported '(raster.par/scatter! out indices values n)
                           'result (scan-form :inclusive :double 'n)] 'result)
        opts (options :double)
        whole (route/attempt source :double (:array-types opts)
                             {:scalar-types (:scalar-types opts) :values (:values opts)})]
    (is (some? (:declined whole)))
    (reject-legacy-scans
     (fn []
       (let [result (lower/segop-lower-pass source opts)]
         (is (= 1 (get-in result [:stats :typed-soac-reused])))
         (is (seq (get-in result [:stats :segops-declined])))
         (is (= 1 (count (:equations (:form result))))))))))

(deftest forged-scan-packets-cannot-recertify-their-source
  (let [original route/attempt
        source (list 'let* ['result (scan-form :inclusive :double 'n)] 'result)]
    (doseq [change [(fn [_] {})
                    #(assoc % :source '(let* [missing (different-source)] missing))
                    #(update % :equations
                             (fn [equations]
                               (mapv (fn [equation] (assoc equation :source '(different-source)))
                                     equations)))]]
      (with-redefs [route/attempt (fn [& arguments]
                                  (update (apply original arguments) :program change))]
        (is (= :raster/bug
               (try (lower/segop-lower-pass source (options :double)) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))))

(deftest scheduled-public-storage-requires-a-retained-value-contract
  (let [source (list 'let* ['result (scan-form :inclusive :double 'n)] 'result)
        opts (options :double)
        typed (:program (route/attempt source :double (:array-types opts)
                                       {:scalar-types (:scalar-types opts) :values (:values opts)}))
        scheduled (:form (lower/segop-lower-pass typed opts))]
    (doseq [missing ['input 'out]]
      (is (= :typed-packet-graph-values
             (try ((var lower/packet-equations)
                   {:program (update scheduled :values dissoc missing)} [:binding 'result] (atom -1))
                  nil
                  (catch clojure.lang.ExceptionInfo failure (:stage (ex-data failure)))))
          "a graph declaration cannot replace the missing independent input/destination proof"))))
