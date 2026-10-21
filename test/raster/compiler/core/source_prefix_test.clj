(ns raster.compiler.core.source-prefix-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.par :as par]
            [raster.compiler.core.source-prefix :as prefix]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.invocation-plan :as invocation]))

(deftm selected-prefix [n :- Long] :- (Array double)
  (let [size (numeric/* n 2)
        out (double-array size)]
    (par/map! out i size double 1.0)))

(deftm shadowed-prefix [n :- Long] :- (Array double)
  (let [n (numeric/* n 2)
        out (double-array n)]
    (par/map! out i n double 1.0)))

(defn- source-with-bindings [source bindings]
  (with-meta (list* (first source) bindings (nnext source)) (meta source)))

(defn- capture [function]
  (let [observed (atom {})
        begin prefix/begin close prefix/close
        target :ocl:source-prefix-proof
        descriptor {:device-id target :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}
        compilation
        (with-redefs [prefix/begin (fn [frontend parameters]
                                   (swap! observed assoc :frontend frontend)
                                   (begin frontend parameters))
                      prefix/close (fn [& arguments]
                                     (let [result (apply close arguments)]
                                       (swap! observed assoc :inputs arguments :correspondence result)
                                       result))]
          (equation/compile function {:target target :dtype :double} descriptor))]
    (assoc @observed :compilation compilation)))

(deftest original-occurrence-is-separate-from-logical-result-conversion
  (let [{:keys [frontend inputs correspondence compilation]} (capture #'selected-prefix)
        [input normalized represented program] inputs
        step (first (filter invocation/scalar-compute?
                            (get-in program [:attributes :invocation-plan :steps])))
        original (prefix/original-step correspondence frontend program step)]
    (is (some? input))
    (is (some? correspondence))
    (is (= 0 (:ordinal original)))
    (is (= 'size (:binding original)))
    (is (identical? (first (:body frontend)) (:scope original)))
    (is (identical? (second (second (:scope original))) (:expression original)))
    (is (= '.invk (first (:expression original))))
    (is (= 'clojure.core/long (first (:result-expression original))))
    (is (= (:expression original) (second (:result-expression original))))
    (is (identical? step (:step original)))
    (is (some? (prefix/close input normalized represented program)))
    (is (nil? (prefix/original-step correspondence frontend program nil)))
    (is (nil? (prefix/original-step correspondence frontend program (into {} step))))
    (is (nil? (prefix/original-step correspondence frontend (assoc program :copied true) step)))
    (is (nil? (prefix/original-step correspondence (into {} frontend) program step)))
    (let [failure (try (equation/lower compilation [1]) nil
                       (catch clojure.lang.ExceptionInfo error (ex-data error)))]
      (is (= :unclosed-ordered-source-executor (:reason failure)))
      (is (= (select-keys original [:ordinal :binding :expression]) (:original-source failure)))
      (is (identical? (:expression original) (get-in failure [:original-source :expression]))))))

(deftest history-and-original-frontend-identity-remain-required
  (let [{:keys [frontend inputs correspondence]} (capture #'selected-prefix)
        [input normalized represented program] inputs
        step (first (filter invocation/scalar-compute?
                            (get-in program [:attributes :invocation-plan :steps])))
        source (get-in frontend [:source-entry :source-var])
        metadata (meta source)]
    (is (nil? (prefix/close nil normalized represented program)))
    (is (nil? (prefix/close {} normalized represented program)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (prefix/begin (assoc frontend :body (mapv identity (:body frontend))) '[n])))
    (try
      (alter-meta! source assoc :raster.core/source-policy {:unchecked-math true})
      (is (thrown? clojure.lang.ExceptionInfo
                   (prefix/close input normalized represented program)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (prefix/original-step correspondence frontend program step)))
      (finally (reset-meta! source metadata)))
    (is (some? (prefix/original-step correspondence frontend program step)))))

(deftest whole-prefix-must-survive-not-just-the-selected-call
  (let [{:keys [inputs]} (capture #'selected-prefix)
        [input normalized represented program] inputs
        pairs (second normalized)
        rewrites [(source-with-bindings normalized (vec (drop 2 pairs)))
                  (source-with-bindings normalized (vec (concat (take 2 pairs) pairs)))
                  (source-with-bindings normalized (vec (concat (drop 2 pairs) (take 2 pairs))))
                  ;; Same numerical product, different original operand order.
                  (source-with-bindings normalized (assoc pairs 1 (list '.invk (second (second pairs)) 2 'n)))
                  (source-with-bindings normalized (update pairs 1 vary-meta assoc :tag 'double))
                  (source-with-bindings normalized (update pairs 1 vary-meta assoc :raster.effect/preserve-evaluation true))
                  (source-with-bindings normalized (assoc pairs 1 '(foreign.namespace/producer n 2)))]]
    (doseq [changed rewrites]
      (is (nil? (prefix/close input changed represented program))))
    (doseq [changed rewrites]
      (is (nil? (prefix/close input normalized changed program))))))

(deftest result-materialization-remains-an-independently-validated-contract
  (let [{:keys [inputs]} (capture #'selected-prefix)
        [input normalized represented program] inputs
        expression (second (second (:source program)))]
    (doseq [replacement [(second expression) (list 'clojure.core/double (second expression))]
            :let [changed (update program :source
                                  #(source-with-bindings % (assoc (second %) 1 replacement)))]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (prefix/close input normalized represented changed))))))

(deftest shadowed-original-parameters-do-not-acquire-prefix-authority
  (let [{:keys [correspondence]} (capture #'shadowed-prefix)]
    (is (nil? correspondence))))

(deftest alpha-transport-checks-order-capture-literal-bits-and-declarations
  ;; Isolate the bounded transformation verifier; public tests above exercise
  ;; actual producers and the exact program/step/source-owner fences.
  (let [relation @(ns-resolve 'raster.compiler.core.source-prefix 'prefix-renaming)
        original '[[a (free-helper x)] [b (free-helper a)]]]
    (is (= '{x x a first b second}
           (relation original '[[first (free-helper x)] [second (free-helper first)]] '[x] {'x 'long})))
    (is (nil? (relation original '[[x (free-helper x)] [b (free-helper x)]] '[x] {'x 'long})))
    (is (nil? (relation original
                        '[[free-helper (free-helper x)] [b (free-helper free-helper)]] '[x] {'x 'long})))
    ;; A future binder is not in scope in an earlier initializer. Removing
    ;; every binder from the free set would incorrectly approve this capture.
    (is (nil? (relation '[[a 1] [b (helper c)] [c 2]]
                        '[[c 1] [b (helper c)] [d 2]] [] {})))
    (is (nil? (relation '[[a 1] [c (helper c)]]
                        '[[c 1] [d (helper c)]] [] {})))
    (is (some? (relation '[[a (helper a)] [b a]]
                         '[[a (helper a)] [result a]] [] {})))
    (is (nil? (relation '[[a -0.0]] '[[a 0.0]] [] {})))
    (is (nil? (relation [['a (long 1)]] [['a (int 1)]] [] {})))
    (let [typed (with-meta 'x {:raster.type/tag 'long :tag 'long})]
      (is (some? (relation '[[a x]] [['a typed]] '[x] {'x 'long})))
      (is (nil? (relation '[[a x]] [['a typed]] '[x] {})))
      (is (nil? (relation '[[a x]] [['a typed]] '[x] {'x 'double}))))
    (let [compare-expression @(ns-resolve 'raster.compiler.core.source-prefix 'same-expression?)
          quoted '(quote (helper x))]
      (is (true? (compare-expression quoted quoted {'x 'renamed} {})))
      (is (false? (compare-expression quoted (apply list quoted) {} {})))
      (is (not (compare-expression '(unknown-factory x)
                                   (with-meta '(unknown-factory x) {:tag 'doubles}) {} {}))))))
