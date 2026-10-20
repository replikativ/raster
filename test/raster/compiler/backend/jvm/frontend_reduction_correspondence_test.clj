(ns raster.compiler.backend.jvm.frontend-reduction-correspondence-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm source-sum [xs :- (Array double) count :- Long] :- Double
  (par/reduce acc 0.0 i count (+ acc (aget xs i))))

(deftm source-float-sum [xs :- (Array float) count :- Long] :- Float
  (par/reduce acc (float 0.0) i count (+ acc (aget xs i))))

(deftm cold-sum [xs :- (Array double) count :- Long] :- Double
  (par/reduce acc 0.0 i count (+ acc (aget xs i))))

(use-fixtures :each
  (fn [run]
    (binding [entry/*emission-inputs* (atom {:static [] :values []})]
      (run))))

(defn- receiver [source]
  (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl"))))

(defn- captured [function]
  (let [source (core/resolve-deftm-var function)]
    (core/ensure-current-owned-bytecode! (receiver source))
    (entry/original-source-frontend source)))

(defn- reduction-forms [body]
  (filter par-ir/par-reduce-form? (tree-seq coll? seq body)))

(deftest frozen-source-occurrence-reaches-only-its-own-emitted-reduction
  (doseq [[function payload] [[#'source-sum Double] [#'source-float-sum Float]]]
    (let [frontend (captured function)
          original (first (reduction-forms (:body frontend)))
          observed (entry/frontend-reduction-emission frontend original)
          emitted-source (first (vals (:reduction-catalog (:emission frontend))))]
      (is (some? observed))
      (is (identical? original (:source-input observed)))
      (is (not (identical? original (:source observed))))
      (is (= {:carrier Object :presence :present :builtin-class payload}
             (:result-fact observed)))
      (is (nil? (entry/frontend-reduction-emission
                 frontend (with-meta (apply list original) (meta original)))))
      (is (nil? (entry/frontend-reduction-emission frontend (assoc (vec original) 2 7.0))))
      (is (nil? (entry/frontend-reduction-emission frontend emitted-source))))))

(deftest input-body-and-output-body-identities-are-independent-fences
  (let [frontend (captured #'source-sum)
        original (first (reduction-forms (:body frontend)))
        emission (:emission frontend)]
    (is (nil? (bytecode/retained-input-reduction-emission
               emission (with-meta (vec (map identity (:body frontend))) (meta (:body frontend))) original)))
    (is (nil? (bytecode/retained-input-reduction-emission
               (assoc emission :body (with-meta (vec (map identity (:body emission))) (meta (:body emission))))
               (:body frontend) original)))
    (is (nil? (bytecode/retained-input-reduction-emission
               (dissoc emission :reduction-input-origins) (:body frontend) original)))
    (is (nil? (bytecode/retained-input-reduction-emission
               (assoc emission :reduction-input-origins {}) (:body frontend) original)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (entry/frontend-reduction-emission
                  (assoc frontend :body (vec (map identity (:body frontend)))) original)))
    (is (nil? (entry/frontend-reduction-emission
               (assoc frontend :emission (assoc emission :implementation (Object.))) original)))
    (is (nil? (entry/frontend-reduction-emission
               (assoc frontend :emission (assoc emission :frontend-source-owner (Object.))) original)))
    (let [changed (into {} emission)]
      (is (= emission changed))
      (is (not (identical? emission changed)))
      (is (nil? (entry/frontend-reduction-emission (assoc frontend :emission changed) original))))
    (is (nil? (binding [entry/*emission-inputs* nil]
                (entry/frontend-reduction-emission frontend original))))))

(deftest equal-distinct-occurrences-and-repeated-object-occurrences-do-not-collapse
  ;; Isolate the identity relation from physical emission. Actual emission and
  ;; source-owner guards are exercised by the preceding and following tests.
  (let [left '(raster.par/reduce acc 0.0 i count (+ acc i))
        right (apply list left)
        body [left right]
        owned (bytecode/retain-invocation-occurrences body)
        emission {:body (:body owned) :reduction-input-origins (:reduction-input-origins owned)}]
    (is (= left right))
    (is (not (identical? left right)))
    (with-redefs [bytecode/retained-reduction-emission (fn [_ source] {:source source})]
      (let [left-result (bytecode/retained-input-reduction-emission emission body left)
            right-result (bytecode/retained-input-reduction-emission emission body right)]
        (is (identical? left (:source-input left-result)))
        (is (identical? right (:source-input right-result)))
        (is (not (identical? (:source left-result) (:source right-result)))))
      (let [ambiguous-body [left left]
            ambiguous (bytecode/retain-invocation-occurrences ambiguous-body)]
        (is (nil? (bytecode/retained-input-reduction-emission
                   {:body (:body ambiguous) :reduction-input-origins (:reduction-input-origins ambiguous)}
                   ambiguous-body left)))))))

(deftest bodies-without-reductions-retain-no-input-origin-owner
  (is (nil? (:reduction-input-origins
             (bytecode/retain-invocation-occurrences '[(+ x 1.0)])))))

(deftest edited-observations-cannot-acquire-the-original-frontend-authority
  (let [frontend (captured #'source-sum)
        original (first (reduction-forms (:body frontend)))
        emission (:emission frontend)
        observed (entry/frontend-reduction-emission frontend original)
        witness (get-in observed [:expansion :witness])
        carry (second (get-in observed [:loop :variables]))
        changed (-> emission
                    (assoc-in [:reduction-contexts [witness :loop] :locals carry :type] :float)
                    (assoc-in [:reduction-contexts [witness :recur] :targets 1 :type] :float)
                    (assoc-in [:reduction-contexts [witness :exit] :local :type] :float)
                    (assoc-in [:reduction-contexts [witness :exit] :input-type] :float))]
    ;; The low-level observer assumes emitter-owned records. The public owner
    ;; boundary must authenticate the ENTIRE map, not just its opaque bodies.
    (is (= Float (get-in (bytecode/retained-input-reduction-emission
                          changed (:body frontend) original) [:result-fact :builtin-class])))
    (is (nil? (entry/frontend-reduction-emission (assoc frontend :emission changed) original)))
    (is (= Double (get-in (entry/frontend-reduction-emission frontend original)
                         [:result-fact :builtin-class])))))

(deftest cold-query-does-not-materialize-a-jvm-emission
  (let [source (core/resolve-deftm-var #'cold-sum)
        implementation (receiver source)
        _ (core/prepare-deftm-frontend! source)
        before (var-get implementation)
        frontend (entry/original-source-frontend source)
        original (first (reduction-forms (:body frontend)))]
    (is (some? (:raster.core/jit-spec (meta implementation))))
    (is (nil? (:emission frontend)))
    (is (nil? (entry/frontend-reduction-emission frontend original)))
    (is (identical? before (var-get implementation)))
    (is (nil? (:raster.core/bytecode-emission (meta implementation))))))

(deftest source-owner-staleness-is-not-repaired-by-the-origin-relation
  (let [frontend (captured #'source-sum)
        original (first (reduction-forms (:body frontend)))
        source (get-in frontend [:source-entry :source-var])
        metadata (meta source)]
    (try
      (alter-meta! source assoc :raster.core/source-policy {:unchecked-math true})
      (is (thrown? clojure.lang.ExceptionInfo
                   (entry/frontend-reduction-emission frontend original)))
      (finally (reset-meta! source metadata)))
    (is (some? (entry/frontend-reduction-emission frontend original)))))
