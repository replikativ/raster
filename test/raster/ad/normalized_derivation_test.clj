(ns raster.ad.normalized-derivation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-scalar-test :as scalar-test]
            [raster.ad.normalized-scalar-public-test :as public]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-map-sum-test :as map-sum]
            [raster.ad.normalized-indexed-map-test :as indexed]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.reverse :as reverse]
            [raster.ad.templates :as templates]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]))

(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- recertify [b] (assoc b :certificate (#'normalized/certificate b)))
(defn- build []
  (normalized/linearize-program
   (#'scalar-test/linear-program '(raster.numeric/* coefficient t) '[coefficient t])
   {:params '[coefficient t] :tags '[double double] :active-inputs '[t]
    :source-ns (the-ns 'raster.ad.normalized-derivation-test)}))
(defn- emit [body]
  (binding [*ns* (:source-ns body)]
    (eval (list 'fn (:params body) (first (:walked-body body))))))
(defn- scale-result [program]
  (let [values (:values (soac/facts program))
        inputs (select-keys values (:inputs (soac/facts program)))
        bs (#'normalized/bindings program)
        [result expression] (take-last 2 bs)
        dt (:dtype (get values result))
        intermediate (with-meta 'wrong-derivative-result {:raster.type/tag (dtype/scalar-tag-for-dtype dt)})
        two (if (= dt :float) (float 2.0) 2.0)]
    (#'normalized/program inputs
                          (into (vec (drop-last 2 bs))
                                [intermediate expression result (list 'raster.numeric/* intermediate two)])
                          (soac/outputs program))))

(deftest recorded-typed-generation-preserves-public-float-and-double-oracles
  (doseq [[v wrt arguments tangent expected expected-jvp]
          [[#'public/square nil [3.0] [0.5] [9.0 6.0] [9.0 3.0]]
           [#'public/float-affine [0] [(float 2.0) (float 3.0)] [(float 0.5)]
            [(float 8.0) (float 4.0) nil] [(float 8.0) (float 2.0)]]]]
    (let [b (normalized/prepare (reverse/resolve-deftm-var v) wrt) stream (:generation b)
          vg (emit (normalized/value+grad-body b)) jvp (emit (normalized/jvp-body b))]
      (is (identical? b (normalized/validate! b)))
      (is (= expected (apply vg arguments)))
      (is (= expected-jvp (apply jvp (concat arguments tangent))))
      (is (= (count stream) (count (distinct (map :id stream)))))
      (is (every? #(contains? #{:linear :transpose} (:phase %)) stream))
      (is (not-any? (set (keys (:values (soac/facts (:primal b))))) (map :id stream)))))
  (let [b (normalized/prepare (reverse/resolve-deftm-var #'public/repeated-affine) nil)]
    (is (< 1 (count (:residuals b))))
    (is (= :ad-normalized-derivation-correspondence
           (reason #(normalized/validate! (recertify (update b :residuals (comp vec reverse)))))))))

(deftest recertified-linear-arithmetic-and-transpose-arithmetic-are-not-authority
  (let [b (build)]
    (doseq [phase [:linear :transpose]]
      (let [changed (update b phase scale-result)]
        ;; These remain well-typed homogeneous LINEAR graphs. The new rule correspondence,
        ;; not an existing nonlinear/affine/type check, must reject the changed arithmetic.
        (is (set? (normalized/validate-linear! (get changed phase)
                                              (if (= phase :linear) (:tangent-inputs b) [(:cotangent-seed b)])
                                              (:residuals b))))
        (is (= :ad-normalized-derivation-correspondence
               (reason #(normalized/validate! (recertify changed))))))))
  (doseq [dt [:float :double] phase [:linear :transpose]]
    (let [b (normalized/linearize-map-program (#'element/map-program dt)
                                           (the-ns 'raster.ad.normalized-derivation-test))
        scalar-path [:map-programs :proof :nodes 0 :scalar-derivation]
        changed (update-in b (conj scalar-path phase) scale-result)
        scalar (get-in changed scalar-path)
        scalar (assoc scalar :certificate (#'normalized/certificate scalar))
        changed (assoc-in changed scalar-path scalar)
        interface (get-in changed [:map-programs :interface])
        equation-ids (get-in changed [:map-programs :proof :nodes 0 :equation-ids])
        ;; Rebuild matching map programs as well, defeating stage reconstruction alone.
        changed (update changed :map-programs
                        merge
                        (#'normalized/construct-map-stage
                         scalar (get-in b [:map-programs :source]) interface equation-ids))
        changed (recertify changed)
        envelope (-> (canonical/adapt-map b)
                     (merge (select-keys (:map-programs changed) [:source :primal :linear :transpose]))
                     (assoc-in [:derivation :bundle] changed))
        envelope (assoc envelope :certificate (#'canonical/certificate envelope))]
    ;; The shared node replay now owns this correspondence directly; there is no parallel
    ;; proof-less map-program validator that can bless the forged phases first.
    (is (= :ad-normalized-derivation-correspondence (reason #(normalized/validate! changed))))
    ;; Recomputed inner AND outer certificates cannot authorize a matching forged map lift.
    (is (= :ad-normalized-derivation-correspondence (reason #(canonical/validate! envelope)))))))

(deftest recertified-inner-scalar-arithmetic-is-not-sum-authority
  (let [bundle (#'map-sum/build (#'map-sum/source :double) 'xs)
        scalar-path [:map-programs :proof :nodes 0 :scalar-derivation]
        changed (update-in bundle (conj scalar-path :linear) scale-result)
        scalar (get-in changed scalar-path)
        scalar (assoc scalar :certificate (#'normalized/certificate scalar))
        changed (assoc-in changed scalar-path scalar)
        changed (update changed :map-programs merge
                        (#'normalized/construct-map-stage
                         scalar (get-in changed [:map-programs :source])
                         (get-in changed [:map-programs :interface])
                         (get-in changed [:map-programs :proof :nodes 0 :equation-ids])))
        changed (recertify changed)]
    (is (= :ad-normalized-derivation-correspondence
           (reason #(normalized/validate! changed))))))

(deftest recertified-inner-scalar-arithmetic-is-not-indexed-authority
  (let [bundle (normalized/linearize-indexed-map-program
                (#'indexed/indexed-map :double :long 'n 'm)
                (the-ns 'raster.ad.normalized-derivation-test))
        scalar-path [:gather-programs :proof :scalar-derivation]
        changed (update-in bundle (conj scalar-path :linear) scale-result)
        scalar (get-in changed scalar-path)
        changed (assoc-in changed scalar-path
                          (assoc scalar :certificate (#'normalized/certificate scalar)))
        changed (recertify changed)]
    (is (= :ad-normalized-derivation-correspondence
           (reason #(normalized/validate! changed))))))

(deftest malformed-generation-never-falls-back-to-fresh-allocation
  (let [b (build) stream (:generation b) first-entry (first stream)
        source-id (first (:params b))
        own-meta (meta (:id first-entry))]
    (doseq [change [#(dissoc % :generation)
                    #(assoc % :generation nil) #(assoc % :generation {}) #(assoc % :generation 1)
                    #(assoc % :generation [1]) #(assoc % :generation [nil])
                    #(assoc-in % [:generation 0 :id] "not-a-symbol")
                    #(assoc-in % [:generation 0 :id] (with-meta source-id own-meta))
                    #(assoc-in % [:generation 0 :id] (with-meta 'foreign/id own-meta))
                    #(assoc-in % [:generation 0 :id] (vary-meta (:id first-entry) assoc :raster.op/original 'forged/op))
                    #(assoc-in % [:generation 0 :requested-tag] 'float)
                    #(assoc-in % [:generation 0 :phase] :other)
                    #(assoc-in % [:generation 0 :prefix] "wrong-request")
                    #(assoc % :generation (assoc stream 1 first-entry))
                    #(assoc % :generation (vec (butlast stream)))
                    #(assoc % :generation (conj stream (assoc (last stream) :id (with-meta 'unused-extra-id own-meta))))]]
      (is (= :ad-normalized-generation-stream (reason #(normalized/validate! (recertify (change b)))))))))

(deftest consistent-generated-id-renaming-is-not-a-spelling-authority
  (let [b (build) stream (:generation b)
        renamed (-> stream (assoc-in [0 :id] (:id (second stream))) (assoc-in [1 :id] (:id (first stream))))
        position (atom 0)
        fresh (fn [] (#'normalized/typed-gensym
                     :double (fn [_ _] (let [id (:id (nth renamed @position))] (swap! position inc) id))))
        regenerated (#'normalized/linearize (:primal b) (:active-inputs b) (fresh))
        transposed (#'normalized/transpose (:program regenerated) (:tangent-inputs regenerated)
                                           (:coefficients regenerated) (fresh))
        changed (merge b {:generation renamed :linear (:program regenerated) :transpose (:program transposed)
                         :tangent-inputs (:tangent-inputs regenerated) :residuals (:coefficients regenerated)
                         :cotangent-seed (:seed transposed)})
        certified (recertify changed)]
    (is (= (count stream) @position))
    (is (identical? certified (normalized/validate! certified)))
    (is (= [6.0 nil 3.0] ((emit (normalized/value+grad-body certified)) 3.0 2.0)))))

(deftest replay-cost-and-epoch-drift-are-explicit-without-recursive-validation
  (let [b (build) linearize (ns-resolve 'raster.ad.normalized-scalar 'linearize)
        transpose (ns-resolve 'raster.ad.normalized-scalar 'transpose)
        lin @linearize trans @transpose calls (atom []) regenerated (atom nil)]
    (with-redefs-fn
      {linearize (fn [& xs] (swap! calls conj :linear) (let [result (apply lin xs)] (reset! regenerated (:program result)) result))
       transpose (fn [p & xs] (swap! calls conj :transpose) (is (identical? @regenerated p)) (apply trans p xs))}
      (fn [] (normalized/validate! b) (normalized/validate! b)))
    (is (= [:linear :transpose :linear :transpose] @calls))
    (let [revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
          current (revisions) reads (atom 0)]
      ;; The entry sees the old epoch; every later observation sees a stable new epoch.
      ;; Comparing only replay's before/after would incorrectly accept unchanged arithmetic.
      (with-redefs-fn {revisions (fn [] (if (= 1 (swap! reads inc)) current
                                        (update current :template-revision inc)))}
        (fn [] (is (= :ad-normalized-stale-rules (reason #(normalized/validate! b)))))))
    (let [revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions) current (revisions) drift (atom false)]
      (with-redefs-fn {revisions (fn [] (cond-> current @drift (update :template-revision inc)))
                      linearize (fn [& xs] (let [result (apply lin xs)] (reset! drift true) result))}
        (fn [] (is (= :ad-normalized-stale-rules (reason #(normalized/validate! b)))))))
    (let [primary (ex-info "rule callback failed" {:reason :review-rule-failure})]
      (with-redefs-fn {linearize (fn [& _] (throw primary))}
        (fn [] (is (identical? primary (try (normalized/validate! b) nil (catch Exception e e)))))))))

(deftest actual-registered-jvp-callback-replays-identical-state-and-typed-requests
  (let [actual templates/op-jvp-fn observations (atom [])]
    (with-redefs
      [templates/op-jvp-fn
       (fn [operation]
         (when-let [callback (actual operation)]
           (fn [ctx args seeds result fresh]
             (let [requests (atom [])
                   wrapped (fn owner
                             ([prefix] (owner prefix nil))
                             ([prefix requested-tag]
                              (let [id (fresh prefix requested-tag)]
                                (swap! requests conj {:prefix prefix :requested-tag requested-tag :id id}) id)))
                   [emitted tangent] (callback (assoc ctx :gensym-fn wrapped) args seeds result wrapped)]
               (swap! observations conj {:operation operation :args args :seeds seeds :result result
                                         :requests @requests :bindings (:bindings emitted) :tangent tangent})
               [emitted tangent]))))]
      (let [b (build)]
        (is (identical? b (normalized/validate! b)))
        ;; Canonical alias lookup is internal: each actual producer/replay emission
        ;; crosses the public callback once with identical state and typed requests.
        (is (= 3 (count @observations)))
        (is (= (vec (repeat 3 'raster.numeric/*))
               (mapv :operation @observations)))
        (is (seq (:requests (first @observations))))
        (is (apply = @observations))
        (is (apply = (map #(dissoc % :operation) @observations))))))
  (let [b (build) actual templates/op-jvp-fn]
    (with-redefs [templates/op-jvp-fn
                  (fn [operation]
                    (when-let [callback (actual operation)]
                      (fn [& args]
                        (let [[ctx tangent] (apply callback args)]
                          ;; Same IDs and typed linear carrier, but replace the true coefficient
                          ;; with a different Double constant. No registry or epoch is mutated.
                          [(update ctx :bindings #(util/subst-syms {'coefficient 2.0} %)) tangent]))))]
      (is (= :ad-normalized-derivation-correspondence (reason #(normalized/validate! b)))))))
