(ns raster.compiler.core.source-entry-lineage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.dl.optim :as optim]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]))

(def body-calls (atom 0))

(defn- fresh-entry []
  ;; Repeated warm-REPL tests must still begin at a genuinely cold entry.
  (binding [*ns* (the-ns 'raster.compiler.core.source-entry-lineage-test)]
    (eval '(core/deftm entry-probe [values :- (Array double), count :- Long,
                                  scale :- Double, reference :- Object] :- Double
             (do (swap! body-calls inc) scale))))
  (let [operation 'raster.compiler.core.source-entry-lineage-test/entry-probe
        generic (find-var operation)
        method (first (get @(:raster.core/dispatch-table (meta generic)) 4))
        name (types/mangle 'entry-probe (:tags method))
        source (ns-resolve (:ns (meta generic)) name)
        implementation (symbol (str (:mangled-ns method)) (str name "-impl"))]
    {:operation operation :implementation implementation :source source
     :receiver (find-var implementation)}))

(defn- projection [{:keys [operation implementation source]}]
  (entry/call-with-definition-inputs
   #(entry/selected-source-entry-carriers operation implementation 4 source)))

(deftest cold-and-warm-entries-have-the-same-formal-facts-without-running-the-body
  (let [{:keys [receiver source] :as selected} (fresh-entry)
        before @body-calls
        cold (projection selected)]
    (is (some? (:raster.core/jit-spec (meta receiver))))
    (is (= :typed-source-body-entry (:scope cold)))
    (is (= {'values {:carrier (class (double-array 0)) :presence :unknown}
            'count {:carrier Long/TYPE :presence :present}
            'scale {:carrier Double/TYPE :presence :present}
            'reference {:carrier Object :presence :unknown}}
           (:parameter-facts cold)))
    (is (= Object (:result-carrier cold)))
    (is (= before @body-calls))
    (core/ensure-bytecode-implementation! receiver)
    (let [emission (:raster.core/bytecode-emission (meta receiver))
          warm (projection selected)]
      (is (nil? (:raster.core/jit-spec (meta receiver))))
      (is (entry/source-entry-current? (:source-entry emission)))
      (is (identical? source (get-in emission [:source-entry :source-var])))
      (is (identical? (:raster.core/deftm-source-body (meta source))
                      (get-in emission [:source-entry :source-body])))
      (is (= (:parameter-facts cold) (:parameter-facts warm)))
      (is (= Object (:result-carrier warm)))
      (is (= before @body-calls)))))

(deftest bytecode-publishes-source-walking-before-dependencies-are-sealed
  (let [{:keys [receiver source]} (fresh-entry)]
    (core/ensure-bytecode-implementation! receiver)
    (let [cache (:raster.core/deftm-walked-body (meta source))
          emission (:raster.core/bytecode-emission (meta receiver))
          before (meta source)]
      (is (seq cache))
      (is (= cache (core/ensure-walked-body! source)))
      (is (identical? cache (:raster.core/deftm-walked-body (meta source))))
      (is (= before (meta source)))
      (is (identical? emission (core/ensure-current-owned-bytecode! receiver)))
      ;; Legacy concrete producers without their genuine walked output must
      ;; re-emit from raw source, never re-seal the old code's observations.
      (alter-meta! source dissoc :raster.core/deftm-walked-body)
      (let [fresh (core/ensure-current-owned-bytecode! receiver)]
        (is (seq (:raster.core/deftm-walked-body (meta source))))
        (is (not (identical? emission fresh)))
        (is (not (identical? (:implementation emission) (:implementation fresh))))
        (is (entry/source-entry-current? (:source-entry fresh)))
        (let [populated (:raster.core/deftm-walked-body (meta source))
              lineage (:source-entry fresh)]
          (core/do-bytecode-upgrade! receiver (:source-name lineage) (:params lineage)
                                    (:tags lineage) (:return-tag lineage) nil
                                    (:source-ns lineage) (:typed-interface lineage)
                                    :annotations (:annotations lineage) :source-body (:source-body lineage)
                                    :source-policy (:source-policy lineage))
          (is (identical? populated (:raster.core/deftm-walked-body (meta source)))))))))

(deftest frozen-interface-spelling-is-nominal-not-a-new-physical-proof
  (doseq [spelling [identity symbol]]
    (let [{:keys [receiver source]} (fresh-entry)
          lineage (:source-entry (core/prepare-implementation-frontend! receiver))
          before @body-calls
          implementation @receiver
          pending (:raster.core/jit-spec (meta receiver))
          options [:annotations (:annotations lineage) :source-body (:source-body lineage)
                   :source-policy (:source-policy lineage)]
          upgrade! (fn [name]
                     (apply core/do-bytecode-upgrade! receiver (:source-name lineage)
                            (:params lineage) (:tags lineage) (:return-tag lineage) nil
                            (:source-ns lineage) name options))]
      (is (string? (:typed-interface lineage)))
      (is (= lineage (entry/capture-source-entry
                       source (update lineage :typed-interface symbol))))
      (is (= :frontend-emission-source-mismatch
             (try (upgrade! "unrelated.Interface") nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (identical? implementation @receiver))
      (is (identical? pending (:raster.core/jit-spec (meta receiver))))
      (upgrade! (spelling (:typed-interface lineage)))
      (let [emission (:raster.core/bytecode-emission (meta receiver))]
        (is (= (:typed-interface lineage) (str (:typed-interface emission))))
        (is (= lineage (:source-entry emission)))
        (is (nil? (:raster.core/jit-spec (meta receiver))))
        (is (identical? emission (core/ensure-current-owned-bytecode! receiver))))
      (is (= before @body-calls)))))

(deftest missing-or-mismatched-entry-lineage-abstains
  (let [{:keys [receiver source operation implementation] :as selected} (fresh-entry)
        original (meta receiver)]
    (binding [entry/*emission-inputs* nil]
      (is (nil? (entry/selected-source-entry-carriers operation implementation 4 source))))
    (is (nil? (projection (assoc selected :source #'body-calls))))
    (try
      (doseq [[field value] [[:mangled-sym 'missing]
                              [:params '[wrong]] [:tags '[double]] [:annotations '[Object]]
                              [:source-body '(nil)] [:source-ns-name 'missing]
                              [:source-policy {:unchecked-math true}]
                              [:typed-iface-name 'missing] [:return-tag 'Object]]]
        (reset-meta! receiver (assoc-in original [:raster.core/jit-spec field] value))
        (is (nil? (projection selected)) (pr-str field)))
      (reset-meta! receiver (dissoc original :raster.core/jit-spec :raster.core/bytecode-emission))
      (is (nil? (projection selected)))
      (finally (reset-meta! receiver original)))
    (core/ensure-bytecode-implementation! receiver)
    (let [warm (meta receiver)]
      (try
        (doseq [changed [(update warm :raster.core/bytecode-emission dissoc :source-entry)
                         (assoc-in warm [:raster.core/bytecode-emission :implementation] (Object.))
                         (assoc-in warm [:raster.core/bytecode-emission :typed-interface] 'missing)
                         (assoc-in warm [:raster.core/bytecode-emission :source-entry :source-var] #'body-calls)
                         (assoc-in warm [:raster.core/bytecode-emission :source-entry :source-policy]
                                   {:unchecked-math true})
                         (assoc-in warm [:raster.core/bytecode-emission :source-entry :source-body] '(nil))]]
          (reset-meta! receiver changed)
          (is (nil? (projection selected))))
        (finally (reset-meta! receiver warm))))))

(deftest original-source-identity-is-distinct-from-derived-walking-metadata
  (let [{:keys [receiver source] :as selected} (fresh-entry)]
    (core/ensure-bytecode-implementation! receiver)
    (let [lineage (get-in (meta receiver) [:raster.core/bytecode-emission :source-entry])
          original (meta source)]
      (is (some? lineage))
      (try
        (alter-meta! source assoc :raster.core/deftm-walked-body '[different-derived-form])
        (is (entry/source-entry-current? lineage))
        (is (some? (projection selected)))
        ;; Equal raw syntax with changed metadata is not the historical source object.
        (alter-meta! source assoc :raster.core/deftm-source-body
                     (with-meta (:raster.core/deftm-source-body original) {:changed true}))
        (is (false? (entry/source-entry-current? lineage)))
        (is (nil? (projection selected)))
        (finally (reset-meta! source original))))))

(deftest disabled-or-failed-jit-never-acquires-successful-source-lineage
  (let [{:keys [receiver] :as selected} (binding [core/*compile-deftm-bytecode?* false] (fresh-entry))]
    (is (nil? (:raster.core/jit-spec (meta receiver))))
    (is (nil? (projection selected))))
  (doseq [silent? [false true]]
    (let [{:keys [receiver source]} (fresh-entry)
          source-root @source receiver-root @receiver before @body-calls
          walks (atom 0)
          walker-var (ns-resolve 'raster.core 'jit-walk-with-tc)
          original @walker-var]
      (try
        (binding [core/*jit-silent-fallback?* silent?]
          (with-redefs-fn
            {walker-var (fn [& args]
                          (swap! walks inc)
                          (alter-var-root source (constantly (fn [& _] nil)))
                          (apply original args))}
            #(try (core/ensure-bytecode-implementation! receiver)
                  (catch clojure.lang.ExceptionInfo _ nil))))
        (is (identical? receiver-root @receiver))
        (is (= 1 @walks) "the mutation must reach the required walking boundary")
        (is (nil? (:raster.core/bytecode-emission (meta receiver))))
        (is (= before @body-calls))
        (finally (alter-var-root source (constantly source-root)))))))

(deftest parametric-source-publication-retains-the-original-entry
  (binding [*ns* (the-ns 'raster.compiler.core.source-entry-lineage-test)]
    (eval '(core/deftm entry-parametric
             (All [T] [values :- (Array T), count :- Long,
                       scale :- Double, reference :- Object] :- Double
               (do (swap! body-calls inc) scale)))))
  (let [operation 'raster.compiler.core.source-entry-lineage-test/entry-parametric
        generic (find-var operation)
        before @body-calls]
    (doseq [dtype [:float :double]
            :let [source (core/resolve-deftm-var generic {:dtype dtype})
                  implementation (symbol (str (:ns (meta source))) (str (:name (meta source)) "-impl"))
                  receiver (find-var implementation)
                  selected {:operation operation :implementation implementation :source source}
                  before-facts (projection selected)
                  ;; All[T]'s default Double specialization may be an ordinary
                  ;; lazy deftm, while requested Float is eagerly specialized.
                  _ (core/ensure-bytecode-implementation! receiver)
                  emitted (:raster.core/bytecode-emission (meta receiver))
                  facts (projection selected)]]
      (is (some? facts) (pr-str dtype))
      (is (= (:parameter-facts before-facts) (:parameter-facts facts)))
      (is (identical? @source (get-in emitted [:source-entry :source-root])))
      (is (entry/source-entry-current? (:source-entry emitted)))
      (is (= Double/TYPE (get-in facts [:parameter-facts 'scale :carrier])))
      (is (= :unknown (get-in facts [:parameter-facts 'values :presence])))
      (is (= before @body-calls)))))

(deftest actual-clipping-specialization-has-scoped-primitive-entry-evidence
  (doseq [dtype [:float :double]
          :let [source (core/resolve-deftm-var #'optim/clip-grad-norm! {:dtype dtype})
                implementation (symbol (str (:ns (meta source))) (str (:name (meta source)) "-impl"))
                projected (entry/call-with-definition-inputs
                           #(entry/selected-source-entry-carriers
                             'raster.dl.optim/clip-grad-norm! implementation 3 source))]]
    (is (some? projected) (pr-str dtype))
    (is (= :present (get-in projected [:parameter-facts 'max-norm :presence])))
    (is (= Double/TYPE (get-in projected [:parameter-facts 'max-norm :carrier])))
    (is (= :unknown (get-in projected [:parameter-facts 'grads :presence])))))
