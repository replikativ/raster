(ns raster.compiler.pending-frontend-source-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core]
            [raster.par :as par]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.backend.jvm.source-owner :as source-owner]
            [raster.compiler.core.method-entry :as entry]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.runtime.hardware :as hardware]))

(def target :ocl:pending-frontend-source-test)
(use-fixtures :each hardware-fixture/isolated
  (fn [f]
    (hardware/register-target-device!
     target {:name "Synthetic pending frontend target"
             :capabilities {:total-eus 32 :subgroup-sizes [16]
                            :max-workgroup-size 1024 :shared-local-memory 65536}})
    (f)))

(defmacro factor-expression [factor] `(* ~factor 2.0))
(core/deftm gpu-first [factor :- Double xs :- (Array float)] :- (Array float)
  (par/scale (factor-expression factor) xs))
(core/deftm concurrent-source [x :- Double] :- Double (+ x 2.0))
(core/deftm failed-source [x :- Double] :- Double (+ x 3.0))
(def helper-expansions (atom 0))
(defmacro counted-helper-expression [x]
  (swap! helper-expansions inc)
  `(+ ~x 4.0))
(core/deftm cold-helper [x :- Double] :- Double (counted-helper-expression x))

(defn- source [v] (core/resolve-deftm-var v {:dtype :double}))
(defn- receiver [v]
  (ns-resolve (:ns (meta v)) (symbol (str (:name (meta v)) "-impl"))))
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- frontend [v resolved dtype]
  (entry/call-with-definition-inputs #(pipeline/retained-frontend-source v resolved dtype)))

(deftest cold-helper-import-expands-once-and-jvm-reuses-it
  (let [v (source #'cold-helper) impl (receiver v)
        before @helper-expansions
        own-body (:raster.core/deftm-source-body (meta v))
        original-walk core/jit-walk-with-tc
        walks (atom 0)]
    (with-redefs [core/jit-walk-with-tc
                  (fn [body & args]
                    (when (identical? body own-body) (swap! walks inc))
                    (apply original-walk body args))]
      (is (some? (binding [inline/*inline-scalar-bodies?* true]
                   (entry/call-with-definition-inputs
                    #(#'inline/resolve-walked-body-input v)))))
      (is (= 1 @walks))
      ;; Type checking may itself inspect the macro expansion. Publication must
      ;; not expand it again, regardless of that frontend-internal count.
      (let [expanded @helper-expansions]
        (is (< before expanded))
        (core/ensure-bytecode-implementation! impl)
        (is (= 7.0 (cold-helper 3.0)))
        (is (= expanded @helper-expansions))
        (is (= 1 @walks))))))

(deftest gpu-first-freezes-source-without-numerical-invocation-or-jvm-emission
  (let [v (source #'gpu-first) impl (receiver v)
        original-macro @#'factor-expression
        original-implementation @impl
        original-walk core/jit-walk-with-tc
        own-body (:raster.core/deftm-source-body (meta v))
        walks (atom 0)]
    (try
      (with-redefs [core/jit-walk-with-tc
                    (fn [body & args]
                      (when (identical? body own-body) (swap! walks inc))
                      (apply original-walk body args))]
        (binding [entry/*bytecode-compiler* (fn [& _] (throw (AssertionError. "frontend emitted JVM code")))]
          (is (some? (core/prepare-deftm-frontend! v))))
        (is (identical? original-implementation @impl))
        (is (nil? (:raster.core/bytecode-emission (meta impl))))
        (let [before (entry/call-with-definition-inputs
                      #(pipeline/retained-frontend-source #'gpu-first v :double
                                                         {:preserve-declared-array-storage? true}))
              compiled (equation-first/compile #'gpu-first
                         {:target target :dtype :double :preserve-declared-array-storage? true})
              owner (:checked-result-source-owner compiled)]
          (is (some? owner))
          (alter-var-root #'factor-expression
                          (constantly (fn [_ _ factor] (list 'clojure.core/* factor 7.0))))
          (core/ensure-bytecode-implementation! impl)
          (is (= [(float 4.0)] (vec (gpu-first 2.0 (float-array [1.0])))))
          (is (= 1 @walks) "JVM publication reuses, rather than re-walks, the frontend")
          (is (= :stale-definition-proof-input
                 (reason #(source-owner/assert-checked-result-source!
                           owner (:semantic compiled) (:emitted compiled)))))
          (let [after (entry/call-with-definition-inputs
                       #(pipeline/retained-frontend-source #'gpu-first v :double
                                                          {:preserve-declared-array-storage? true}))
                recompiled (equation-first/compile #'gpu-first
                             {:target target :dtype :double :preserve-declared-array-storage? true})]
            (is (identical? (:source-frontend before) (:source-frontend after)))
            (is (identical? (:body before) (:body after)))
            (is (not (identical? (:source-owner before) (:source-owner after))))
            (is (some? (:checked-result-source-owner recompiled))))))
      (finally (alter-var-root #'factor-expression (constantly original-macro))))))

(deftest concurrent-target-requests-share-one-frontend-producer
  (let [v (source #'concurrent-source) impl (receiver v)
        own-body (:raster.core/deftm-source-body (meta v))
        original-walk core/jit-walk-with-tc
        walks (atom 0)
        start (promise)]
    (with-redefs [core/jit-walk-with-tc
                  (fn [body & args]
                    (when (identical? body own-body) (swap! walks inc))
                    (apply original-walk body args))]
      (let [a (future @start (core/prepare-deftm-frontend! v))
            b (future @start (core/ensure-bytecode-implementation! impl))]
        (deliver start true)
        (let [prepared @a]
          @b
          (is (= 1 @walks))
          (is (identical? (:source-frontend prepared)
                           (:source-frontend (frontend #'concurrent-source v :double))))
          (is (= 5.0 (concurrent-source 3.0))))))))

(deftest changed-specification-cannot-publish-a-partially-prepared-frontend
  (let [v (source #'failed-source) impl (receiver v)
        metadata (meta impl) implementation @impl
        original-walk core/jit-walk-with-tc]
    (try
      (with-redefs [core/jit-walk-with-tc
                    (fn [& args]
                      (let [body (apply original-walk args)]
                        (alter-meta! impl update :raster.core/jit-spec assoc :return-tag 'Object)
                        body))]
        (is (= :unstable-pending-frontend (reason #(core/prepare-deftm-frontend! v)))))
      (is (nil? (get-in (meta impl) [:raster.core/jit-spec :frontend-source-owner])))
      (is (nil? (:raster.core/bytecode-emission (meta impl))))
      (is (identical? implementation @impl))
      (finally (reset-meta! impl metadata)))))
