(ns raster.compiler.backend.gpu.compile-fixture-report-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.compile-fixture-report :as report]
            [raster.compiler.equation-first :as equation-first]))

(deftest every-refusal-is-reported-without-hiding-later-targets
  (let [seen (atom [])
        first-error (ex-info "not yet lowered" {:reason :missing-lowering})
        second-error (IllegalArgumentException. "bad artifact")
        error (try
                (report/collect!
                 #(mapcat
                   (fn [target]
                     (report/collect!
                      (fn []
                        (keep identity
                              (for [function [:first :middle :last]]
                                (report/attempt!
                                 {:function function :target target :dtype :float}
                                 (fn []
                                   (swap! seen conj [target function])
                                   (case function
                                     :first (throw first-error)
                                     :last (throw second-error)
                                     :middle {:source "valid kernel"}))))))))
                   [:cuda :hip]))
                nil
                (catch clojure.lang.ExceptionInfo error error))]
    (is (= [[:cuda :first] [:cuda :middle] [:cuda :last]
            [:hip :first] [:hip :middle] [:hip :last]] @seen))
    (is (= :gpu-public-fixture-failures (:reason (ex-data error))))
    (is (= 6 (:attempted (ex-data error))))
    (is (= 2 (:passed (ex-data error))))
    (is (= [:missing-lowering nil :missing-lowering nil]
           (mapv :reason (:failures (ex-data error)))))
    (is (= [:cuda :cuda :hip :hip] (mapv :target (:failures (ex-data error)))))
    (is (identical? first-error (.getCause error)))))

(deftest successful-collections-preserve-each-artifact-and-order
  (is (= [{:source "a"} {:source "b"}]
         (report/collect!
          #(map (fn [source] (report/attempt! {} (fn [] {:source source}))) ["a" "b"]))))
  (is (= 7 (report/attempt! {} (fn [] 7)))))

(deftest nested-lowering-refusals-retain-bounded-contract-identities-not-expressions
  (let [opaque (lazy-seq (throw (AssertionError. "diagnostic realized an arbitrary value")))
        error (try
                (report/collect!
                 #(vector
                   (report/attempt!
                    {:function 'example/kernel :target :cuda:fixture}
                    (fn []
                      (throw (ex-info "portable map refused"
                                      {:reason :kernel-graph-target-lowering-missing
                                       :kernel-body-decline
                                       {:reason :segmap-kernel-body-declined
                                        :missing-rule :checked-floating-conversion
                                        :source :double :target :float :fallback :none
                                        :requirement :write-only-output
                                        :expression opaque :body opaque}}))))))
                nil (catch clojure.lang.ExceptionInfo error error))
        diagnostic (first (:failures (ex-data error)))]
    (is (= :cuda:fixture (:target diagnostic)))
    (is (= {:reason :segmap-kernel-body-declined
            :missing-rule :checked-floating-conversion
            :source :double :target :float :fallback :none
            :requirement :write-only-output}
           (:kernel-body-decline diagnostic)))
    (is (= {:reason :checked-map-stage-declined :requirement :single-output-store}
           (#'report/lowering-diagnostic
            {:reason :checked-map-stage-declined :requirement :single-output-store
             :expression opaque})))
    (is (= {} (#'report/lowering-diagnostic
               {:kernel-body-decline {:source opaque :expression opaque}
                :requirement opaque})))))

(deftest cancellation-and-vm-failures-are-not-converted-to-compile-refusals
  (doseq [failure [(InterruptedException. "interrupted")
                   (java.util.concurrent.CancellationException. "cancelled")
                   (AssertionError. "fatal")
                   (ex-info "wrapped interrupt" {} (InterruptedException. "interrupted"))
                   (java.util.concurrent.ExecutionException.
                    (java.util.concurrent.CancellationException. "cancelled"))
                   (ex-info "wrapped VM failure" {} (OutOfMemoryError. "fatal"))]]
    (let [later? (atom false)]
      (is (identical? failure
                      (try
                        (report/collect!
                         #(do (report/attempt! {} (fn [] (throw failure)))
                              (reset! later? true) []))
                        nil
                        (catch Throwable error error))))
      (is (false? @later?))))
  (let [failure (ex-info "outside collection" {})]
    (is (identical? failure
                    (try (report/attempt! {} (fn [] (throw failure)))
                         nil (catch Throwable error error))))))

(deftest ordinary-cause-cycles-and-large-messages-remain-bounded-refusals
  (let [first-error (Exception. (apply str (repeat 2048 "x")))
        second-error (Exception. "second" first-error)
        _ (.initCause first-error second-error)
        error (try (report/collect! #(vector (report/attempt! {} (fn [] (throw first-error)))))
                   nil (catch clojure.lang.ExceptionInfo error error))
        diagnostic (first (:failures (ex-data error)))]
    (is (= :gpu-public-fixture-failures (:reason (ex-data error))))
    (is (= 1024 (count (:message diagnostic))))
    (is (true? (:message-truncated? diagnostic)))))

(deftest fixture-wrapper-forwards-the-public-call-and-rejects-empty-artifacts
  (require 'raster.compiler.backend.gpu.kernel-body-compile-fixtures)
  (let [wrapper (ns-resolve 'raster.compiler.backend.gpu.kernel-body-compile-fixtures
                            'compile-public-fixture)
        function (ns-resolve 'raster.compiler.backend.gpu.kernel-body-compile-fixtures
                            'public-c-family-dot)
        descriptor {:device-id :cuda:fixture-report}
        calls (atom [])
        error
        (with-redefs [equation-first/compile
                      (fn [f options captured]
                        (swap! calls conj [f options captured])
                        (case (:case options)
                          :refused (throw (ex-info "unsupported" {:reason :fixture-refused}))
                          :empty {:kernels []}
                          :success {:kernels [{:source "valid kernel"}]}))]
          (try
            (report/collect!
             #(keep (fn [case-id]
                      (wrapper function {:target :cuda:fixture-report :dtype :float :case case-id}
                               descriptor))
                    [:refused :empty :success]))
            nil
            (catch clojure.lang.ExceptionInfo error error)))]
    (is (= [:refused :empty :success] (mapv #(get-in % [1 :case]) @calls)))
    (is (every? #(and (identical? function (nth % 0)) (identical? descriptor (nth % 2))) @calls))
    (is (= [:fixture-refused :gpu-public-fixture-empty]
           (mapv :reason (:failures (ex-data error)))))
    (is (= 3 (:attempted (ex-data error))))
    (is (= 1 (:passed (ex-data error))))))

(deftest later-generation-failure-preserves-collected-public-refusals
  (let [public-error (ex-info "public refusal" {:reason :public-refusal})
        generation-error (ex-info "private generation refusal" {:reason :private-refusal})
        error (try
                (report/collect!
                 #(do (report/attempt! {:target :cuda} (fn [] (throw public-error)))
                      (throw generation-error)))
                nil (catch Exception error error))]
    (is (= :gpu-public-fixture-failures (:reason (ex-data error))))
    (is (= [:public-refusal] (mapv :reason (:failures (ex-data error)))))
    (is (= :private-refusal (get-in (ex-data error) [:generation-failure :reason])))
    (is (identical? public-error (.getCause error)))
    (is (= [generation-error] (vec (.getSuppressed error)))))
  (let [generation-error (ex-info "private only" {:reason :private-refusal})]
    (is (identical? generation-error
                    (try (report/collect! #(throw generation-error))
                         nil (catch Exception error error)))))
  (let [cancelled (ex-info "cancelled generation" {} (InterruptedException.))]
    (is (identical? cancelled
                    (try
                      (report/collect!
                       #(do (report/attempt! {} (fn [] (throw (ex-info "refused" {}))))
                            (throw cancelled)))
                      nil (catch Exception error error))))))

(deftest public-contraction-fixtures-preserve-their-declared-storage
  (require 'raster.compiler.backend.gpu.kernel-body-compile-fixtures)
  (let [fixture-ns 'raster.compiler.backend.gpu.kernel-body-compile-fixtures
        generate (ns-resolve fixture-ns 'equation-first-artifacts)
        wrapper (ns-resolve fixture-ns 'compile-public-fixture)
        double-contract (ns-resolve 'raster.linalg.contract 'contract-mm)
        float-contract (ns-resolve 'raster.compiler.fixtures.contractions 'fixed-matmul)]
    (doseq [target [:cuda :hip]]
      (let [calls (atom [])]
        (with-redefs-fn
          {wrapper (fn [function options & _]
                     (swap! calls conj [function options])
                     {:kernels []})}
          #(doall (generate target {:subgroup-size 32 :max-workgroup-size 1024
                                    :compute-capability [8 0]})))
        (is (= [:double]
               (mapv #(get-in % [1 :dtype])
                     (filter #(identical? double-contract (first %)) @calls))))
        (is (= [:float :float]
               (mapv #(get-in % [1 :dtype])
                     (filter #(identical? float-contract (first %)) @calls)))
            "both FP32 multiply-add policy fixtures remain covered")))))

(deftest both-public-target-groups-precede-private-generation
  (require 'raster.compiler.backend.gpu.kernel-body-compile-fixtures)
  (let [fixture-ns 'raster.compiler.backend.gpu.kernel-body-compile-fixtures
        public-var (ns-resolve fixture-ns 'equation-first-artifacts)
        private-var (ns-resolve fixture-ns 'emit-target-with-public!)
        emit-all (ns-resolve fixture-ns 'emit-all!)
        seen (atom [])
        error
        (with-redefs-fn
          {public-var (fn [target _]
                        (swap! seen conj [:public target])
                        (report/collect!
                         #(vector (report/attempt!
                                   {:target target :function :test-fixture}
                                   (fn [] (throw (ex-info "public refusal" {:reason :public-refusal})))))))
           private-var (fn [_ target _]
                         (swap! seen conj [:private target])
                         (throw (ex-info "private refusal" {:reason :private-refusal})))}
          #(try (emit-all "unused-test-root")
                nil (catch Exception error error)))]
    (is (= [[:public :cuda] [:public :hip] [:private :cuda]] @seen))
    (is (= 2 (:attempted (ex-data error))))
    (is (= [:cuda :hip] (mapv :target (:failures (ex-data error)))))
    (is (= :private-refusal (get-in (ex-data error) [:generation-failure :reason])))))
