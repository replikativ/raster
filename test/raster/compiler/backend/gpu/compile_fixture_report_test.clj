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
