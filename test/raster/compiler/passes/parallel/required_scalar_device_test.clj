(ns raster.compiler.passes.parallel.required-scalar-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as compiler]
            [raster.compiler.fixtures.required-scalar :as fixture]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as runtime]))

(deftest unused-required-stage-replays-and-fences-publication-on-failure
  (if-not @opencl/opencl-fp64-available?
    (opencl/opencl-skip! "required scalar invocation" :fp64)
    (let [compilation (compiler/compile-typed (fixture/algorithm) {:target :ocl:0})
          plan (compiler/lower compilation [(double-array [1.0 2.0]) 2 2.0])
          input (get-in plan [:attributes :public-buffer-bindings 'x])
          output (first (:outputs plan))
          owned (link/instantiate! plan)
          calls (atom [])
          failure-at (atom nil)
          failure (ex-info "injected required-stage failure" {:reason ::injected})
          ;; Prepare a genuine request owner using the public callback boundary; never
          ;; mutate a sealed prepared program or rely on redefining direct-linked Vars.
          executor (link/resident-program-executor
                    plan (:session owned) ::failure-probe (:node-views owned)
                    false false (:allocation-keys owned))
          prepared (runtime/prepare-sequence-with!
                    (:instances plan)
                    (assoc executor :run!
                           (fn [handle]
                             (swap! calls conj handle)
                             (if (= @failure-at (count @calls))
                               (throw failure)
                               ((:run! executor) handle)))))
          executable (assoc owned :graph-key nil :prepared-program prepared)]
      (try
        (link/run! executable)
        (is (= [2.0 3.0] (vec (link/download executable output))))
        (is (= 3 (count @calls)) "reduction, required scalar, independent map all execute")
        (let [successful-order @calls]
          (link/upload! executable input (double-array [-3.0 4.0]))
          (reset! calls [])
          (link/run! executable)
          (is (= successful-order @calls))
          (is (= [-2.0 5.0] (vec (link/download executable output))))
          (reset! calls [])
          (reset! failure-at 2)
          (do
            (is (identical? failure
                            (try (link/run! executable) nil
                                 (catch clojure.lang.ExceptionInfo error error))))
            (is (= (subvec successful-order 0 2) @calls)
                "failure prevents the later independent map from being submitted")
            (is (thrown? clojure.lang.ExceptionInfo (link/download executable output))
                "a failed replay cannot publish even the preceding replay's output")))
        (finally
          (runtime/release-prepared! prepared)
          (link/close! owned))))))
