(ns raster.gpu.compiled-scan-session-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.gpu.core :as gpu]
            [raster.gpu.ocl-runtime]
            [raster.gpu.ze-runtime]))

(deftest backend-local-scan-runners-are-retired
  (doseq [[namespace names]
          [['raster.gpu.ocl-runtime '[invoke-registered-scan-exclusive-kernel]]
           ['raster.gpu.ze-runtime '[invoke-registered-scan-exclusive-kernel
                                     invoke-full-gpu-scan!]]]
          name names]
    (is (nil? (ns-resolve namespace name))
        "scans execute the retained typed graph, not a second backend-owned algorithm")))

(defn- scan-artifact
  []
  (kart/make
   {:kernel-name "session_scan"
    :source (str "__kernel void session_scan(__global const int* input, "
                 "__global int* output, int _n_bound) {}")
    :abi [(kabi/slot 'input :input :int :role :operand)
          (kabi/slot 'output :output :int :role :result)
          (kabi/slot '_n_bound :scalar :int :role :bound)]
    :arguments '[input output n]
    :launch (klaunch/spec {:workgroup-size [64]
                           :group-count [(klaunch/ceil-div 'n 64)]})
    :effects {:kind :scan :reads ['input] :writes ['output]}
    :attributes {:strategy :scheduled-graph}}))

(defn- scan-dispatch
  []
  (kdispatch/make
   {:id "session-scan-dispatch"
    :alternatives [(scan-artifact)]
    :default-strategy :scheduled-graph
    :selector {:kind :fixed-strategy :strategy :scheduled-graph}}))

(deftest compile-retains-first-class-dispatches-beside-flat-kernels
  (let [session (atom {:device-id :ze:0})
        compilations (atom 0)
        compiled {:kernels [:registered-kernel]
                  :dispatches [(scan-dispatch)]}]
    (with-redefs-fn
      {#'raster.gpu.core/compile-deftm-internal!
       (fn [_ _ _] (swap! compilations inc) compiled)}
      #(do
         (is (= [:registered-kernel]
                (gpu/compile! session :first #'identity)))
         (is (= [:registered-kernel]
                (gpu/compile! session :second #'identity)))))
    (is (= 1 @compilations) "the complete compiled result is cached once")
    (is (= (:dispatches compiled) (get-in @session [:dispatches :first])))
    (is (= (:dispatches compiled) (get-in @session [:dispatches :second])))))

(deftest compile-cache-includes-the-effective-source-routing-threshold
  (let [session (atom {:device-id :ze:0})
        calls (atom [])]
    (with-redefs-fn
      {#'raster.gpu.core/compile-deftm-internal!
       (fn [_ _ opts]
         (swap! calls conj opts)
         {:kernels [(get opts :min-elements 0)] :dispatches []})}
      #(do
         (is (= [0] (gpu/compile! session :default #'identity)))
         (is (= [0] (gpu/compile! session :explicit-default #'identity {:min-elements 0})))
         (is (= [64] (gpu/compile! session :threshold #'identity {:min-elements 64})))
         (is (= [64] (gpu/compile! session :same-threshold #'identity {:min-elements 64})))
         (is (= [nil] (gpu/compile! session :explicit-nil #'identity {:min-elements nil})))
         (is (= [0] (gpu/compile! session :declared-storage #'identity
                                 {:preserve-declared-array-storage? true})))))
    (is (= [{} {:min-elements 64} {:min-elements nil}
            {:preserve-declared-array-storage? true}] @calls)
        "absent/zero share a cache identity, but explicit nil is not silently defaulted")))

(deftest compile-cache-preserves-revision-invalidation-and-unstable-acquisition
  (let [session (atom {:device-id :ze:0})
        revision (atom :first)
        calls (atom 0)
        unstable? (atom false)]
    (with-redefs-fn
      {#'dispatch/compiler-definition-revision #(deref revision)
       #'raster.gpu.core/compile-deftm-internal!
       (fn [_ _ _]
         (let [number (swap! calls inc)]
           (when @unstable? (swap! revision #(keyword (str (name %) "-changed"))))
           {:kernels [number] :dispatches []}))}
      #(do
         (is (= [1] (gpu/compile! session :first #'identity)))
         (is (= [1] (gpu/compile! session :cached #'identity)))
         (reset! revision :second)
         (is (= [2] (gpu/compile! session :new-revision #'identity)))
         (reset! unstable? true)
         (reset! revision :unstable)
         (is (= [3] (gpu/compile! session :unstable #'identity)))
         (reset! unstable? false)
         (is (= [4] (gpu/compile! session :retry #'identity)))
         (is (= [4] (gpu/compile! session :retry-cached #'identity)))))))

(deftest compile-acquisition-rejects-closed-releasing-and-reentrant-retirement
  (let [calls (atom 0)]
    (with-redefs-fn
      {#'raster.gpu.core/compile-deftm-internal!
       (fn [_ _ _] (swap! calls inc) {:kernels [] :dispatches []})}
      #(doseq [state [{:closed? true} {:lifecycle :releasing}]]
         (is (= :session-releasing
                (try (gpu/compile! (atom (assoc state :device-id :ze:0)) :phase #'identity)
                     nil
                     (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
    (is (zero? @calls)))
  (let [session (atom {:device-id :ze:0})
        failure (atom nil)]
    (with-redefs-fn
      {#'raster.gpu.core/compile-deftm-internal!
       (fn [_ _ _]
         (reset! failure (try (gpu/close-session! session) nil
                             (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
         {:kernels [:compiled] :dispatches []})}
      #(is (= [:compiled] (gpu/compile! session :phase #'identity))))
    (is (= :reentrant-root-lifecycle @failure))
    (is (not (:closed? @session)))))

(deftest compile-acquisition-finishes-before-a-requested-session-close
  (let [session (atom {:device-id :ze:0 :arena-id ::arena})
        entered (promise) finish (promise) close-requested (promise)
        calls (atom [])
        compiling (atom nil) closing (atom nil)]
    (with-redefs-fn
      {#'raster.gpu.core/compile-deftm-internal!
       (fn [_ _ _]
         (deliver entered true)
         (when (= ::timeout (deref finish 5000 ::timeout))
           (throw (ex-info "compile fixture timed out" {})))
         (swap! calls conj :register)
         (is (not (:closed? @session)))
         {:kernels [:compiled] :dispatches [:dispatch]})
       #'raster.gpu.core/rt-resolve
       (fn [_ operation]
         (case operation
           "close-kernel-arena!"
           (fn [arena]
             (is (= ::arena arena))
             (is (= [:compiled] (get-in @session [:kernels :phase])))
             (is (= [:dispatch] (get-in @session [:dispatches :phase])))
             (swap! calls conj :close-arena))
           (throw (ex-info "unexpected backend operation" {:operation operation}))))}
      #(try
         (reset! compiling (future (gpu/compile! session :phase #'identity)))
         (is (= true (deref entered 5000 ::timeout)))
         (reset! closing (future (deliver close-requested (Thread/currentThread))
                                 (gpu/close-session! session)))
         (let [close-thread (deref close-requested 5000 nil)
               deadline (+ (System/nanoTime) 5000000000)]
           ;; Observe an actual close request blocked at the existing session monitor, not an
           ;; arbitrary sleep that could pass before the closing future even starts.
           (when close-thread
             (loop []
               (when (and (not= java.lang.Thread$State/BLOCKED (.getState ^Thread close-thread))
                          (not (realized? @closing)) (< (System/nanoTime) deadline))
                 (Thread/yield)
                 (recur))))
           (is (some? close-thread))
           (is (and close-thread (= java.lang.Thread$State/BLOCKED (.getState ^Thread close-thread))))
           (is (empty? @calls)))
         (deliver finish true)
         (is (= [:compiled] (deref @compiling 5000 ::timeout)))
         (let [closed-result (deref @closing 5000 ::timeout)]
           (is (not= ::timeout closed-result))
           (is (= :closed (:lifecycle closed-result))))
         (is (= [:register :close-arena] @calls))
         (is (= :closed (:lifecycle @session)))
         (is (empty? (:kernels @session)))
         (is (empty? (:dispatches @session)))
         (finally
           (deliver finish true)
           ;; Retain and join both futures before the temporary resolver/compiler roots leave
           ;; scope. Cleanup must not replace an unexpected primary fixture exception.
           (doseq [task [@compiling @closing] :when task]
             (try (deref task 5000 ::timeout) (catch Throwable _ nil))))))))

(deftest scan-invocation-binds-the-retained-ordered-executable-abi
  (let [dispatch (scan-dispatch)
        session (atom {:dispatches {:scan [dispatch]}
                       :buffers {:output :resident-output}})
        calls (atom [])]
    (with-redefs [gpu/bind-kernel-executable!
                  (fn [actual-session key executable arguments]
                    (swap! calls conj [:bind actual-session key executable arguments])
                    ::handle)
                  gpu/run-kernel-graph!
                  (fn [actual-session handle]
                    (swap! calls conj [:run actual-session handle]))
                  gpu/release-kernel-graph!
                  (fn [actual-session handle]
                    (swap! calls conj [:release actual-session handle]))]
      (is (= :resident-output
             (gpu/invoke-scan! session :scan [:input] :output 513))))
    (let [[_ actual-session key executable arguments] (first @calls)]
      (is (identical? session actual-session))
      (is (= [:compiled-scan :scan] key))
      (is (= (kdispatch/default-alternative dispatch) executable))
      (is (= [:input :output {:type :int :value 513}] arguments)))
    (is (= [[:run session ::handle]
            [:release session ::handle]]
           (subvec @calls 1))))
  (testing "flat kernels cannot silently stand in for a missing scheduled dispatch"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"exactly one KernelDispatch"
         (gpu/invoke-scan! (atom {:kernels {:scan [:block :prop]}})
                           :scan [:input] :output 8)))))
