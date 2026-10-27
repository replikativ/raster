(ns raster.gpu.public-device-scalar-result-test
  "A direct typed scalar result remains device-resident without becoming a host equation."
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.source-import :as source-import]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm direct-device-scalar-result
  [input :- (Array float)] :- Double
  (let [out (float-array 1)]
    (par/map-void! i 1
      (arrays/aset out i
                   (n/+ (n/* (arrays/aget input i) (float 1.0000001))
                        (float 0.03125001))))
    (double (arrays/aget out 0))))

(deftm completed-float-publication [completed :- Float] :- Float
  (let [out (float-array 1)]
    (par/map-void! i 1 (arrays/aset out i completed))
    (arrays/aget out 0)))

(deftm direct-map-helper-shift [input :- Double] :- Double
  (clojure.core/+ input 100.0))

(deftm direct-map-helper-result!
  [input :- (Array double) output :- (Array double) cnt :- Long] :- (Array double)
  (par/map! output i cnt double (direct-map-helper-shift (aget input i))))

(def ^:private samples
  [(float-array [1.2345679])
   (float-array [-3.1415927])
   (float-array [16777214.0])])

(defn- reference
  [^floats input]
  ;; Each operation is independently narrowed, matching the source's Float arithmetic and store;
  ;; only the terminal public result widens that checkpoint to Double.
  (double
   (float
    (+ (float (* (aget input 0) (float 1.0000001)))
       (float 0.03125001)))))

(defn- double-bits [number]
  (Double/doubleToRawLongBits (double number)))

(defn- direct-scalar-matches
  [prepared]
  (vec
   (for [program (if-let [compilation (get-in prepared [:ordered-invocation :compilation])]
                   [(:emitted compilation)]
                   (map #(get-in % [:call :program])
                        (get-in prepared [:lowering :plan :instances])))
         equation (:equations program)
         :let [algorithm (:algorithm equation)]
         :when (dialect/program-form? algorithm)
         inner (dialect/equations algorithm)
         :when (= 'scalar (dialect/operation-kind inner))
         :let [result (first (nth inner 2))
               emitted (first (:operations equation))
               operation (some (fn [scheduled-equation]
                                 (when (= (second inner) (:id scheduled-equation))
                                   (first (:operations scheduled-equation))))
                               (get-in emitted [:body :equations]))
               scalar-value (get-in (dialect/facts algorithm) [:values result])]
         :when (and (= [result] (dialect/outputs algorithm))
                    (= :resident-scalar-buffer (get-in scalar-value [:representation :kind])))]
     {:program program :equation equation :algorithm algorithm :inner inner
      :result result :operation operation :value scalar-value})))

(deftest source-result-preserves-the-float-checkpoint-before-double-widening
  (doseq [input samples]
    (let [before (vec input)
          expected (reference input)
          actual (direct-device-scalar-result input)]
      (is (= (double-bits expected) (double-bits actual)))
      (is (= before (vec input)) "the local singleton store never mutates the public input"))))

(defn- check-device! [target]
  (let [initial (first samples)
        snapshots (mapv vec samples)
        prepared
        (with-redefs [gpu/make-session (fn [& _]
                                         (throw (AssertionError. "lower opened a session")))
                      gpu/alloc! (fn [& _]
                                   (throw (AssertionError. "lower allocated")))
                      link/instantiate! (fn [& _]
                                          (throw (AssertionError. "lower instantiated")))]
          (compiled/lower #'direct-device-scalar-result [initial]
                          {:compiler :equation-first :target target :dtype :float}))
        matches (direct-scalar-matches prepared)
        result-entry (some #(when (= :result (:key %)) %) (:out-tree prepared))]
    (is (= :none (get-in prepared [:schedule :stats :fallback])))
    (is (= 0 (get-in prepared [:schedule :stats :emission :host-scalar-equations]))
        "the constant allocation extent is folded; no scalar result runs on the host")
    (is (not-any? #(true? (get-in % [:attributes :host-only]))
                  (mapcat #(get-in % [:program :equations]) matches))
        "the emitted equations independently confirm the absence of host scalar work")
    (is (= 1 (count matches)) "one semantic ScalarIR result has one direct device schedule")
    (when-let [{:keys [program equation inner result operation value]} (first matches)]
      (is (= [] (:shape value)) "the semantic scalar remains logically rank zero")
      (is (= :double (:dtype value)))
      (is (= {:kind :resident-scalar-buffer :elements 1} (:representation value)))
      (is (= :device (:memory-space value)))
      (is (= value (get-in program [:values result])))
      (is (= [result] (:results equation)))
      (is (= 1 (count (:operations equation))))
      (is (= :typed-soac (:algorithm-dialect operation)))
      (is (= (second inner) (:algorithm-equation operation)))
      (is (segop/seg-map? operation))
      (is (= 1 (get-in operation [:space :dims 0 :bound])))
      (is (= #{result} (:outputs operation)))
      (is (not (true? (get-in equation [:attributes :host-only])))))
    (is (= :double (:dtype (or (:abstract result-entry) result-entry))))
    (is (= [] (:shape (or (:abstract result-entry) result-entry)))
        "the public value preserves logical scalar shape")
    (let [artifact (compiled/instantiate! prepared)
          latest (atom nil)]
      (try
        (doseq [[index input] (map-indexed vector samples)]
          (let [previous @latest
                outputs (artifact (if (zero? index) {} {:input input}))
                output (:result outputs)
                actual (first (value/->host output))]
            (reset! latest output)
            (is (= #{:result} (set (keys outputs))))
            (is (value/device-array? output))
            (is (= :double (:dtype output)))
            (is (= [] (:shape output)))
            (is (= 1 (alength (value/->host output)))
                "the logical scalar has exactly one physical Double element")
            (is (= (double-bits (reference input)) (double-bits actual)))
            (is (value/live? output))
            (when previous
              (is (not (value/live? previous))
                  "replay invalidates the previous public scalar wrapper"))))
        (is (= snapshots (mapv vec samples)) "captured and replay inputs remain unchanged")
        (compiled/close! artifact)
        (is (not (value/live? @latest)))
        (finally (compiled/close! artifact))))))

(deftest direct-device-scalar-replays-on-both-local-backends
  (if @opencl/opencl-fp64-available?
    (check-device! :ocl:0)
    (opencl/opencl-skip! "direct Double scalar result" :fp64))
  (if @ze/gpu-available?
    (check-device! :ze:0)
    (ze/gpu-skip! "direct Double scalar result")))

(defn- check-direct-map-helper-body! [target projected]
  (let [initial (double-array [0.5 -1.0 0.0 2.0])
        changed (double-array [16777217.0 1.0e-12 -100.00000000001 Math/PI])
        output (double-array 4)
        ;; No explicit helper warmup: ordinary owning orchestration must close
        ;; the original source before creating the current-candidate witness.
        prepared (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                               gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                               link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
                   (compiled/lower #'direct-map-helper-result! [initial output (long 4)]
                                   {:compiler :equation-first :target target :dtype :double}))
        artifact (compiled/instantiate! prepared)
        previous (atom nil)]
    (try
      (is (= :none (get-in prepared [:schedule :stats :fallback])))
      (is (pos? @projected) "this exercises owned numerical projection, not ordinary helper inlining")
      (doseq [[input overrides] [[initial {}] [changed {:input changed}]]]
        (let [result (:result (artifact overrides))]
          (is (= (mapv #(+ 100.0 %) input) (vec (value/->host result))))
          (is (value/live? result))
          (when @previous (is (not (value/live? @previous))))
          (reset! previous result)))
      (is (= [0.5 -1.0 0.0 2.0] (vec initial)))
      (is (= [16777217.0 1.0e-12 -100.00000000001 Math/PI] (vec changed)))
      (is (= [0.0 0.0 0.0 0.0] (vec output)))
      (compiled/close! artifact)
      (is (not (value/live? @previous)))
      (finally (compiled/close! artifact)))))

(defn- check-direct-map-helper! [target]
  (let [helper (core/resolve-deftm-var #'direct-map-helper-shift)
        qualified (symbol (str (.ns ^clojure.lang.Var helper)) (str (:name (meta helper))))
        retained #{'raster.gpu.public-device-scalar-result-test/direct-map-helper-shift
                   qualified (symbol (namespace qualified) (str (name qualified) "-impl"))}
        ordinary-inline @#'inline/try-resolve-deftm
        project source-import/project-direct-double-map-results
        projected (atom 0)]
    ;; Deny only an optional optimization. All selected source, Method, argument
    ;; and module evidence stays real; no positive compiler fact is fabricated.
    ;; Keep the policy installed through execution so retained guards stay valid.
    (with-redefs-fn
      {#'inline/try-resolve-deftm
       (fn [callee & arguments]
         (when-not (contains? retained callee) (apply ordinary-inline callee arguments)))
       #'source-import/project-direct-double-map-results
       (fn [witness]
         (let [result (project witness)]
           (when result (swap! projected + (count (:applications result))))
           result))}
      #(check-direct-map-helper-body! target projected))))

(deftest direct-map-helper-replays-on-both-local-backends
  (if @opencl/opencl-fp64-available?
    (check-direct-map-helper! :ocl:0)
    (opencl/opencl-skip! "direct Double helper map" :fp64))
  (if @ze/gpu-available?
    (check-direct-map-helper! :ze:0)
    (ze/gpu-skip! "direct Double helper map")))

(defn- check-completed-float! [target]
  ;; This is an already completed source scalar. It does not claim to qualify
  ;; the separate source-result import and checked conversion machinery.
  (let [prepared (compiled/lower #'completed-float-publication [(float -0.0)]
                                 {:compiler :equation-first :target target :dtype :float})
        artifact (compiled/instantiate! prepared)
        previous (atom nil)]
    (try
      (dotimes [_ 2]
        (let [output (:result (artifact))
              host (value/->host output)]
          (is (= (class (float-array 0)) (class host)))
          (is (= [] (:shape output)))
          (is (= :float (:dtype output)))
          (is (= [Integer/MIN_VALUE] (mapv #(Float/floatToRawIntBits %) host)))
          (when @previous (is (not (value/live? @previous))))
          (reset! previous output)))
      (compiled/close! artifact)
      (is (not (value/live? @previous)))
      (finally (compiled/close! artifact)))))

(deftest completed-float-keeps-its-physical-class-and-signed-zero
  (if @opencl/opencl-available?
    (check-completed-float! :ocl:0)
    (opencl/opencl-skip! "completed Float publication"))
  (if @ze/gpu-available?
    (check-completed-float! :ze:0)
    (ze/gpu-skip! "completed Float publication")))
