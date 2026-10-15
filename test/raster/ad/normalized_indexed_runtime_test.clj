(ns raster.ad.normalized-indexed-runtime-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-indexed-map-test :as construction]
            [raster.numeric :as numeric]
            [raster.par :as par]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- bundle [dt it n m]
  (normalized/linearize-indexed-map-program
   (#'construction/indexed-map dt it n m) *ns*))
(defn- args [prepared environment] (mapv environment (:inputs prepared)))
(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- gradient [prepared pullback dy]
  (nth (pullback dy) (.indexOf ^java.util.List (:inputs prepared) 'data)))

(deftest indexed-execution-is-owned-by-the-canonical-envelope
  (let [evidence (bundle :double :long 'n 'm)
        owner (atom nil) validated (atom []) materializations (atom 0)
        adapt canonical/adapt validate canonical/validate!
        materialize route/materialize-host-program]
    (with-redefs [canonical/adapt
                  (fn [b] (let [result (adapt b)] (reset! owner result) result))
                  canonical/validate!
                  (fn [b] (swap! validated conj b) (validate b))
                  route/materialize-host-program
                  (fn [p] (swap! materializations inc) (materialize p))]
      (let [prepared (runtime/prepare evidence)
            preparation-proofs (count @validated)
            preparation-materializations @materializations
            input (args prepared {'n (long 2) 'indices (long-array [1 1])
                                  'data (double-array [3.0 5.0 7.0])})
            [output pb] ((:vjp prepared) input)
            g (gradient prepared pb (double-array [1.0 2.0]))
            [jy tangent] ((:jvp prepared) input (double-array [0.25 0.5 0.75]))]
        (is (= [5.0 5.0] (vec output) (vec jy)))
        (is (= [0.0 3.0 0.0] (vec g)))
        (is (= [0.5 0.5] (vec tangent)))
        (is (= '[m] (get-in @owner [:interface :tangent-inputs 0 :value :shape])
               (get-in @owner [:interface :cotangent-results 0 :value :shape])))
        (is (= '[n] (get-in @owner [:interface :primal-result :value :shape])
               (get-in @owner [:interface :tangent-result :value :shape])
               (get-in @owner [:interface :cotangent-input :value :shape])))
        (is (= (mapv :id (get-in @owner [:interface :inputs])) (:inputs prepared)))
        (is (= 2 preparation-proofs) "adapter canonical proof and independent post-capture proof")
        (is (= preparation-proofs (count @validated)) "no hot canonical proof replay")
        (is (= 3 preparation-materializations @materializations) "only three generated projections, none hot")
        (is (every? #(identical? @owner %) @validated)
            "preparation and execution validate the same canonical owner, not a legacy facet")))))

(deftest indexed-success-is-fenced-and-original-failure-remains-primary
  (let [prepared (runtime/prepare (bundle :double :long 'n 'm))
        input (args prepared {'n (long 1) 'indices (long-array [0]) 'data (double-array [2.0 3.0])})
        [_ pb] ((:vjp prepared) input)
        invoke-var #'runtime/invoke invoke @invoke-var
        baseline (normalized/revisions)]
    (doseq [operation [#((:vjp prepared) input)
                       #((:jvp prepared) input (double-array 2))
                       #(pb (double-array [1.0]))]]
      (let [current (atom baseline)]
        (with-redefs-fn {#'normalized/revisions #(deref current)
                        invoke-var (fn [& xs]
                                     (let [result (apply invoke xs)]
                                       (swap! current update :template-revision inc)
                                       result))}
          #(is (= :ad-normalized-stale-rules (reason operation))))))
    (let [current (atom baseline) failure (ex-info "original indexed projection failure" {})]
      (with-redefs-fn {#'normalized/revisions #(deref current)
                      invoke-var (fn [& _]
                                   (swap! current update :template-revision inc)
                                   (throw failure))}
        #(is (identical? failure (try ((:vjp prepared) input) nil (catch Throwable e e))))))))

(deftest canonical-runtime-contract-tampering-refuses-before-compilation
  (let [evidence (bundle :double :long 'n 'm)
        owner (canonical/adapt-indexed-map evidence)
        [destination _] (first (:physical-bindings owner))]
    (doseq [changed [(assoc-in owner [:interface :active-inputs] '[indices])
                     (assoc-in owner [:interface :cotangent-input :value :shape] '[m])
                     (assoc-in owner [:residual-bindings 'indices :origin :input] 'data)
                     (assoc-in owner [:residual-bindings 'indices :id] 'data)
                     (assoc-in owner [:physical-bindings destination :requirement :exclusive] false)
                     (assoc-in owner [:physical-bindings destination :consumer :result] destination)]]
      (let [changed (assoc changed :certificate (#'canonical/certificate changed))
            compiled (atom 0)]
        (with-redefs-fn {#'canonical/adapt (constantly changed)
                        #'runtime/compile-projection (fn [& _] (swap! compiled inc))}
          #(do
             (is (= :ad-program-bundle-correspondence (reason (fn [] (runtime/prepare evidence)))))
             (is (zero? @compiled))))))))

(deftest indexed-prepared-owner-captures-read-update-and-host-macro-dependencies
  (let [prepared (runtime/prepare (bundle :double :long 'n 'm))
        input (args prepared {'n (long 1) 'indices (long-array [0]) 'data (double-array [2.0 3.0])})
        [_ pb] ((:vjp prepared) input)]
    (doseq [dependency [#'clojure.core/aget #'numeric/+ #'par/map-void!]]
      (let [metadata (meta dependency)]
        (try
          (with-redefs-fn {dependency (fn [& _] nil)}
            #(do
               (is (= :ad-normalized-execution-callee (reason (fn [] ((:vjp prepared) input)))))
               (is (= :ad-normalized-execution-callee (reason (fn [] ((:jvp prepared) input (double-array 2))))))
               (is (= :ad-normalized-execution-callee (reason (fn [] (pb (double-array [1.0]))))))))
          ;; Restoring a macro's root may clear its macro flag. Restore exact fixture
          ;; metadata AFTER with-redefs, not an assumed hand-written flag.
          (finally (reset-meta! dependency metadata)))))
    (is (= [2.0] (vec (first ((:vjp prepared) input)))))))

(deftest indexed-original-read-alias-is-not-erased-by-preparation
  (let [source-ns (create-ns (gensym "raster-indexed-source-alias-"))]
    (try
      (binding [*ns* source-ns] (clojure.core/refer 'clojure.core))
      (let [source (#'construction/indexed-map :double :long 'n 'm)
            [equation] (soac/equations source)
            {:keys [kind attributes arrays captures lambda]} (soac/operation-parts equation)
            {:keys [parameters]} (soac/lambda-parts lambda)
            aliased (soac/make (soac/facts source)
                              [(list '= (second equation) (nth equation 2)
                                     (list kind attributes arrays captures
                                           (soac/lambda-form parameters '[(aget data-array index)])))]
                              (soac/outputs source))
            evidence (normalized/linearize-indexed-map-program aliased source-ns)
            prepared (runtime/prepare evidence)
            input (args prepared {'n (long 1) 'indices (long-array [0]) 'data (double-array [2.0 3.0])})
            [_ pb] ((:vjp prepared) input)]
        (ns-unmap source-ns 'aget)
        (intern source-ns 'aget (fn [& _] 99.0))
        (is (= :ad-normalized-execution-callee (reason #((:vjp prepared) input))))
        (is (= :ad-normalized-execution-callee (reason #((:jvp prepared) input (double-array 2)))))
        (is (= :ad-normalized-execution-callee (reason #(pb (double-array [1.0])))))
        (ns-unmap source-ns 'aget)
        (binding [*ns* source-ns] (clojure.core/refer 'clojure.core :only '[aget]))
        (is (= [2.0] (vec (first ((:vjp prepared) input))))))
      (finally (remove-ns (ns-name source-ns))))))

(deftest indexed-runtime-owns-snapshot-and-every-pullback-destination
  (doseq [[dt array cast] [[:double double-array double] [:float float-array float]]
          [it index-array] [[:long long-array] [:int int-array]]
          n [0 3]]
    (let [prepared (runtime/prepare (bundle dt it 'n 'm))
          indices (index-array (take n [2 0 2])) data (array [2.0 5.0 7.0 11.0])
          input (args prepared {'indices indices 'data data 'n (long n)})
          dx (array [0.5 1.0 2.0 3.0]) dy (array (take n [0.5 -2.0 3.0]))
          [y pb] ((:vjp prepared) input) [jy tangent] ((:jvp prepared) input dx)
          expected (if (zero? n) [0.0 0.0 0.0 0.0] [-2.0 0.0 3.5 0.0])]
      (is (= (vec (take n [7.0 2.0 7.0])) (vec y) (vec jy)))
      (is (= (vec (take n [2.0 0.5 2.0])) (vec tangent)))
      (is (= (vec (take n [2 0 2])) (vec indices)))
      (is (= [2.0 5.0 7.0 11.0] (vec data)))
      (dotimes [i n] (aset indices i (if (= it :int) (int 1) (long 1))) (aset y i (cast -99.0)))
      (dotimes [i 4] (aset data i (cast 99.0)))
      (let [g (gradient prepared pb dy)]
        (is (= (class data) (class g)))
        (is (= expected (vec g)))
        (is (= (reduce + 0.0 (map * (seq tangent) (seq dy)))
               (reduce + 0.0 (map * (seq dx) (seq g)))))
        (doseq [[id slot] (map vector (:inputs prepared) (pb dy))]
          (is (= (= 'data id) (some? slot))))
        (dotimes [i 4] (aset g i (cast -9.0)))
        (let [again (gradient prepared pb dy)]
          (is (not (identical? g again)))
          (is (= expected (vec again))))))))

(deftest repeated-indices-preserve-ordered-float-rounding-and-double-fd
  (let [prepared (runtime/prepare (bundle :float :long 'n 'm))
        [_ pb] ((:vjp prepared) (args prepared {'n 3 'indices (long-array [0 0 0]) 'data (float-array [1.0])}))
        dy (float-array [16777216.0 1.0 -16777216.0])
        expected (reduce (fn [s v] (float (+ (float s) (float v)))) (float 0.0) (seq dy))]
    (is (= (Float/floatToRawIntBits expected)
           (Float/floatToRawIntBits (aget ^floats (gradient prepared pb dy) 0))))
    (is (= 0.0 expected)))
  (let [prepared (runtime/prepare (bundle :double :int 'n 'm))
        data (double-array [1.0 4.0 9.0]) indices (int-array [2 0 2]) seed (double-array [0.5 -2.0 3.0])
        env {'n 3 'indices indices 'data data} [_ pb] ((:vjp prepared) (args prepared env))
        g (gradient prepared pb seed)]
    (doseq [i (range 3)]
      (let [h 1.0e-5
            objective (fn [delta]
                        (let [x (aclone data)]
                          (aset x i (+ (aget data i) delta))
                          ;; Independent ordinary gather objective, not a generated AD primal.
                          (reduce + (map (fn [weight index] (* weight (aget ^doubles x index)))
                                         (seq seed) (seq indices)))))
            fd (/ (- (objective h) (objective (- h))) (* 2 h))]
        (is (< (Math/abs (- fd (aget g i))) 1.0e-8))))))

(deftest all-index-and-capacity-validation-precedes-primal
  (let [prepared (runtime/prepare (bundle :double :long 'n 'm))
        calls (atom 0) invoke (ns-resolve 'raster.ad.normalized-map-runtime 'invoke)
        original @invoke env {'n 1 'indices (long-array [0]) 'data (double-array 2)}]
    (with-redefs-fn {invoke (fn [& xs] (swap! calls inc) (apply original xs))}
      (fn []
        (doseq [index [-1 2 Long/MAX_VALUE]]
          (is (= :ad-map-runtime-index
                 (reason #((:vjp prepared) (args prepared (assoc env 'indices (long-array [index]))))))))
        (is (= :ad-map-runtime-array
               (reason #((:vjp prepared) (args prepared (assoc env 'indices (int-array [0])))))))
        (is (= :ad-map-runtime-array
               (reason #((:vjp prepared) (args prepared (assoc env 'data (float-array 2)))))))
        (doseq [dx [(float-array 2) (double-array 1)]]
          (is (= :ad-map-runtime-array (reason #((:jvp prepared) (args prepared env) dx)))))
        (is (zero? @calls)))))
  (doseq [m [2 'n]]
    (let [prepared (runtime/prepare (bundle :double :int 'n m))]
      (is (= :ad-map-runtime-capacity
             (reason #((:vjp prepared) (args prepared {'n 1 'indices (int-array [0]) 'data (double-array 3)})))))))
  (let [prepared (runtime/prepare (bundle :double :int 0 3))
        [_ pb] ((:vjp prepared) (args prepared {'indices (int-array 0) 'data (double-array 3)}))]
    (is (= [0.0 0.0 0.0] (vec (gradient prepared pb (double-array 0)))))))

(deftest physical-projection-keeps-effect-nil-but-returns-the-owned-destination
  (let [primal (get-in (bundle :double :long 'n 'm) [:gather-programs :primal])
        out (first (soac/outputs primal))
        with-ownership (fn [ownership]
                         (soac/make (assoc-in (soac/facts primal) [:values out :ownership] ownership)
                                    (soac/equations primal) (soac/outputs primal)))]
    (is (= (soac/outputs primal) (:outputs (route/materialize-host-indexed-map (with-ownership nil)))))
    (is (= :typed-soac-host-indexed-boundary
           (reason #(route/materialize-host-indexed-map (with-ownership :borrowed))))))
  (let [b (bundle :float :int 'n 'm)
        {:keys [transpose interface]} (:gather-programs b)
        projection (route/materialize-host-indexed-map transpose)
        zero (:zero-gradient interface)]
    (is (= [zero] (:physical-inputs projection)))
    (is (not (some #{zero} (:inputs projection))))
    (is (= [(get (:value-symbols projection) zero)] (last (:body projection))))
    (is (= 'raster.par/map-void! (first (second (second (:body projection))))))
    (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map transpose)))))
  (let [b (bundle :double :long 'n 'm) prepared (runtime/prepare b)
        invoke (ns-resolve 'raster.ad.normalized-map-runtime 'invoke) original @invoke
        indices (long-array [1]) data (double-array 3) snapshot (atom nil)
        [y pb] (with-redefs-fn
                 {invoke (fn [executable environment]
                           (let [results (original executable environment)]
                             (reset! snapshot (second results)) results))}
                 (fn [] ((:vjp prepared) (args prepared {'n 1 'indices indices 'data data}))))
        zero (get-in b [:gather-programs :interface :zero-gradient]) owners (atom [])]
    (is (not (identical? indices @snapshot)))
    (is (not (identical? y @snapshot)))
    (with-redefs-fn
      {invoke (fn [executable environment]
                (let [owner (get environment zero)]
                  (is (identical? @snapshot (get environment (get-in b [:gather-programs :interface :index-snapshot]))))
                  (is (not-any? #(or (identical? data %) (identical? indices %) (identical? y %)) (vals environment)))
                  (is (= (long 1) (get environment 'n)))
                  (is (= (long 3) (get environment 'm)))
                  (is (= [0.0 0.0 0.0] (vec owner)))
                  (swap! owners conj owner)
                  (let [result (original executable environment)]
                    (is (identical? owner (first result)))
                    result)))}
      (fn [] (pb (double-array [2.0])) (pb (double-array [2.0]))))
    (is (not (identical? (first @owners) (second @owners))))
    (with-redefs-fn {invoke (fn [executable environment]
                             [(aclone ^doubles (first (original executable environment)))])}
      (fn [] (is (= :ad-map-runtime-physical-owner (reason #(pb (double-array [2.0])))))))
    (is (= :ad-map-runtime-array (reason #(pb (float-array 1)))))
    (is (= :ad-map-runtime-array (reason #(pb (double-array 2)))))))

(deftest real-functional-frontend-and-stale-or-forged-bundles
  (let [options {:dtype :double :array-types {'data :double 'indices :long}
                 :scalar-types {'n :long 'm :long}
                 :values {'data (av/tensor {:dtype :double :shape '[m]})
                          'm (av/tensor {:dtype :long :shape []})}}
        form '(let* [out (raster.par/pmap i n nil (clojure.core/aget data (clojure.core/aget indices i)))] out)
        p (frontend/form->program (frontend/normalize-source form options) options)
        b (normalized/linearize-indexed-map-program p *ns*) prepared (runtime/prepare b)
        arguments (args prepared {'n 2 'indices (long-array [1 1]) 'data (double-array [3.0 5.0 7.0])})
        [y pb] ((:vjp prepared) arguments)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions) revision (revisions)
        invoke (ns-resolve 'raster.ad.normalized-map-runtime 'invoke) calls (atom 0)]
    (is (= [5.0 5.0] (vec y)))
    (is (= [0.0 3.0 0.0] (vec (gradient prepared pb (double-array [1.0 2.0])))))
    (is (some? (reason #(runtime/prepare (assoc-in b [:gather-programs :required-zero-gradient :exclusive] false)))))
    (doseq [facet [:map-programs :sum-programs]]
      (is (= :ad-program-bundle-scope (reason #(runtime/prepare (assoc b facet {}))))))
    (with-redefs-fn {revisions (constantly (update revision :template-revision inc)) invoke (fn [& _] (swap! calls inc))}
      (fn []
        (is (= :ad-normalized-stale-rules (reason #((:vjp prepared) arguments))))
        (is (= :ad-normalized-stale-rules (reason #((:jvp prepared) arguments (double-array 3)))))
        (is (= :ad-normalized-stale-rules (reason #(pb (double-array 2)))))
        (is (zero? @calls))))))
