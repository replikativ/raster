(ns raster.compiler.passes.parallel.structured-control-frontend-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.passes.parallel.structured-control-frontend :as frontend]
            [raster.compiler.passes.parallel.structured-control-route :as route]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed-frontend]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]))

(def ^:private initial
  (av/tensor {:dtype :double :shape ['extent] :representation {:kind :plain}}))

(defn- loop-source
  [copy-length]
  (list
   'let*
   ['n '(int (clojure.core/alength u0))
    'u '(clojure.core/aclone u0)
    'scratch '(raster.numeric/similar_m_doubles u0)
    'time-loop
    (list 'dotimes '[step steps]
          '(raster.par/stencil! scratch [u] 1 :dirichlet double i n
                                (+ (clojure.core/aget u i) alpha))
          (list 'let* ['next '(raster.par/pmap j n double
                                               (+ (clojure.core/aget u j)
                                                  (clojure.core/aget scratch j)
                                                  (* 0.0 step)))]
                (list 'java.lang.System/arraycopy 'next 0 'u 0 copy-length)))
    'after '(raster.par/pmap k n double (clojure.core/aget u k))]
   'after))

(def ^:private options
  {:dtype :double
   :values {'u0 initial
            'steps (av/tensor {:dtype :long :shape []})
            'alpha (av/tensor {:dtype :double :shape []})}
   :scalar-types {'steps :long 'alpha :double}})

(defn- update-bindings
  [source f]
  (apply list (first source) (f (second source)) (drop 2 source)))

(deftest source-correspondence-belongs-to-the-whole-frontend-construction
  (let [source (loop-source 'n)
        decomposition (frontend/form->structured-loop source options)
        typed (route/program-envelope decomposition options)
        equation (program/equation-for-binding typed (:loop-binding decomposition)
                                               (:loop-source decomposition))
        original-loop (:loop decomposition)
        changed-loop (control/make (control/facts original-loop)
                                   [(first (control/loop-index original-loop)) 0]
                                   (control/invariants original-loop) (control/carried original-loop)
                                   (control/body original-loop) (control/outer-values original-loop))
        reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (identical? decomposition (frontend/validate-decomposition! decomposition)))
    (is (identical? (nth (second source) 7) (:loop-source decomposition)))
    (is (= (:loop-source decomposition) (:source equation)))
    (is (not-any? fn? (tree-seq coll? seq typed)) "construction witnesses never enter pure IR")
    (doseq [changed (concat
                    [(into {} decomposition)
                     (assoc (frontend/form->structured-loop source options)
                            :provenance-seal (:provenance-seal decomposition))
                     (dissoc decomposition :source)]
                    (for [[key replacement] [[:source '(let* [] nil)]
                                             [:loop-source '(dotimes [i 0] nil)]
                                             [:loop-binding 'another-loop]
                                             [:prefix-bindings []] [:suffix-bindings []]
                                             [:outer-body []] [:parameter-values {}] [:outer-values {}]
                                             [:outer-equalities {}]
                                             [:typed-body-source nil] [:flattened-body-source nil]
                                             [:copy-certificate nil] [:write-certificates []]
                                             [:loop changed-loop]]]
                      (assoc decomposition key replacement)))]
      (is (= :structured-control-source-construction
             (reason #(route/program-envelope changed options)))))))

(deftest selected-loop-binding-must-be-unambiguous
  (let [source (update-bindings (loop-source 'n) #(into % ['time-loop 0]))]
    (is (= :structured-loop-source-binding
           (try (frontend/form->structured-loop source options) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest normalized-prefix-relations-survive-the-loop-analysis-boundary
  (let [options (assoc options :array-types {'u0 :double})
        source (typed-frontend/normalize-source (loop-source 'n) options)
        decomposition (frontend/form->structured-loop source options)
        copy (:copy-certificate decomposition)
        source-indices (filter #(and (symbol? %)
                                      (.startsWith (name %) "rstr_loop_source_index_"))
                               (tree-seq coll? seq (:flattened-body-source decomposition)))]
    (is (some? decomposition))
    (is (= 'u (:destination copy)))
    (is (= :double (:dtype copy)))
    (is (= [(:extent copy)] (get-in decomposition [:outer-values 'u :shape])))
    (is (identical? source (:source decomposition)))
    (is (seq source-indices))
    (is (not-any? #(contains? (:outer-values decomposition) %) source-indices))
    (is (not (contains? (:parameter-values decomposition) 'u)))
    (is (not (contains? (:outer-values decomposition) 'after)))))

(deftest loop-index-shadowing-cannot-inherit-an-outer-extent-equality
  (let [source (update-bindings
                (loop-source 'n)
                (fn [bindings]
                  (update bindings 7
                          #(apply list (first %) '[n steps] (drop 2 %)))))
        options (assoc options :array-types {'u0 :double})]
    (doseq [source [source (typed-frontend/normalize-source source options)]]
      (is (nil? (frontend/form->structured-loop source options))
          "the copy count now names the varying iteration, not the outer array length"))))

(deftest destination-writing-loop-becomes-a-typed-functional-fixpoint
  (let [{:keys [loop prefix-bindings suffix-bindings copy-certificate
                write-certificates fusion-stats typed-body-source]}
        (frontend/form->structured-loop (loop-source 'n) options)
        body (control/body loop)
        scheduled (lower/schedule loop {:target-device :cpu:0 :dtype :double})
        graph (:graph scheduled)
        body-inputs (:inputs (soac/facts body))]
    (is (= loop (control/validate! loop)))
    (is (soac/program-form? body))
    (is (= 3 (count prefix-bindings)))
    (is (= 1 (count suffix-bindings)))
    (is (= {:source 'rstr_loop_value_1 :destination 'u :extent 'n :dtype :double}
           copy-certificate))
    (is (= [{:destination 'scratch :extent 'n :dtype :double :kind :stencil}]
           write-certificates))
    (is (map? fusion-stats))
    (testing "destination mutation is represented as logical SSA dataflow"
      (is (not-any? #{'scratch} body-inputs))
      (is (some #{'rstr_loop_value_0}
                (tree-seq coll? seq typed-body-source))))
    (testing "the ordinary SOAC scheduler owns the iteration graph"
      (is (lower/scheduled-loop? scheduled))
      (is (not-any? #{'scratch} (map :id (:inputs graph)))))))

(deftest expanded-copy-helper-may-return-its-mutated-carry
  (let [source (loop-source 'n)
        bindings (second source)
        dotimes (nth bindings 7)
        copy-expression (last dotimes)
        returning-copy (apply list (concat (butlast dotimes)
                                           [(list 'do copy-expression 'u)]))
        source (apply list 'let* (assoc bindings 7 returning-copy) (drop 2 source))
        {:keys [loop copy-certificate typed-body-source]}
        (frontend/form->structured-loop source options)]
    (is (= loop (control/validate! loop)))
    (is (= {:source 'rstr_loop_value_1 :destination 'u :extent 'n :dtype :double}
           copy-certificate))
    (is (not-any? #{'java.lang.System/arraycopy}
                  (tree-seq coll? seq typed-body-source))
        "the physical state transition is not retained as numerical body work")))

(deftest fresh-prefix-allocation-may-flow-through-a-lexical-alias
  (let [source (loop-source 'n)
        source (update-bindings
                source #(vec (concat (subvec % 0 4)
                                     ['scratch-storage (nth % 5)
                                      'scratch 'scratch-storage]
                                     (subvec % 6))))
        {:keys [loop write-certificates]}
        (frontend/form->structured-loop source options)]
    (is (= loop (control/validate! loop)))
    (is (= [{:destination 'scratch :extent 'n :dtype :double :kind :stencil}]
           write-certificates))))

(deftest unsupported-or-ambiguous-state-transitions-decline-or-fail-exactly
  (testing "a partial copy is not silently interpreted as a loop carry"
    (is (nil? (frontend/form->structured-loop (loop-source '(dec n)) options))))
  (testing "two writes to one physical destination violate one-iteration SSA"
    (let [source (loop-source 'n)
          bindings (second source)
          dotimes (nth bindings 7)
          dotimes (apply list (concat (take 3 dotimes)
                                      [(nth dotimes 2)]
                                      (drop 3 dotimes)))
          source (apply list 'let* (assoc bindings 7 dotimes) (drop 2 source))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"one physical destination may be written only once"
           (frontend/form->structured-loop source options))))))

(deftest loop-boundary-does-not-admit-suffix-or-escaping-scratch-values
  (testing "a later binding is not falsely available as a loop invariant"
    (let [source (loop-source 'n)
          bindings (second source)
          dotimes (nth bindings 7)
          dotimes (walk/postwalk-replace {'alpha 'late-value} dotimes)
          bindings (assoc bindings 7 dotimes)
          bindings (vec (concat (subvec bindings 0 8)
                                ['late-value 1.0]
                                (subvec bindings 8)))
          source (apply list 'let* bindings (drop 2 source))]
      (is (nil? (frontend/form->structured-loop source options)))))
  (testing "a functionalized physical scratch buffer may not escape after the loop"
    (let [source (loop-source 'n)
          source (update-bindings
                  source #(assoc % 9 '(raster.par/pmap k n double
                                                       (clojure.core/aget scratch k))))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"may not escape"
           (frontend/form->structured-loop source options)))))
  (testing "an alias of caller storage is not accepted as private scratch"
    (let [source (loop-source 'n)
          source (update-bindings source #(assoc % 5 'u0))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"require fresh prefix allocations"
           (frontend/form->structured-loop source options))))))

(deftest generated-boundaries-are-alpha-safe-and-retain-dotimes-semantics
  (testing "all source binders are reserved from generated SSA names"
    (let [source (walk/postwalk-replace {'next 'rstr_loop_value_0} (loop-source 'n))
          {:keys [typed-body-source]} (frontend/form->structured-loop source options)]
      (is (= 'rstr_loop_value_1 (first (second typed-body-source))))))
  (testing "a negative literal dotimes count has zero-trip semantics"
    (let [source (loop-source 'n)
          source (update-bindings
                  source #(update % 7 (fn [dotimes]
                                        (apply list (first dotimes)
                                               [(first (second dotimes)) -4]
                                               (drop 2 dotimes)))))
          loop (:loop (frontend/form->structured-loop source options))]
      (is (= 0 (second (control/loop-index loop))))
      (is (= :clamp-nonnegative
             (get-in (control/facts loop) [:attributes :trip-count-semantics]))))))
