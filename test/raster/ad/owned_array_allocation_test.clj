(ns raster.ad.owned-array-allocation-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as p]
            [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.value-completion :as value-completion]
            [raster.compiler.pipeline :as pipeline]))

(deftm unused-double-storage [x :- Double size :- Long] :- Double
  (let [scratch (double-array size)] (n/* x x)))
(deftm unused-float-storage [x :- Double size :- Long] :- Double
  (let [scratch (float-array size)] (n/* x x)))
(deftm read-fresh-storage [x :- Double size :- Long] :- Double
  (let [scratch (double-array size) zero (aget scratch 0)] (n/+ (n/* x x) zero)))
(deftm unused-storage-read [x :- Double size :- Long] :- Double
  (let [scratch (double-array size) ignored (aget scratch 0)] (n/* x x)))
(deftm unused-read-helper [x :- Double size :- Long] :- Double
  (unused-storage-read x size))
(deftm read-before-second-allocation [x :- Double size :- Long later-size :- Long] :- Double
  (let [scratch (double-array size)
        ignored (aget scratch 0)
        later (float-array later-size)]
    (n/* x x)))
(deftm nested-read-before-second-allocation [x :- Double size :- Long later-size :- Long] :- Double
  (read-before-second-allocation x size later-size))
(deftm active-count [x :- Double] :- Double
  (let [scratch (double-array (long x))] (n/* x x)))
(deftm effectful-count [x :- Double events :- (Array long) size :- Long] :- Double
  (let [scratch (double-array (do (aset events 0 (inc (aget events 0))) size))] (n/* x x)))
(deftm double-storage-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-double-storage :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))
(deftm float-storage-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-float-storage :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))
(deftm unused-read-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-storage-read :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))
(deftm helper-read-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-read-helper :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))
(deftm read-before-allocation-gradient [x :- Double size :- Long later-size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'read-before-second-allocation :wrt [0]) x size later-size)]
    (p/map [i 2] (double (nth result i)))))
(deftm nested-read-before-allocation-gradient
  [x :- Double size :- Long later-size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'nested-read-before-second-allocation :wrt [0])
                x size later-size)]
    (p/map [i 2] (double (nth result i)))))

(use-fixtures :once (fn [run] (n/* 1.0 1.0) (n/+ 1.0 1.0) (run)))

(defn- compile-owned [v]
  (#'reverse/compile-prepared-value+grad (reverse/prepare-owned-bytecode-value+grad v [0])))

(deftest source-obligations-survive-executable-traversal
  (let [read (with-meta '(clojure.core/aget storage 0)
               {:raster.effect/preserve-evaluation true :line 17})
        form (with-meta (list 'let* ['unused read] 1.0) {:line 16})
        traversed (#'reverse/postwalk-code identity form)
        quoted (list 'quote read)]
    (is (= (meta form) (meta traversed)))
    (is (= (meta read) (meta (second (second traversed)))))
    (is (identical? quoted (#'reverse/postwalk-code
                            (fn [node] (throw (ex-info "quoted data visited" {}))) quoted)))
    (is (= (meta read)
           (meta (second (second (#'reverse/binaryize-core-arithmetic form))))))
    (let [lexical (with-meta '(double-array size) {:raster.op/lexical-callee true})
          retained (#'reverse/retain-source-memory-evaluations lexical)]
      (is (= (meta lexical) (meta retained)))
      (is (not (:raster.effect/preserve-evaluation (meta retained)))))))

(deftest allocation-receipts-follow-executable-occurrences-only
  (let [allocation-a (with-meta '(double-array size) {:line 21})
        allocation-b (with-meta '(double-array size) {:line 22})
        equal-copy (with-meta '(double-array size) {:line 21})
        quoted (list 'quote allocation-a)
        body (list 'do quoted allocation-a allocation-b equal-copy)
        observe (fn [_ node]
                  (when (or (identical? node allocation-a)
                            (identical? node allocation-b))
                    {:fresh? true :source node}))
        occurrences (with-redefs [bytecode/retained-zero-array-allocation observe]
                      (#'reverse/retained-helper-allocations ::emission body))]
    (is (= 2 (count occurrences)))
    (is (identical? allocation-a (ffirst occurrences)))
    (is (identical? allocation-b (first (second occurrences))))
    (is (not-any? #(identical? equal-copy (first %)) occurrences)
        "structurally equal source is not an occurrence-identity receipt")
    (is (empty? (with-redefs [bytecode/retained-zero-array-allocation observe]
                  (#'reverse/retained-helper-allocations ::emission quoted)))
        "quoted payloads are never traversed as executable occurrences")
    (is (= :ad-paired-helper-memory-interface
           (with-redefs [bytecode/retained-zero-array-allocation
                         observe]
             (try (#'reverse/retained-helper-allocations
                   ::emission (list 'do allocation-a allocation-a))
                  nil
                  (catch clojure.lang.ExceptionInfo failure
                    (:reason (ex-data failure))))))
        "one identity receipt cannot authorize duplicated execution")))

(deftest owned-input-memory-capability-is-local-and-primitive-only
  (let [x (with-meta 'x {:raster.type/tag 'double})
        method (.getMethod Math "abs" (into-array Class [Double/TYPE]))
        input {:operand x :ports {x {:type :double}}
               :unretained-sibling-calls? false}
        direct {:input input :method method :spec {:walked-body [x]}}
        completion (value-completion/complete-value
                    (value-completion/bytecode-floating-input-contract
                     Double/TYPE (value-completion/bytecode-operand-context {})) x)
        completed (assoc direct :spec {:walked-body [completion]})
        effectful (assoc-in direct [:input :operand]
                            '(do (clojure.core/aset events 0 1) x))]
    (is (#'reverse/memory-free-owned-input-region? direct))
    (is (#'reverse/memory-free-owned-input-region? completed)
        "an exact primitive completion may still throw without touching memory")
    (is (not (#'reverse/memory-free-owned-input-region?
              (assoc-in completed [:input :ports x :type] :ref)))
        "a boxed/reference operand has no memory-free primitive-load proof")
    (is (not (#'reverse/memory-free-owned-input-region? effectful)))
    (is (not (#'reverse/memory-free-owned-input-region?
              (assoc-in direct [:input :unretained-sibling-calls?] true))))))

(deftest owned-allocation-receipt-pins-its-physical-formals
  (let [source (core/resolve-deftm-var #'unused-storage-read)
        receiver (ns-resolve (:ns (meta source))
                             (symbol (str (:name (meta source)) "-impl")))
        _ (core/ensure-bytecode-implementation! receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        artifact (#'reverse/get-paired-helper-program emission [0])
        arguments '[caller-x caller-size]
        reason (fn [candidate]
                 (try (#'reverse/paired-helper-memory-region
                       candidate arguments (set arguments))
                      nil
                      (catch clojure.lang.ExceptionInfo failure
                        (:reason (ex-data failure)))))
        parameters (:parameters artifact)
        retagged (assoc parameters 0
                        (vary-meta (first parameters) assoc :raster.type/tag 'doubles))
        facts (:value-facts artifact)
        fact-id (first parameters)
        fact (get facts fact-id)
        fact-root (:value fact)
        alternate-tag (if (= 'doubles (:raster.type/tag (meta fact-root)))
                        'double 'doubles)
        altered-facts
        (assoc (dissoc facts fact-id)
               (vary-meta fact-id assoc :raster.type/tag alternate-tag)
               (assoc fact :value
                      (vary-meta fact-root assoc :raster.type/tag alternate-tag)))]
    (is (nil? (reason artifact)))
    (is (= :ad-paired-helper-memory-interface
           (reason (assoc artifact :parameters retagged)))
        "symbol equality cannot erase the retained primitive/array carrier")
    (is (= :ad-paired-helper-memory-interface
           (reason (assoc artifact :parameters (vec (reverse parameters)))))
        "formal ordering is part of the exact replay projection")
    (is (= facts altered-facts)
        "Clojure equality deliberately ignores the carrier metadata")
    (is (not (identical? facts altered-facts)))
    (is (= :ad-paired-helper-memory-interface
           (reason (assoc artifact :value-facts altered-facts)))
        "an equal copied fact map cannot retag replay-root classification")))

(deftest unused-allocation-keeps-size-completion-and-failure-before-result
  (doseq [v [#'unused-double-storage #'unused-float-storage]]
    (let [gradient (compile-owned v)]
      (doseq [x [0.5 -1.2] size [0 1 3]]
        (let [[value derivative inactive] (gradient x size)]
          (is (= (* x x) value))
          (is (= (* 2.0 x) derivative))
          (is (nil? inactive))))
      (is (thrown? NegativeArraySizeException (gradient 2.0 -1)))
      (is (= [4.0 4.0 nil] (gradient 2.0 4294967296))
          "preserve original raw Int count completion instead of checked int"))))

(deftest fresh-array-reads-remain-visible-and-preserve-bounds-failures
  (let [gradient (compile-owned #'read-fresh-storage)]
    (is (= [9.0 6.0 nil] (gradient 3.0 1)))
    (is (thrown? ArrayIndexOutOfBoundsException (gradient 3.0 0)))
    (is (thrown? NegativeArraySizeException (gradient 3.0 -1)))))

(deftest invoked-jvm-aot-preserves-unused-allocation-failures
  (doseq [v [#'double-storage-gradient #'float-storage-gradient]]
    (let [compiled (pipeline/compile-aot v)]
      (is (= [9.0 6.0] (vec (compiled 3.0 0))))
      (is (thrown? NegativeArraySizeException (compiled 3.0 -1)))
      (is (= [9.0 6.0] (vec (compiled 3.0 4294967296)))))))

(deftest unused-source-reads-retain-bounds-and-evaluation-order
  (let [read-runtime (compile-owned #'unused-storage-read)
        ordered-runtime (compile-owned #'read-before-second-allocation)
        read-aot (pipeline/compile-aot #'unused-read-gradient)
        helper-aot (pipeline/compile-aot #'helper-read-gradient)
        ordered-aot (pipeline/compile-aot #'read-before-allocation-gradient)
        nested-ordered-aot (pipeline/compile-aot #'nested-read-before-allocation-gradient)]
    (doseq [[read expected] [[unused-storage-read 9.0]
                            [read-runtime [9.0 6.0 nil]]
                            [read-aot [9.0 6.0]]
                            [helper-aot [9.0 6.0]]]]
      (is (= expected (let [value (read 3.0 1)]
                        (if (number? value) value (vec value)))))
      (is (thrown? ArrayIndexOutOfBoundsException (read 3.0 0)))
      (is (thrown? NegativeArraySizeException (read 3.0 -1)))
      (is (thrown? ArrayIndexOutOfBoundsException (read 3.0 4294967296))
          "raw JVM Int count completion still wraps to zero"))
    (doseq [[ordered expected] [[read-before-second-allocation 9.0]
                               [nested-read-before-second-allocation 9.0]
                               [ordered-runtime [9.0 6.0 nil nil]]
                               [ordered-aot [9.0 6.0]]
                               [nested-ordered-aot [9.0 6.0]]]]
      (is (= expected (let [value (ordered 3.0 1 0)]
                        (if (number? value) value (vec value)))))
      (is (thrown? ArrayIndexOutOfBoundsException (ordered 3.0 0 -1))
          "the original read must fail before the later negative allocation")
      (is (thrown? NegativeArraySizeException (ordered 3.0 1 -1)))
      (is (thrown? NegativeArraySizeException (ordered 3.0 -1 0))))))

(deftest allocation-freshness-does-not-authorize-active-or-effectful-counts
  (doseq [v [#'active-count #'effectful-count]]
    (let [failure (try (reverse/prepare-owned-bytecode-value+grad v [0]) nil
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= :ad-unclosed-bytecode-body-region (:reason failure)))
      (is (= :array-allocation (:phase failure))))))
