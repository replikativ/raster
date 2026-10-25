(ns raster.compiler.jvm-loop-anf-test
  "Completed typed aset operands use private JVM locals, not source ANF.
   Only aset's own operands leave the value-loop stack; parent stacks and
   unrelated calls/constructors retain their existing evaluation semantics."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as rn]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bc]
            [raster.compiler.pipeline :as pipeline])
  (:import [java.lang.reflect InvocationTargetException]
           [java.lang.classfile ClassFile Instruction Attributes]
           [java.lang.classfile.instruction BranchInstruction]))

(defn- compile-form [params expression]
  (let [{:keys [methods] :as artifact}
        (bc/compile-specialized-class!
         "raster.compiled.CompletedStoreOperands"
         [{:name 'run :params params :return-tag 'Object
           :walked-body [expression] :source-ns *ns*}])
        method (get methods 'run)]
    (assoc artifact :run
           (fn [& arguments]
             (try (.invoke ^java.lang.reflect.Method method nil (object-array arguments))
                  (catch InvocationTargetException error (throw (.getCause error))))))))

(defn- store-form [tag]
  (list 'clojure.core/aset
        (with-meta '(do (mark :array) array) {:tag tag :raster.type/tag tag})
        '(do (mark :index) index)
        '(loop* [iteration (clojure.core/int 0)]
           (if (clojure.core/< iteration 2)
             (recur (clojure.core/unchecked-inc-int iteration))
             (do (mark :value) value)))))

(defn- invoke-store [artifact array index value]
  (let [trace (atom [])
        result (try [:value ((:run artifact) array index #(swap! trace conj %) value)]
                    (catch Throwable error [:throw (class error)]))]
    {:trace @trace :result result}))

(defn- method-code [artifact]
  (let [model (.parse (ClassFile/of) ^bytes (:bytes artifact))
        method (first (filter #(= "run" (.stringValue (.methodName %))) (.methods model)))]
    (.orElse (.code method) nil)))

(defn- loop-header-stacks [artifact]
  (let [^java.lang.classfile.attribute.CodeAttribute code (method-code artifact)
        instructions (filter #(instance? Instruction %) code)
        [_ headers]
        (reduce (fn [[offset headers] instruction]
                  [(+ offset (.sizeInBytes ^Instruction instruction))
                   (if (and (instance? BranchInstruction instruction)
                            (< (.labelToBci code (.target ^BranchInstruction instruction)) offset))
                     (conj headers (.labelToBci code (.target ^BranchInstruction instruction)))
                     headers)])
                [0 #{}] instructions)
        table (.orElse (.findAttribute code (Attributes/stackMapTable)) nil)
        frames (into {} (map (fn [frame] [(.labelToBci code (.target frame)) (vec (.stack frame))]))
                     (.entries table))]
    (into {} (map (fn [header] [header (get frames header ::missing-frame)])) headers)))

(deftest completed-store-operands-preserve-exception-and-effect-order
  (let [artifact (compile-form '[array index mark value] (store-form 'doubles))]
    (testing "array completion fails before index evaluation or the value loop"
      (is (= {:trace [:array] :result [:throw ClassCastException]}
             (invoke-store artifact (int-array 1) 0 7.0))))
    (testing "index completion fails before the value loop"
      (doseq [[index failure] [[nil NullPointerException] [(Object.) ClassCastException]]]
        (is (= {:trace [:array :index] :result [:throw failure]}
               (invoke-store artifact (double-array 1) index 7.0)))))
    (testing "null and bounds checks belong to the store, after value evaluation"
      (is (= {:trace [:array :index :value] :result [:throw NullPointerException]}
             (invoke-store artifact nil 0 7.0)))
      (let [array (double-array [3.0])]
        (is (= {:trace [:array :index :value] :result [:throw ArrayIndexOutOfBoundsException]}
               (invoke-store artifact array 1 7.0)))
        (is (= [3.0] (vec array)))))
    (testing "element completion fails without performing the store"
      (doseq [[value failure] [[nil NullPointerException] [(Object.) ClassCastException]]]
        (let [array (double-array [3.0])]
          (is (= {:trace [:array :index :value] :result [:throw failure]}
                 (invoke-store artifact array 0 value)))
          (is (= [3.0] (vec array))))))))

(deftest reference-store-check-stays-after-value-and-preserves-object-identity
  (let [artifact (compile-form '[array index mark value] (store-form 'objects))
        array (into-array String ["kept"])
        value (Object.)
        output (object-array 1)]
    (is (= {:trace [:array :index :value] :result [:throw ArrayStoreException]}
           (invoke-store artifact array 0 value)))
    (is (= ["kept"] (vec array)))
    (let [{:keys [trace result]} (invoke-store artifact output 0 value)]
      (is (= [:array :index :value] trace))
      (is (= :value (first result)))
      (is (identical? value (second result)))
      (is (identical? value (aget output 0))))))

(deftest completed-spills-use-actual-carriers-and-preserve-used-or-discarded-results
  (doseq [[tag make-array value expected carrier]
          [['doubles double-array 7.25 7.25 Double]
           ['floats float-array 7.25 (float 7.25) Float]
           ['ints int-array 7 7 Integer]
           ['longs long-array 4294967297 4294967297 Long]]
          used? [true false]]
    (let [store (store-form tag)
          artifact (compile-form '[array index mark value]
                                 (if used? store (list 'do store :discarded)))
          array (make-array 1)
          {:keys [trace result]} (invoke-store artifact array 0 value)]
      (is (= [:array :index :value] trace))
      (is (= [expected] (vec array)))
      (is (= :value (first result)))
      (if used?
        (do (is (= expected (second result)))
            (is (instance? carrier (second result))))
        (is (= :discarded (second result)))))))

(deftest loop-header-has-no-owned-aset-operands-but-keeps-an-incoming-parent
  (let [store (store-form 'doubles)
        plain (compile-form '[array index mark value] store)
        nested (compile-form '[array index mark value] (list 'clojure.core/+ 10.0 store))
        plain-stacks (loop-header-stacks plain)
        nested-stacks (loop-header-stacks nested)]
    (is (seq plain-stacks) "the emitted bytecode really contains a loop backedge")
    (is (every? empty? (vals plain-stacks))
        "array and completed Int index live in private locals, not at the value-loop header")
    (is (seq nested-stacks))
    (is (every? #(= 1 (count %)) (vals nested-stacks))
        "only the incoming Double parent remains; this is not universal OSR normalization")
    (is (= {:trace [:array :index :value] :result [:value 17.0]}
           (invoke-store nested (double-array 1) 0 7.0)))))

(deftest loop-detection-respects-method-frames-and-does-not-rewrite-source
  (is (nil? (ns-resolve 'raster.compiler.backend.jvm.bytecode 'lift-loop-operands)))
  (is (#'bc/same-frame-loop? '(do (loop* [i 0] i))))
  (doseq [form ['(quote (loop* [i 0] i))
                '(fn* [] (loop* [i 0] i))
                '(fn [] (loop* [i 0] i))
                '(reify* [clojure.lang.IDeref] (deref [this] (loop* [i 0] i)))
                '(deftype* Example Example [] :implements [] (method [this] (loop* [i 0] i)))]]
    (is (not (#'bc/same-frame-loop? form))))
  (let [source '(new java.util.AbstractMap$SimpleEntry java.lang.Integer (loop* [x 4] x))
        seen (atom [])
        emit #'bc/emit-form
        original @emit
        artifact (with-redefs-fn
                   {emit (fn [code form locals ctx]
                           (when (and (seq? form) (= 'new (first form))) (swap! seen conj form))
                           (original code form locals ctx))}
                   #(compile-form [] source))
        entry ((:run artifact))]
    (is (some #(= source %) @seen) "constructor/class syntax reaches its existing emitter unchanged")
    (is (identical? Integer (.getKey ^java.util.Map$Entry entry)))
    (is (= 4 (.getValue ^java.util.Map$Entry entry)))))

(deftm anf-host-mm [A :- (Array double) B :- (Array double) m :- Long k :- Long n :- Long] :- (Array double)
  (let [C (double-array (* m n))]
    (par/contract C [[i m] [j n]] [[l k]]
                  (rn/* (ra/aget A (+ (* i k) l)) (ra/aget B (+ (* l n) j))))
    C))

(defn- naive-mm [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (loop [l 0 acc 0.0]
                (if (< l k) (recur (inc l) (+ acc (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))) acc)))))
    C))

(deftest host-expanded-contraction-uses-completed-spills-and-keeps-ordered-results
  (let [spills (atom 0)
        spill #'bc/emit-completed-value-spill!
        original @spill
        f (with-redefs-fn {spill (fn [& arguments]
                                  (swap! spills inc)
                                  (apply original arguments))}
            #(pipeline/compile-aot #'anf-host-mm))
        [m k n] [7 5 3]
        r (java.util.Random. 7)
        A (double-array (repeatedly (* m k) #(.nextGaussian r)))
        B (double-array (repeatedly (* k n) #(.nextGaussian r)))]
    (is (pos? @spills) "the contraction is handled at the actual completed operand boundary")
    (is (= (vec (naive-mm A B m k n)) (vec (f A B m k n))))))
