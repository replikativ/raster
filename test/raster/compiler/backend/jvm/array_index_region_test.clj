(ns raster.compiler.backend.jvm.array-index-region-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.tooling.inspect :as inspect]))

(deftm indexed [xs :- (Array double) i :- Long stride :- Long offset :- Long] :- Double
  (aget xs (+ (* i stride) offset)))
(deftm interop-index [xs :- (Array double)] :- Double
  (aget xs (Math/abs (long 0))))
(deftm nested-read-index [xs :- (Array double) indices :- (Array long)] :- Double
  (aget xs (long (aget indices 0))))
(deftm lexical-index [xs :- (Array double) f :- Object] :- Double
  (aget xs (long (f 0))))

(defn- owned-read
  ([] (owned-read #'indexed))
  ([v]
  (let [source (core/resolve-deftm-var v)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        emission (core/ensure-current-owned-bytecode! receiver)
        read (first (filter #(and (seq? %) (= 'clojure.core/aget (first %)))
                            (vals (:invocation-catalog emission))))]
    {:emission emission :read read :primal @receiver})))

(deftest outlined-index-preserves-physical-ports-overflow-and-raw-narrowing
  (let [{:keys [emission read primal]} (owned-read)
        observed (bytecode/retained-array-read emission read)
        region (bytecode/compile-array-index-region emission read)
        invoke (fn [i stride offset]
                 (apply (:invoke region) (map {'i i 'stride stride 'offset offset} (:parameters region))))]
    (is (= :long (get-in observed [:input :type])))
    (is (true? (:primitive-index? observed)))
    (is (= Integer/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
    (is (= #{'i 'stride 'offset} (set (:parameters region))))
    (is (= 7 (invoke 2 3 1)))
    (is (= 0 (invoke 0 1 4294967296)))
    (is (= 2.0 (primal (double-array [2.0]) 0 1 4294967296)))
    (is (= -1 (invoke 0 1 -1)))
    (is (thrown? ArithmeticException (invoke Long/MAX_VALUE 2 0)))
    (is (thrown? ArithmeticException (invoke Long/MAX_VALUE 1 1)))
    (is (nil? (bytecode/retained-array-read emission (with-meta (apply list read) (meta read)))))
    (is (= :unowned-bytecode-array-index-region
           (try (bytecode/compile-array-index-region emission (with-meta (apply list read) (meta read)))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest unobserved-computation-routes-stay-open-even-under-an-integer-cast
  (doseq [v [#'interop-index #'nested-read-index #'lexical-index]]
    (let [{:keys [emission]} (owned-read v)
          reads (keep #(bytecode/retained-array-read emission %) (vals (:invocation-catalog emission)))
          outer (first (filter #(= 'xs (:array %)) reads))]
      (is (some? outer))
      (is (false? (:primitive-index? outer))))))

(deftest index-siblings-use-a-direct-primitive-call-and-retain-their-declaring-owner
  (let [{:keys [emission read]} (owned-read)
        region (bytecode/compile-array-index-region emission read)
        spec (:spec region)
        params (:params spec)
        call (list* 'input params)
        artifact (bytecode/compile-specialized-class!
                  "raster.compiled.IndexSiblingControl"
                  [{:name 'run :params params :return-tag 'int :walked-body [call] :source-ns *ns*}
                   {:name 'delayed :params params :return-tag 'Object
                    :walked-body [(list 'fn [] call)] :source-ns *ns*}
                   spec])
        run ^java.lang.reflect.Method (get-in artifact [:methods 'run])
        delayed ^java.lang.reflect.Method (get-in artifact [:methods 'delayed])
        values (mapv {'i 2 'stride 3 'offset 1} (:parameters region))
        code (with-out-str (inspect/disassemble (:bytes artifact) {:method "run"}))]
    (is (= Integer/TYPE (.getReturnType run)))
    (is (= 7 (.invoke run nil (object-array values))))
    (is (= 7 ((.invoke delayed nil (object-array values)))))
    (is (re-find #"(?s)INVOKESTATIC.*input" code))
    (is (not (re-find #"reflect|IFn|Integer" code)))))
