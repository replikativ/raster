(ns raster.compiler.passes.scalar.host-entry-contract-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.passes.scalar.host-entry-contract :as entry]))

(def condition {:op :host-object-distinct :scope :original-host-arguments
                :left 'a :right 'b})
(def assertion '(if (clojure.core/identical? a b)
                 (throw (clojure.core/ex-info "different" {:input (quote a)})) nil))
(defn project [source] (entry/project-entry-assertions source '{a doubles b doubles x double} '[a b]))
(defn reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest explicit-original-host-scope
  (is (= :kernel-precondition-scope (reason #(precondition/validate! [condition] #{}))))
  (is (= :kernel-precondition-scope (reason #(precondition/check! [condition] identity))))
  (is (= [condition] (precondition/validate! [condition] #{} {:host-object-identities '#{a b}})))
  (let [a (double-array 0) b (double-array 0)]
    (is (precondition/check! [condition] identity {:host-objects {'a a 'b b}}))
    (is (= :kernel-precondition-failed
           (reason #(precondition/check! [condition] identity {:host-objects {'a a 'b a}})))))
  (doseq [invalid [(assoc condition :scope :device-storage)
                   (assoc condition :right 'a) (assoc condition :extra true)]]
    (is (= :kernel-precondition-invalid
           (reason #(precondition/validate! [invalid] #{} {:host-object-identities '#{a b}})))))
  (is (= :kernel-precondition-scope
         (reason #(precondition/validate! [condition] #{} {:host-object-identities '#{a}}))))
  (is (precondition/check! [{:op :>= :expression 'n :value 2}] {'n 3})))

(deftest exact-unused-entry-assertion
  (let [source (list 'let* ['guard assertion 'y 'x] 'y)
        result (project source)]
    (is (= [condition] (:preconditions result)))
    (is (= '(let* [y x] y) (:form result))))
  (testing "repeated/reversed requirements deduplicate without conflating identities"
    (let [reverse-assertion (assoc (vec assertion) 1 '(clojure.core/identical? b a))
          source (list 'let* ['g assertion 'h (apply list reverse-assertion)] 'x)]
      (is (= [condition] (:preconditions (project source))))))
  (is (= [condition]
         (:preconditions (project (list 'let* ['v 'x 'g assertion] 'v))))))

(deftest no-speculative-premises
  (doseq [source [(list 'let* ['g assertion] 'g)
                  (list 'let* ['g assertion] '{:used g})
                  (list 'let* ['v '(aget a 0) 'g assertion] 'x)
                  (list 'let* ['v '(double-array 2) 'g assertion] 'x)
                  (list 'let* ['v '(+ x 1) 'g assertion] 'x)
                  (list 'let* ['v '(aset a 0 1.0) 'g assertion] 'x)
                  (list 'let* ['a 'b 'g assertion] 'x)
                  (list 'let* ['g (list 'if 'x assertion nil)] 'x)
                  (list 'let* ['g '(if (clojure.core/identical? a b)
                                     (throw (helper!)) nil)] 'x)
                  (list 'let* ['g '(if (clojure.core/identical? a b)
                                     (throw (clojure.core/ex-info "different" {:v (helper!)})) nil)] 'x)
                  (list 'let* ['g '(if predicate (throw (clojure.core/ex-info "x" {})) nil)] 'x)
                  (list 'let* ['g '(if (identical? a b)
                                     (throw (clojure.core/ex-info "x" {})) nil)] 'x)
                  (list 'let* ['g '(if (clojure.core/identical? a local-array)
                                     (throw (clojure.core/ex-info "x" {})) nil)] 'x)]]
    (is (= {:form source :preconditions []} (project source)) (str source)))
  (doseq [shadowed '[clojure.core/identical? clojure.core/ex-info]]
    (let [source (list 'let* ['g assertion] 'x)]
      (is (= [] (:preconditions
                  (entry/project-entry-assertions source
                    (assoc '{a doubles b doubles x double} shadowed nil) '[a b])))))))

(deftest parameter-and-call-authority
  (doseq [[environment arrays] [['{a doubles b doubles} '[a a]]
                                ['{a doubles b double} '[a b]]
                                ['{a doubles} '[a b]]
                                ['{a doubles b doubles} '[a 1]]]]
    (is (= :invalid-host-entry-parameters
           (reason #(entry/project-entry-assertions '(let* [] nil) environment arrays)))))
  (doseq [expression [(list 'if (with-meta '(clojure.core/identical? a b)
                                          {:raster.op/lexical-callee true})
                           (nth assertion 2) nil)
                      (list 'if (second assertion)
                            (list 'throw (with-meta '(clojure.core/ex-info "x" {})
                                                    {:raster.op/lexical-callee true})) nil)]]
    (let [source (list 'let* ['g expression] 'x)]
      (is (= {:form source :preconditions []} (project source))))))
