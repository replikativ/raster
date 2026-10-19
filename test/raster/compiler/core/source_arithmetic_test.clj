(ns raster.compiler.core.source-arithmetic-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.source-arithmetic :as arithmetic]))

(deftest comparison-carriers-are-not-arithmetic-promotion
  (doseq [[left expected] [[:int [:int :long :double :double]]
                          [:long [:long :long :double :double]]
                          [:float [:double :double :float :double]]
                          [:double [:double :double :double :double]]]
          [right result] (map vector [:int :long :float :double] expected)]
    (is (= result (arithmetic/comparison-carrier left right))))
  (doseq [[left right] [[:ref :int] [nil :float] [:float nil]]]
    (is (= :double (arithmetic/comparison-carrier left right))))
  ;; Source oracle: narrowing the integer to Float here silently changes true to false.
  (is (< (float 16777216) (int 16777217)))
  (is (< (float 16777216) (long 16777217))))

(deftest primitive-recipes-preserve-existing-widths
  ;; Ordered rows/columns are actual int, long, float, double carriers.
  (doseq [operator ["+" "-" "*"]
          [left expected] (map vector [:int :long :float :double]
                               [[:int :long :float :double]
                                [:long :long :double :double]
                                [:double :double :float :double]
                                [:double :double :double :double]])
          [right result] (map vector [:int :long :float :double] expected)]
    (is (= {:kind :primitive :input-carriers [result result] :result-carrier result}
           (arithmetic/binary-plan operator left right))))
  (doseq [left [:int :long :float :double]
          right [:int :long :float :double]
          :let [result (if (= [:float :float] [left right]) :float :double)]]
    (is (= result (:result-carrier (arithmetic/binary-plan "/" left right))))))

(deftest generic-recipes-name-real-numbers-overloads
  (let [classes {:long Long/TYPE :double Double/TYPE :ref Object}
        normalize {:int :long :long :long :float :double :double :double
                   :ref :ref :bool :ref}]
    (doseq [[operator method-name] [["+" "add"] ["-" "minus"]
                                    ["*" "multiply"] ["/" "divide"]]
            left (keys normalize) right (keys normalize)
            :when (or (#{:bool :ref} left) (#{:bool :ref} right))
            :let [plan (arithmetic/binary-plan operator left right)
                  inputs [(normalize left) (normalize right)]
                  method (.getMethod clojure.lang.Numbers method-name
                                     (into-array Class (map classes inputs)))]]
      (is (= :numbers (:kind plan)))
      (is (= inputs (:input-carriers plan)))
      (is (= method-name (:method-name plan)))
      (is (= (if (some #{:double} inputs) :double :ref) (:result-carrier plan)))
      (is (= (if (= :double (:result-carrier plan)) Double/TYPE Number)
             (.getReturnType method))))))

(deftest plans-are-not-completion-or-generic-call-proofs
  (testing "Object/Object and primitive/Object division keep different source overloads"
    (is (= [:ref :ref] (:input-carriers (arithmetic/binary-plan "/" :ref :ref))))
    (is (= [:double :ref] (:input-carriers (arithmetic/binary-plan "/" :double :ref))))
    (is (not (contains? (arithmetic/binary-plan "/" :double :ref) :nonthrowing))))
  (doseq [[op left right] [["sqrt" :double :double] ["/" :Double :Double]
                           ["+" :void :int] ["*" :int nil]]]
    (is (thrown? clojure.lang.ExceptionInfo (arithmetic/binary-plan op left right)))))
