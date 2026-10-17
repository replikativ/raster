(ns raster.compiler.core.value-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.value-completion :as completion]))

(defn- completed [expression]
  (completion/complete-value completion/object-input-contract expression))

(defn- snapshot-product [fields]
  ;; Each field is consumed by its OWN original Object context. No CSE or
  ;; primitive-local inference may merge or rebox two materializations.
  (let [ids (mapv #(symbol (str "field__" %)) (range (count fields)))]
    (list 'let* (vec (mapcat vector ids (map completed fields))) ids)))

(defn- source [expression]
  (completion/project-to-jvm-source expression))

(defn- run [expression input]
  ((eval (list 'fn* '[input] (source expression))) input))

(deftest object-materialization-retains-original-literal-identity
  (doseq [value [1 1000 5.125]]
    (let [fields [value '(do nil)]
          original (list 'identical? (list 'first fields) value)
          transformed (list 'identical? (list 'first (snapshot-product fields)) value)]
      (is (true? (run original nil)) "Independent original constant identity oracle")
      (is (true? (run transformed nil))))))

(deftest primitive-and-boxed-inputs-retain-their-own-materialization-contexts
  (doseq [hint [nil 'long 'double]]
    (let [param (cond-> 'input hint (with-meta {:tag hint}))
          original '[input input]
          transformed (snapshot-product original)
          compile-shape (fn [body]
                          (eval (list 'fn* [param]
                                      (list 'let* ['pair (source body)]
                                            (list 'identical? '(nth pair 0) '(nth pair 1))))))
          a (compile-shape original) b (compile-shape transformed)
          input (case hint long 1000 double 5.125 (Long/valueOf 1000))]
      (is (= (nil? hint) (a input)) "Boxed aliases stay identical; primitive inputs box per field")
      (is (= (a input) (b input))))))

(deftest ordinary-object-inputs-preserve-reference-and-nullness
  (doseq [input [nil false :keyword "value" (Object.) (double-array [1.0])
                 (Long/valueOf 1000) (Double/valueOf -0.0) (Double/valueOf Double/NaN)]]
    (let [result (run (snapshot-product '[input input]) input)]
      (is (identical? input (first result)))
      (is (identical? (first result) (second result))))))

(deftest numeric-bits-are-not-numeric-conversions
  (doseq [input [(Double/valueOf -0.0)
                 (Double/valueOf (Double/longBitsToDouble 9221120237041090626))]]
    (let [result (run (completed 'input) input)]
      (is (identical? input result))
      (is (= (Double/doubleToRawLongBits input) (Double/doubleToRawLongBits result)))))
  (doseq [input [(Float/valueOf (float -0.0))
                 (Float/valueOf (Float/intBitsToFloat 2143289410))]]
    (let [result (run (completed 'input) input)]
      (is (identical? input result))
      (is (= (Float/floatToRawIntBits input) (Float/floatToRawIntBits result))))))

(deftest nested-products-and-opaque-observers-keep-reference-relations
  (let [original '[[input (do nil)]]
        transformed (snapshot-product [(snapshot-product '[input (do nil)])])
        input (Object.)
        observes (fn [product]
                   (list 'let* ['pair product 'alias 'pair]
                         ['(identical? pair alias)
                          '(identical? (ffirst (identity alias)) input)]))]
    (is (= [true true] (run (observes original) input)))
    (is (= [true true] (run (observes transformed) input)))))

(deftest arrays-inside-completed-products-remain-borrowed-references
  (let [input (double-array [1.0])
        product (run (snapshot-product '[input]) input)]
    (aset input 0 9.0)
    (is (identical? input (first product)))
    (is (= 9.0 (aget ^doubles (first product) 0)))))

(deftest original-effect-and-trap-regions-are-not-speculated
  (let [fields '[(do (swap! input conj :first) 7)
                 (if false (/ 1 0) (do (swap! input conj :selected) false))
                 (loop* [i 0] (if (< i 2) (recur (inc i)) i))]]
    (doseq [expression [fields (snapshot-product fields)]]
      (let [input (atom [])]
        (is (= [7 false 2] (run expression input)))
        (is (= [:first :selected] @input)))))
  (let [fields '[(do (swap! input conj :first) 7)
                 (do (swap! input conj :trap) (/ 1 0))
                 (do (swap! input conj :unreachable) 9)]]
    (doseq [expression [fields (snapshot-product fields)]]
      (let [input (atom [])]
        (is (thrown? ArithmeticException (run expression input)))
        (is (= [:first :trap] @input))))))

(deftest quoted-data-and-value-declarations-stay-distinct
  (let [operand (with-meta 'input {:raster.type/tag 'double :line 4 :column 2 :file "sample.clj"
                                   :tag 'double :ownership :fresh :raster.op/original 'opaque/op})
        term (completed operand)
        projected (source term)]
    (is (= {:raster.type/tag 'double :line 4 :column 2 :file "sample.clj"} (meta term)))
    (is (= (meta term) (meta projected)))
    (is (identical? Object (second (second projected))))
    (is (= operand (get-in (completion/canonical-parts term) [:operand]))))
  (let [invalid (list completion/completion-symbol {:policy :erase-checks} 'input)
        quoted (list 'quote [invalid])]
    (is (= quoted (source quoted))))
  (let [expression (list 'let* ['Object String]
                         (completed 'input))
        input (Object.)]
    (is (identical? input (run expression input)))))

(deftest malformed-or-forged-completions-do-not-become-identity
  (is (not= completion/completion-symbol
            'raster.compiler.core.value-completion/complete-value)
      "Reserved syntax must not accidentally invoke the constructor as a runtime operation")
  (doseq [contract [nil {} (assoc completion/object-input-contract :ownership :fresh)
                    {:source-context :primitive :target-carrier :jvm-object
                     :policy :preserve-source-object}]]
    (is (= :unsupported-value-completion
           (try (completion/complete-value contract 'input) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (doseq [expression [(list completion/completion-symbol)
                      (list completion/completion-symbol completion/object-input-contract)
                      (list completion/completion-symbol {:policy :erase-checks} 'input)]]
    (is (= :invalid-value-completion
           (try (source expression) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (let [term (completed 'input) projected (source term)]
    (is (= projected (completion/verified-jvm-source-projection term projected)))
    (is (= :value-completion-projection-mismatch
           (try (completion/verified-jvm-source-projection term 'input) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
