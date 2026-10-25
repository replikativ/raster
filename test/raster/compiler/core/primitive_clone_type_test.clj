(ns raster.compiler.core.primitive-clone-type-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.tc-extensions]
            [raster.compiler.passes.scalar.rewalk :as rewalk]
            [typed.clj.checker.check :as check]
            [typed.cljc.checker.check.invoke :as invoke]
            [typed.cljc.checker.object-rep :as object]
            [typed.cljc.checker.type-rep :as r]
            [typed.cljc.checker.utils :as u]))

(def ^:private primitive-families
  [[Double/TYPE 'doubles] [Float/TYPE 'floats] [Long/TYPE 'longs] [Integer/TYPE 'ints]
   [Byte/TYPE 'bytes] [Short/TYPE 'shorts] [Character/TYPE 'chars] [Boolean/TYPE 'booleans]])

(deftest primitive-clone-preserves-representation-not-source-identity
  (let [handler (get-method check/-invoke-special 'clojure.core/aclone)]
    (doseq [[component _] primitive-families]
      (let [array-type (r/PrimitiveArray-maker component r/-any r/-any)
            source-result (assoc (r/ret array-type) :o (object/-path nil 'source-array))
            calls (atom 0)
            argument {:op :local u/expr-type source-result}
            result (handler {:args [argument]} nil
                            {:typed.cljc.checker.check/check-expr
                             (fn [expression expected _]
                               (is (nil? expected))
                               (swap! calls inc)
                               expression)})
            returned (u/expr-type result)]
        (is (= 1 @calls))
        (is (= [argument] (:args result)))
        (is (= array-type (r/ret-t returned)))
        (is (= component (:jtype (r/ret-t returned))))
        (is (not (identical? source-result returned)))
        (is (= object/-empty (r/ret-o returned)))))))

(deftest clone-specialization-does-not-invent-a-primitive-representation
  (let [handler (get-method check/-invoke-special 'clojure.core/aclone)
        checked (fn [type] {:op :local u/expr-type (r/ret type)})
        options {:typed.cljc.checker.check/check-expr (fn [expression _ _] expression)}]
    (doseq [argument [(checked (r/PrimitiveArray-maker Object r/-any r/-any))
                      (checked (r/PrimitiveArray-maker String r/-any r/-any))
                      (checked r/-any)]]
      (with-redefs [invoke/normal-invoke
                    (fn [expression function arguments expected already-checked actual-options]
                      (is (= {} function))
                      (is (= [argument] arguments))
                      (is (nil? expected))
                      (is (= options actual-options))
                      (is (= {:cargs [argument]} already-checked))
                      expression)]
        (is (= {:fn {} :args [argument]} (handler {:fn {} :args [argument]} nil options))
            "reference arrays and unknown inputs retain ordinary checking without rechecking")))
    (is (nil? (handler {:args []} nil options)))
    (is (nil? (handler {:args [(checked r/-any) (checked r/-any)]} nil options)))))

(deftest primitive-array-clones-retain-their-type-through-tc-and-rewalking
  (doseq [[_ tag] primitive-families]
    (let [source '(let* [cloned (clojure.core/aclone input)] cloned)
          inferred (inference/tc-infer-binding-tags {'input tag} source)
          result (:form (rewalk/pe-rewalk source {:simplify? true :dtype (if (= tag 'floats) :float :double)
                                                :source-ns 'raster.compiler.core.primitive-clone-type-test
                                                :preserve-declared-array-storage? true
                                                :param-env {'input tag}}))]
      (is (= tag (get inferred 'cloned)))
      (is (= tag (:raster.type/tag (meta (first (second result)))))))))

(deftest clone-rule-retains-ordinary-type-errors
  (is (= 'doubles
         (:ret-tag (inference/tc-analyze-deftm-body
                    'valid-clone '[input] '[(Array double)]
                    '[(clojure.core.typed/ann-form (clojure.core/aclone input) (Array double))] *ns*))))
  (doseq [body ['(clojure.core/aclone input input)
                '(clojure.core/aclone 1)
                '(clojure.core.typed/ann-form (clojure.core/aclone input) (Array long))]]
    (is (nil? (inference/tc-analyze-deftm-body
               'invalid-clone '[input] '[(Array double)] [body] *ns*)))))

(deftest nested-reference-clones-check-each-argument-once
  (let [handler (get-method check/-invoke-special 'clojure.core/aclone)
        array-type (r/PrimitiveArray-maker Object r/-any r/-any)]
    (doseq [depth [1 4 16]]
      (let [checks (atom 0)
            source {:op :local u/expr-type (r/ret array-type)}
            expression (nth (iterate #(hash-map :op :invoke :fn {} :args [%]) source) depth)
            options {:typed.cljc.checker.check/check-expr
                     (fn [expression expected options]
                       (swap! checks inc)
                       (if (= :invoke (:op expression))
                         (handler expression expected options)
                         expression))}]
        (with-redefs [invoke/normal-invoke
                      (fn [expression _ _ _ {:keys [cargs]} _]
                        (assoc expression :args cargs u/expr-type (r/ret array-type)))]
          (is (= array-type (r/ret-t (u/expr-type (handler expression nil options)))))
          (is (= depth @checks) "generic fallback must not repeatedly check a nested subtree"))))))
