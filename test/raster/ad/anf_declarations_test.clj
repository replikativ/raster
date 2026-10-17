(ns raster.ad.anf-declarations-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse.normalize :as anf]))

(defn- fresh-symbols []
  (let [counter (atom 0)]
    (fn [prefix] (with-meta (symbol (str prefix "__" (swap! counter inc)))
                  {:test/generated true}))))

(deftest existing-binders-retain-newly-exposed-result-declarations
  (let [value (with-meta 'value {:line 4})
        initializer (with-meta '(producer input)
                      {:raster.type/tag 'floats :raster.op/original 'producer
                       :raster.effect/effectful true :ownership :not-a-proof})
        [result expression] (anf/anf-normalize-bindings [value initializer] (fresh-symbols))]
    (is (= {:line 4 :raster.type/tag 'floats} (meta result)))
    (is (= (meta initializer) (meta expression)))
    (is (nil? (:tag (meta result))))
    (is (= value result)))
  (let [[result _] (anf/anf-normalize-bindings
                    [(with-meta 'value {:raster.type/tag 'Object})
                     (with-meta '(producer input) {:raster.type/tag 'floats})]
                    (fresh-symbols))]
    (is (= 'Object (:raster.type/tag (meta result))))
    (is (nil? (:tag (meta result)))))
  (let [[result _] (anf/anf-normalize-bindings '[value (unknown input)] (fresh-symbols))]
    (is (nil? (:raster.type/tag (meta result)))))
  (doseq [tag ['floats 'doubles 'Object]]
    (let [metadata {:tag tag :line 5}
          [result _] (anf/anf-normalize-bindings
                       [(with-meta 'value metadata)
                        (with-meta '(producer input) {:raster.type/tag 'floats})]
                       (fresh-symbols))]
      (is (= metadata (meta result))
          "neither matching nor conflicting existing JVM declarations are restamped"))))

(deftest lifted-arguments-retain-declared-types-without-inferring
  (doseq [tag ['int 'long 'float 'double 'doubles 'floats]]
    (let [argument (with-meta '(producer x)
                     {:raster.type/tag tag :line 12 :column 8 :file "source.clj"
                      :raster.op/original 'producer :raster.effect/effectful true
                      :ownership :not-a-proof})
          [bindings expression] (anf/anf-normalize-expr
                                 (list 'consumer argument) (fresh-symbols))
          [value initializer] bindings]
      (is (= {:test/generated true :raster.type/tag tag :line 12 :column 8 :file "source.clj"}
             (meta value)))
      (is (= value (second expression)))
      (is (= (meta argument) (meta initializer)) "initializer keeps its own operation facts")))
  (let [[bindings _] (anf/anf-normalize-expr '(consumer (unknown x)) (fresh-symbols))]
    (is (nil? (:raster.type/tag (meta (first bindings)))))))

(deftest discarded-statements-retain-only-their-own-value-declaration
  (let [statement (with-meta '(effect x) {:raster.type/tag 'int :raster.op/original 'effect})
        [bindings result] (anf/anf-normalize-expr (list 'do statement 3.0) (fresh-symbols))]
    (is (= 'int (:raster.type/tag (meta (first bindings)))))
    (is (nil? (:raster.op/original (meta (first bindings)))))
    (is (= {:raster.type/tag 'int :raster.op/original 'effect} (meta (second bindings))))
    (is (= 3.0 result))))

(deftest do-arguments-use-the-selected-result-declaration
  (let [result (with-meta '(double x) {:raster.type/tag 'double})
        argument (with-meta (list 'do '(effect x) result) {:raster.type/tag 'Object})
        [bindings _] (anf/anf-normalize-expr (list 'consumer argument) (fresh-symbols))]
    (is (= 'double (:raster.type/tag (meta (nth bindings 2)))))))

(deftest lifting-retains-sequencing-and-scope-boundaries
  (let [argument (with-meta '(do (swap! trace conj :first) (double 2.0))
                   {:raster.type/tag 'double})
        source (list '+ argument '(do (swap! trace conj :second) 3.0))
        [bindings result] (anf/anf-normalize-expr source (fresh-symbols))
        execute (fn [body]
                  (eval (list 'let ['trace '(atom [])] [body '(deref trace)])))]
    (is (= [5.0 [:first :second]] (execute source)))
    (is (= (execute source) (execute (list 'let bindings result)))))
  (doseq [scope ['(let* [x 2.0] x) '(loop* [i 0] (if (< i 1) (recur (inc i)) i))]]
    (let [[bindings result] (anf/anf-normalize-expr scope (fresh-symbols))]
      (is (empty? bindings))
      (is (= scope result)))))
