(ns raster.compiler.huge-helper-outline-test
  "Oversized loop bodies are split below HotSpot's 8000-byte JIT limit."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk]
            [raster.core]
            [raster.compiler.core.outline :as outline]
            [raster.compiler.pipeline :as pipeline]))

(defn- chain-pairs
  "n chained double bindings v1..vn over v0, each reading the previous two."
  [n]
  (vec (mapcat (fn [k]
                 [(with-meta (symbol (str "v" k)) {:tag 'double})
                  (list 'clojure.core/+
                        (list 'clojure.core/* (symbol (str "v" (dec k))) (+ 1.0 (/ k 1.0e4)))
                        (list 'clojure.core/* (symbol (str "v" (max 0 (- k 2)))) -0.25))])
               (range 1 (inc n)))))

(defn- size-of [form]
  (cond (seq? form) (+ 20 (reduce + (map size-of (rest form))))
        (vector? form) (reduce + 0 (map size-of form))
        :else 5))

(deftest outline-splits-typed-blocks-inside-loops
  (let [n 300
        form (list 'loop* [(with-meta 'i {:tag 'long}) 0 (with-meta 'acc {:tag 'double}) 0.0]
                   (list 'if (list 'clojure.core/< 'i 'n)
                         (list 'let* (into [(with-meta 'v0 {:tag 'double}) (list 'clojure.core/aget 'xs 'i)]
                                           (chain-pairs n))
                               (list 'recur (list 'clojure.core/inc 'i)
                                     (list 'clojure.core/+ 'acc (symbol (str "v" n)))))
                         'acc))
        counter (atom 0)
        result (outline/outline form {:size-of size-of :block-threshold 2000 :chunk-budget 800
                                      :new-name #(symbol (str "chunk_" (swap! counter inc)))
                                      :env '{xs doubles n long}})]
    (testing "chunks are emitted with typed parameters and a double return"
      (is (< 1 (count (:helpers result))))
      (is (every? #(= 'double (:return-tag %)) (:helpers result)))
      (is (every? (fn [{:keys [params]}] (every? #(:tag (meta %)) params)) (:helpers result))))
    (testing "each chunk stays within the budget"
      (is (every? #(<= (size-of (:body %)) 2000) (:helpers result))))
    (testing "untyped symbols leave the form unchanged"
      (let [untyped (outline/outline (clojure.walk/postwalk #(if (symbol? %) (with-meta % nil) %) form)
                                     {:size-of size-of :block-threshold 2000 :chunk-budget 800
                                      :new-name #(gensym "chunk_") :env {}})]
        (is (empty? (:helpers untyped)))))))

(defn- reference [^doubles xs n]
  (loop [i 0 acc 0.0]
    (if (< i (alength xs))
      (let [vs (reduce (fn [vs k]
                         (conj vs (+ (* (nth vs (dec k)) (+ 1.0 (/ k 1.0e4)))
                                     (* (nth vs (max 0 (- k 2))) -0.25))))
                       [(aget xs i)] (range 1 (inc n)))]
        (recur (inc i) (+ acc (peek vs))))
      acc)))

(deftest compiled-huge-loop-matches-reference
  (let [n 600
        body (list 'loop ['i 0 'acc 0.0]
                   (list 'if (list '< 'i (list 'alength 'xs))
                         (list 'let (into ['v0 (list 'aget 'xs 'i)]
                                          (mapcat (fn [k]
                                                    [(symbol (str "v" k))
                                                     (list '+ (list '* (symbol (str "v" (dec k))) (+ 1.0 (/ k 1.0e4)))
                                                           (list '* (symbol (str "v" (max 0 (- k 2)))) -0.25))])
                                                  (range 1 (inc n))))
                               (list 'recur (list 'inc 'i) (list '+ 'acc (symbol (str "v" n)))))
                         'acc))
        name (gensym "huge_loop_")]
    (binding [*ns* (the-ns 'raster.compiler.huge-helper-outline-test)]
      (eval (list 'raster.core/deftm name '[xs :- (Array double)] :- 'Double body)))
    (let [f (pipeline/compile-aot (ns-resolve 'raster.compiler.huge-helper-outline-test name))
          xs (double-array (map #(Math/sin (double %)) (range 64)))
          expected (reference xs n)
          actual (double (f xs))]
      (is (< (Math/abs (- expected actual)) (* 1e-9 (max 1.0 (Math/abs expected))))))))

(deftest compiled-huge-straight-line-matches-reference
  ;; Small bindings stay inline in the compute orchestrator, so a long
  ;; straight-line body exercises orchestrator outlining.
  (let [n 800
        body (list 'let (into ['v0 (list 'aget 'xs 0) 'w0 (list 'aget 'xs 1)]
                              (mapcat (fn [k]
                                        [(symbol (str "v" k)) (list '+ (list '* (symbol (str "v" (dec k))) (+ 1.0 (/ k 1.0e5)))
                                                                    (list '* (symbol (str "w" (dec k))) -0.25))
                                         (symbol (str "w" k)) (list '- (symbol (str "w" (dec k))) (list '* (symbol (str "v" k)) 1.0e-3))])
                                      (range 1 (inc n))))
                   (list '+ (symbol (str "v" n)) (symbol (str "w" n))))
        name (gensym "huge_straight_")
        reference (fn [^doubles xs]
                    (loop [k 1 v (aget xs 0) w (aget xs 1)]
                      (if (> k n)
                        (+ v w)
                        (let [v' (+ (* v (+ 1.0 (/ k 1.0e5))) (* w -0.25))]
                          (recur (inc k) v' (- w (* v' 1.0e-3)))))))]
    (binding [*ns* (the-ns 'raster.compiler.huge-helper-outline-test)]
      (eval (list 'raster.core/deftm name '[xs :- (Array double)] :- 'Double body)))
    (let [f (pipeline/compile-aot (ns-resolve 'raster.compiler.huge-helper-outline-test name))
          xs (double-array [0.3 -0.7])
          expected (reference xs)
          actual (double (f xs))]
      (is (< (Math/abs (- expected actual)) (* 1e-9 (max 1.0 (Math/abs expected))))))))
