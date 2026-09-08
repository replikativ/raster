(ns raster.compiler.core.contraction-scope-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.walker :as walker]
            [raster.par]))

(defn walked [extra-types]
  (walker/walk
   '(raster.par/contract out [[i rows]] [[blk 2] [t 4]]
      (* (aget a (+ (* i 8) (* blk 4) t)) (aget b (+ (* blk 4) t)))
      :decode {a (- x shift)}
      :stages [{:axis blk :extent 2 :dtype :double :init 0.0 :lift (* inner gain)}
               {:axis t :extent 4 :dtype :float :init 0.0}]
      :epilogue {:acc value :expr (+ value bias) :dtype :double})
   (walker/make-ctx
    {:type-env (into {} (map (fn [[id tag]] [id {:tag tag}]))
                     (merge {'out 'doubles 'a 'floats 'b 'floats 'rows 'long
                             'shift 'float 'gain 'float 'bias 'double
                             'x 'long 'inner 'long 'value 'long} extra-types))})))

(deftest contraction-local-binders-have-authoritative-types
  (let [source (walked {})
        opts (apply hash-map (drop 5 source))]
    (is (not (contains? opts :decode)) "load transforms expand before their consumers are typed")
    (is (= 'float (:raster.type/tag (meta (get-in opts [:stages 0 :lift])))))
    (is (= 'double (:raster.type/tag (meta (get-in opts [:epilogue :expr])))))
    (is (= 'rows (second (first (nth source 2)))) "extent stays in the outer scope")
    (is (= 'float (:raster.type/tag (meta (nth source 4)))))))

(deftest raw-load-binder-type-is-not-inherited-from-an-outer-x
  (doseq [types [{} {'a 'bytes 'shift 'long} {'a 'Object}]]
    (let [source (walked types)]
      (is (not-any? #{'x} (tree-seq coll? seq (nth source 4)))
          "the placeholder is replaced hygienically before ordinary type inference"))))

(deftest unknown-lexical-bindings-mask-rewalk-metadata
  (let [x (with-meta 'x {:raster.type/tag 'long :tag 'long :line 17})
        unknown (walker/make-ctx {:type-env {'x {:tag nil}}})
        absent (walker/make-ctx {})]
    (is (nil? (walker/ctx-get-tag unknown x)))
    (is (= {:line 17} (meta (walker/walk x unknown))))
    (is (= 'long (walker/ctx-get-tag absent x)))
    (is (= 'long (:raster.type/tag (meta (walker/walk x absent)))))))

(deftest generated-casts-are-stable-across-rewalks
  (doseq [tag ['float 'long]]
    (let [ctx (walker/make-ctx {:type-env {'x {:tag tag}}})
          expression (list (symbol "clojure.core" (name tag)) 'x)]
      (let [walks (take 5 (rest (iterate #(walker/walk % ctx) expression)))]
        (is (every? #(= expression %) walks))
        (is (every? #(= tag (:raster.type/tag (meta %))) walks))))))
