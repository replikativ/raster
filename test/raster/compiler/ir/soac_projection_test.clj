(ns raster.compiler.ir.soac-projection-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-indexed-map-test :as indexed]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- source [] (#'indexed/indexed-map :double :long 'n 'm))
(defn- with-facts [program f]
  (soac/make (f (soac/facts program)) (soac/equations program) (soac/outputs program)))

(deftest indexed-projection-preserves-capacity-without-adding-an-input
  (let [program (source)
        calls (atom 0) validate soac/validate!
        projections (with-redefs [soac/validate!
                                 (fn [p] (when (identical? p program) (swap! calls inc)) (validate p))]
                      (soac/equation-subprograms program))]
    (is (= 1 @calls))
    (doseq [projected (conj projections (soac/equation-subprogram program (first (soac/equations program))))]
      (is (= '[indices data n] (:inputs (soac/facts projected))))
      (is (identical? (get-in (soac/facts program) [:values 'm])
                      (get-in (soac/facts projected) [:values 'm])))
      (is (= (soac/equations program) (soac/equations projected)))
      (is (some? (normalized/linearize-indexed-map-program projected
                                                         (the-ns 'raster.ad.normalized-indexed-map-test)))))))

(deftest projection-retains-only-declared-direct-shape-facts
  (doseq [dimension ['m 'count 'seq '(value :capacity) '(value [:capacity 0])]]
    (let [id (if (symbol? dimension) dimension (second dimension))
          fact (av/tensor {:dtype :long :shape [] :attributes {:source :retained}})
          program (with-facts (source)
                    #(-> % (assoc-in [:values 'data :shape] [dimension])
                         (assoc-in [:values id] fact)
                         (assoc-in [:values 'unrelated] (av/tensor {:dtype :double :shape []}))))
          projected (first (soac/equation-subprograms program))]
      (is (identical? fact (get-in (soac/facts projected) [:values id])))
      (is (= '[indices data n] (:inputs (soac/facts projected))))
      (is (= #{'indices 'data 'n 'out id} (set (keys (:values (soac/facts projected))))))
      (is (= [dimension] (get-in (soac/facts projected) [:values 'data :shape])))))
  (doseq [dimension [4 '(extent m) '(unknown-dimension m) '(+ m 1) '(custom-shape m)]]
    (let [program (with-facts (source) #(assoc-in % [:values 'data :shape] [dimension]))
          projected (first (soac/equation-subprograms program))]
      (is (= #{'indices 'data 'n 'out} (set (keys (:values (soac/facts projected))))))
      (is (= [dimension] (get-in (soac/facts projected) [:values 'data :shape]))))))

(deftest projection-preserves-denial-evidence-but-never-invents-a-fact
  (doseq [fact [(av/tensor {:dtype :float :shape []})
                (av/tensor {:dtype :long :shape [3]})]]
    (let [program (with-facts (source) #(assoc-in % [:values 'm] fact))
          projected (first (soac/equation-subprograms program))]
      (is (identical? fact (get-in (soac/facts projected) [:values 'm])))
      (is (thrown? clojure.lang.ExceptionInfo
                   (normalized/linearize-indexed-map-program projected
                                                            (the-ns 'raster.ad.normalized-indexed-map-test))))))
  (let [program (with-facts (source) #(update % :values dissoc 'm))
        projected (first (soac/equation-subprograms program))]
    (is (not (contains? (:values (soac/facts projected)) 'm)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (normalized/linearize-indexed-map-program projected
                                                          (the-ns 'raster.ad.normalized-indexed-map-test))))))

(deftest result-and-physical-destination-shapes-retain-their-own-facts
  (let [bundle (normalized/linearize-indexed-map-program
                (source) (the-ns 'raster.ad.normalized-indexed-map-test))
        transpose (get-in bundle [:gather-programs :transpose])
        destination (get-in bundle [:gather-programs :interface :zero-gradient])
        capacity (av/tensor {:dtype :long :shape [] :attributes {:source :physical-capacity}})
        program (with-facts transpose
                  #(-> % (assoc-in [:values destination :shape] '[physical-capacity])
                       (assoc-in [:values 'physical-capacity] capacity)
                       (assoc-in [:values 'unrelated] (av/tensor {:dtype :double :shape []}))
                       (assoc :provenance {:source :retained})
                       (update :attributes assoc :projection-marker :retained)))
        equation (first (soac/equations program))
        projected (soac/equation-subprogram program equation)
        original (soac/facts program) facts (soac/facts projected)]
    ;; Neither result-only m nor physical-only capacity is an executable dependency.
    (is (not (some #{'m 'physical-capacity} (:inputs facts))))
    (is (identical? (get-in original [:values 'm]) (get-in facts [:values 'm])))
    (is (identical? capacity (get-in facts [:values 'physical-capacity])))
    (is (not (contains? (:values facts) 'unrelated)))
    (is (= (:provenance original) (:provenance facts)))
    (is (= (:attributes original) (:attributes facts)))
    (is (identical? (get-in original [:equations (second equation)])
                    (get-in facts [:equations (second equation)])))
    (is (= (soac/result-storage program (second equation))
           (soac/result-storage projected (second equation))))))
