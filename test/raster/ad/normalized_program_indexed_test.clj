(ns raster.ad.normalized-program-indexed-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-indexed-map-test :as indexed]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.par :as par]))

(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- build [dt it extent]
  (canonical/adapt-indexed-map
   (normalized/linearize-indexed-map-program (#'indexed/indexed-map dt it extent 'm)
                                           (the-ns 'raster.ad.normalized-program-indexed-test))))
(defn- recertify [b] (assoc b :certificate (#'canonical/certificate b)))
(defn- recertify-both [b]
  (recertify (assoc-in b [:derivation :bundle :certificate]
                      (#'normalized/certificate (get-in b [:derivation :bundle])))))
(defn- executable [b phase]
  (canonical/validate! b)
  (let [p (route/materialize-host-indexed-map (get b phase))]
    (assoc p :execute (binding [*ns* (:source-ns b)]
                       (eval (list 'fn (:parameters p) (:body p)))))))
(defn- invoke [p env]
  (apply (:execute p) (mapv env (concat (:inputs p) (:physical-inputs p)))))

(deftest canonical-indexed-programs-preserve-spaces-snapshot-and-physical-obligation
  (doseq [[dt array cast] [[:float float-array float] [:double double-array double]]
          [it indices-array] [[:int int-array] [:long long-array]] n [0 3]]
    (let [b (build dt it (if (zero? n) 0 'n))
          interface (:interface b) indices0 (vec (take n [2 0 2]))
          indices (indices-array indices0) data (array [2.0 -1.0 5.0])
          env {'indices indices 'data data 'n (long n) 'm (long 3)}
          p (executable b :primal) l (executable b :linear) t (executable b :transpose)
          [y snapshot] (invoke p env)
          residual (get-in b [:residual-bindings 'indices])
          [zero obligation] (first (:physical-bindings b))
          seed (array (take n [0.5 2.0 -1.0])) dx (array [0.25 0.5 -0.25])
          saved {'n (long n) 'm (long 3) (:id residual) snapshot}
          owner (array 3)
          [gradient] (invoke t (assoc saved zero owner (get-in interface [:cotangent-input :id]) seed))
          [tangent] (invoke l (assoc saved (get-in interface [:tangent-input :id]) dx))]
      (is (identical? b (canonical/validate! b)))
      (is (= '[data] (:active-inputs interface)))
      (is (= '[m] (get-in interface [:tangent-input :value :shape])
             (get-in interface [:cotangent-result :value :shape])))
      (is (= [(if (zero? n) 0 'n)] (get-in interface [:cotangent-input :value :shape])))
      (is (= {:equation 'g :input 'indices} (:origin residual)))
      (is (= (:id residual) (get-in residual [:producer :result])))
      (is (= it (get-in residual [:value :dtype])))
      (is (= :owned (get-in residual [:value :ownership])))
      (is (not (identical? indices snapshot)))
      (is (not (identical? y snapshot)))
      (is (= indices0 (vec snapshot)))
      (is (= (mapv #(cast (nth [2.0 -1.0 5.0] %)) indices0) (vec y)))
      (is (= {:value zero :dtype dt :shape '[m] :ownership :owned
              :initialization :typed-zero :exclusive true} (:requirement obligation)))
      (is (= zero (get-in obligation [:consumer :alias]) (get-in obligation [:consumer :destination])))
      (is (= (get-in interface [:cotangent-result :id]) (get-in obligation [:consumer :result])))
      (is (= [zero] (:physical-inputs t)))
      (doseq [phase [:source :primal :linear :transpose]]
        (is (not (some #{'m zero} (:inputs (soac/facts (get b phase))))))
        (is (= :long (get-in (soac/facts (get b phase)) [:values 'm :dtype]))))
      (is (identical? owner gradient))
      (let [expected (reduce (fn [g [index dy]] (update g index #(cast (+ % dy))))
                             (vec (repeat 3 (cast 0.0))) (map vector indices0 (seq seed)))]
        (is (= expected (vec gradient))))
      (is (= (mapv #(nth (vec dx) %) indices0) (vec tangent)))
      (is (= (reduce + 0.0 (map * (seq tangent) (seq seed)))
             (reduce + 0.0 (map * (seq dx) (seq gradient)))))
      ;; Caller-owned data/index/output mutation cannot change the owned residual.
      (dotimes [i n]
        (case it :int (aset ^ints indices i (int 1)) :long (aset ^longs indices i (long 1)))
        (aset y i (cast 99.0)))
      (dotimes [i 3] (aset data i (cast -99.0)) (aset gradient i (cast 77.0)))
      (let [next-owner (array 3)
            [next-gradient] (invoke t (assoc saved zero next-owner
                                            (get-in interface [:cotangent-input :id]) seed))]
        (is (identical? next-owner next-gradient))
        (is (not (identical? gradient next-gradient)))
        (is (= indices0 (vec snapshot)))
        (is (= (reduce (fn [g [index dy]] (update g index #(cast (+ % dy))))
                       (vec (repeat 3 (cast 0.0))) (map vector indices0 (seq seed)))
               (vec next-gradient)))))))

(deftest ordered-float-scatter-and-double-independent-finite-differences
  (let [b (build :float :long 'n) t (executable b :transpose)
        [zero _] (first (:physical-bindings b))
        snapshot (get-in b [:residual-bindings 'indices :id])
        seed (get-in b [:interface :cotangent-input :id]) owner (float-array 3)
        [gradient] (invoke t {'n (long 3) snapshot (long-array [1 1 1]) zero owner
                             seed (float-array [16777216.0 1.0 -16777216.0])})]
    (is (= [0.0 0.0 0.0] (vec gradient)) "each original Float update rounds in source order"))
  (let [b (build :double :int 'n) t (executable b :transpose)
        [zero _] (first (:physical-bindings b))
        snapshot (get-in b [:residual-bindings 'indices :id])
        seed (get-in b [:interface :cotangent-input :id]) indices [2 0 2] dy [0.5 2.0 -1.0]
        [gradient] (invoke t {'n (long 3) snapshot (int-array indices) zero (double-array 3)
                             seed (double-array dy)})
        objective (fn [data] (reduce + (map #(double (* (nth data %1) %2)) indices dy)))
        original [2.0 -1.0 5.0] h 0.125]
    (doseq [i (range 3)]
      (is (= (aget ^doubles gradient i)
             (/ (- (objective (update original i + h)) (objective (update original i - h))) (* 2.0 h)))))))

(deftest recertified-canonical-and-nested-evidence-tampering-refuses
  (let [b (build :double :long 'n) snapshot (get-in b [:residual-bindings 'indices :id])
        [zero _] (first (:physical-bindings b))]
    (doseq [changed [(assoc-in b [:residual-bindings 'indices :origin :input] 'data)
                     (assoc-in b [:residual-bindings 'indices :producer :result] 'out)
                     (assoc-in b [:residual-bindings 'indices :value :dtype] :int)
                     (assoc-in b [:interface :active-inputs] '[indices])
                     (assoc-in b [:interface :tangent-input :value :shape] '[n])
                     (assoc-in b [:physical-bindings zero :requirement :initialization] :unknown)
                     (assoc-in b [:physical-bindings zero :requirement :exclusive] false)
                     (assoc-in b [:physical-bindings zero :consumer :alias] snapshot)]]
      (is (= :ad-program-bundle-correspondence (reason #(canonical/validate! (recertify changed))))))
    (doseq [changed [(assoc-in b [:derivation :bundle :gather-programs :required-zero-gradient :ownership] :borrowed)
                     (assoc-in b [:derivation :bundle :gather-programs :interface :index-snapshot] 'data)]]
      (is (contains? #{:ad-normalized-gather-programs :ad-normalized-gather-interface}
                     (reason #(canonical/validate! (recertify-both changed))))))
    (let [changed (assoc-in b [:derivation :bundle :gather-programs :transpose]
                           (let [p (:transpose b)]
                             (apply list (assoc (vec p) 1 (assoc-in (soac/facts p)
                                                                  [:values 'm :dtype] :float)))))]
      (is (= :ad-normalized-gather-programs (reason #(canonical/validate! (recertify-both changed))))))))

(deftest family-namespace-and-epoch-boundaries-stay-delegated
  (let [b (build :double :int 'n) evidence (get-in b [:derivation :bundle])]
    (doseq [facet [:map-programs :sum-programs]]
      (is (= :ad-program-bundle-scope
             (reason #(canonical/adapt-indexed-map (assoc evidence facet {}))))))
    (is (= :ad-program-bundle-source-namespace
           (reason #(canonical/validate! (assoc b :source-ns (the-ns 'user))))))
    (is (= :ad-normalized-stale-rules
           (reason #(canonical/validate!
                     (assoc-in b [:derivation :bundle :compiler-revision] -1)))))
    (is (fingerprint/equivalent? (:source b) (get-in b [:derivation :bundle :gather-programs :source])))))
