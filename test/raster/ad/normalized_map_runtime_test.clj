(ns raster.ad.normalized-map-runtime-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-surface-map-test :as surface]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- bundle [program]
  (normalized/linearize-map-program program (the-ns 'raster.ad.normalized-map-runtime-test)))
(defn- args [prepared environment] (mapv environment (:inputs prepared)))
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest real-surface-map-snapshots-and-explicit-seed-pairing
  (doseq [[v dt array cast] [[#'surface/square-double :double double-array double]
                             [#'surface/square-float :float float-array float]]
          size [0 3]]
    (let [prepared (runtime/prepare (bundle (#'surface/source-program v dt)))
          original (vec (take size [-2.0 0.0 3.0]))
          x (array original)
          arguments (args prepared {'xs x 'size size})
          dx (array (repeat size 0.25)) dy (array (repeat size -0.5))
          [output pullback] ((:vjp prepared) arguments)
          [jvp-output tangent] ((:jvp prepared) arguments dx)]
      (is (= (vec (v x size)) (vec output) (vec jvp-output)))
      (is (= (class x) (class output) (class tangent)))
      (dotimes [i size] (aset x i (cast 99.0)) (aset output i (cast -99.0)))
      (let [slots (pullback dy)
            gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) 'xs))]
        (is (= (mapv #(cast (* 2.0 % -0.5)) original) (vec gradient)))
        (is (= (mapv #(when (some? %) (vec %)) slots)
               (mapv #(when (some? %) (vec %)) (pullback dy))))
        (is (= (reduce + (map * (seq tangent) (seq dy)))
               (reduce + (map * (seq dx) (seq gradient)))))
        (dotimes [i size] (aset gradient i (cast 123.0)))
        (is (= (mapv #(cast (* 2.0 % -0.5)) original)
               (vec (nth (pullback dy) (.indexOf ^java.util.List (:inputs prepared) 'xs))))))
      (is (= (vec original) (mapv #(cast (/ % 0.5)) (seq tangent)))))))

(deftest inactive-captures-and-coordinate-finite-differences
  (let [program (#'element/map-program :double
                                      {:captures '[scale bias]
                                       :body '(raster.numeric/+ (raster.numeric/* element scale) bias)})
        prepared (runtime/prepare (bundle program))
        x (double-array [2.0 -1.0])
        environment {'n 2 'x x 'scale 3.0 'bias 5.0}
        arguments (args prepared environment)
        [output pullback] ((:vjp prepared) arguments)
        seed (double-array [0.5 -2.0])
        slots (pullback seed)
        gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) 'x))]
    (is (= [11.0 2.0] (vec output)))
    (is (= [1.5 -6.0] (vec gradient)))
    (doseq [[id slot] (map vector (:inputs prepared) slots)]
      (is (= (= id 'x) (some? slot))))
    (doseq [i (range 2)]
      (let [h 1.0e-5
            objective (fn [delta]
                        (let [xs (aclone x)]
                          (aset xs i (+ (aget xs i) delta))
                          (reduce + (map * (seq seed)
                                         (seq (first ((:vjp prepared)
                                                      (args prepared (assoc environment 'x xs)))))))))
            fd (/ (- (objective h) (objective (- h))) (* 2.0 h))]
        (is (< (Math/abs (- fd (aget ^doubles gradient i))) 1.0e-8))))))

(deftest arguments-must-match-checked-tensor-and-scalar-spaces
  (let [prepared (runtime/prepare (bundle (#'element/map-program :double {:captures '[scale]})))
        valid {'n 2 'x (double-array [1.0 2.0]) 'scale 3.0}]
    (is (= :ad-map-runtime-arity (reason #((:vjp prepared) []))))
    (doseq [bad [(assoc valid 'n -1) (assoc valid 'n (inc (long Integer/MAX_VALUE)))]]
      (is (= :ad-map-runtime-extent (reason #((:vjp prepared) (args prepared bad))))))
    (is (= :ad-map-runtime-extent-type
           (reason #((:vjp prepared) (args prepared (assoc valid 'n (int 2)))))))
    (doseq [bad [(assoc valid 'x (float-array 2)) (assoc valid 'x (double-array 3))]]
      (is (= :ad-map-runtime-array (reason #((:vjp prepared) (args prepared bad))))))
    (is (= :ad-map-runtime-capture
           (reason #((:vjp prepared) (args prepared (assoc valid 'scale (float 3.0)))))))
    (let [[_ pullback] ((:vjp prepared) (args prepared valid))]
      (is (= :ad-map-runtime-array (reason #(pullback (double-array 1)))))
      (is (= :ad-map-runtime-array (reason #(pullback (float-array 2)))))
      (is (= :ad-map-runtime-array
             (reason #((:jvp prepared) (args prepared valid) (double-array 1))))))))

(deftest nonindexed-context-preserves-port-order-errors-and-minimal-saves
  (doseq [[dt array cast] [[:float float-array float] [:double double-array double]]
          [dimension-dtype dimension-cast wrong-dimension-cast]
          [[:long long int] [:int int long]]]
    (let [source (#'element/map-program
                  dt {:captures '[scale bias]
                      :body '(raster.numeric/+ (raster.numeric/* element scale) bias)
                      :values {'n (av/tensor {:dtype dimension-dtype :shape []})}})
          ;; Public input order is not executable capture/array order. Bias is
          ;; needed by the primal, but not by the derivative's retained state.
          source (soac/make (-> (soac/facts source)
                                (assoc :inputs '[scale x n bias]))
                            (soac/equations source) (soac/outputs source))
          prepared (runtime/prepare (bundle source))
          x (array [1.0 2.0])
          valid {'scale (cast 3.0) 'x x
                 'n (dimension-cast 2) 'bias (cast 5.0)}
          wrong-scalar (if (= dt :float) (double 3.0) (float 3.0))
          call (fn [environment] ((:vjp prepared) (args prepared environment)))]
      (is (= '[scale x n bias] (:inputs prepared)))
      (is (= :ad-map-runtime-arity (reason #((:vjp prepared) []))))
      (doseq [extent [-1 (inc (long Integer/MAX_VALUE)) 2.0]]
        (is (= :ad-map-runtime-extent
               (reason #(call (assoc valid 'n extent 'x nil 'scale wrong-scalar))))))
      (is (= :ad-map-runtime-extent-type
             (reason #(call (assoc valid 'n (wrong-dimension-cast 2) 'x nil)))))
      (is (= :ad-map-runtime-array
             (reason #(call (assoc valid 'x (array 1) 'scale wrong-scalar)))))
      (is (= :ad-map-runtime-capture
             (reason #(call (assoc valid 'scale wrong-scalar)))))
      (let [[output pullback] (call valid)
            invoke-var #'runtime/invoke
            invoke @invoke-var
            saved (atom nil)
            seed (array [0.5 -1.0])
            gradients (with-redefs-fn
                        {invoke-var (fn [executable environment]
                                      (reset! saved environment)
                                      (invoke executable environment))}
                        #(pullback seed))]
        (is (= (mapv cast [8.0 11.0]) (vec output)))
        (is (= [nil (mapv cast [1.5 -3.0]) nil nil]
               (mapv #(when (some? %) (vec %)) gradients)))
        (is (= (dimension-cast 2) (get @saved 'n)))
        (is (= (cast 3.0) (get @saved 'scale)))
        (is (not-any? #(contains? @saved %) '[bias x]))
        (is (not-any? #(or (identical? x %) (identical? output %)) (vals @saved)))
        (is (= #{'n 'scale}
               (set (filter #(contains? valid %) (keys @saved)))))))))

(deftest direct-port-shapes-use-only-literal-or-typed-integral-dimensions
  (let [long-shape (av/tensor {:dtype :long :shape []})
        int-shape (av/tensor {:dtype :int :shape []})
        float-shape (av/tensor {:dtype :float :shape []})
        stable-id [:shape :rows]
        values {'n long-shape 'rows int-shape stable-id long-shape 'bad float-shape}
        environment {'n (long 2) 'rows (int 2) stable-id (long 2) 'bad (float 2.0)}
        check (fn [value abstract role]
                (#'runtime/source-value! value abstract environment values role))]
    (let [value (double-array 2)]
      (is (identical? value
                      (check value (av/tensor {:dtype :double :shape [2]}) :literal))))
    (is (= 2 (count (check (float-array 2)
                           (av/tensor {:dtype :float :shape ['n]}) :long-dimension))))
    (is (= 2 (count (check (double-array 2)
                           (av/tensor {:dtype :double :shape ['rows]}) :int-dimension))))
    (is (= 2 (count (check (double-array 2)
                           (av/tensor {:dtype :double
                                       :shape [(list 'value stable-id)]})
                           :wrapped-dimension))))
    ;; An ordinary integral scalar is type checked, but is not silently treated as a dimension.
    (is (= (long -3) (#'runtime/source-value! (long -3) long-shape {} values :ordinary-long)))
    (doseq [abstract [(assoc (av/tensor {:dtype :double :shape []}) :shape 2)
                      (assoc (av/tensor {:dtype :double :shape []}) :shape '(n))
                      (av/tensor {:dtype :double :shape ['n 'rows]})
                      (av/tensor {:dtype :double :shape '[(clojure.core/+ n 1)]})
                      (av/tensor {:dtype :double :shape ['missing]})
                      (av/tensor {:dtype :double :shape ['bad]})]]
      (is (= :ad-map-runtime-shape
             (reason #(check (double-array 2) abstract :unsupported-shape)))))
    (is (= :ad-map-runtime-extent-type
           (reason #(#'runtime/source-value!
                      (double-array 2) (av/tensor {:dtype :double :shape ['n]})
                      (assoc environment 'n (int 2)) values :wrong-long-class))))
    (is (= :ad-map-runtime-shape
           (reason #(#'runtime/differential-value!
                      (int 1) int-shape environment values :integral-differential))))))

(deftest generated-results-are-checked-against-their-own-ports
  (let [prepared (runtime/prepare (bundle (#'element/map-program :double)))
        arguments (args prepared {'n (long 2) 'x (double-array [1.0 2.0])})
        invoke-var #'runtime/invoke
        invoke @invoke-var]
    (with-redefs-fn {invoke-var (fn [& _] [])}
      (fn [] (is (= :ad-map-runtime-results
                     (reason (fn [] ((:vjp prepared) arguments)))))))
    (with-redefs-fn
      {invoke-var (fn [executable environment]
                    (assoc (vec (invoke executable environment)) 0 (double-array 1)))}
      (fn [] (is (= :ad-map-runtime-array
                     (reason (fn [] ((:vjp prepared) arguments)))))))
    (with-redefs-fn
      {invoke-var (fn [executable environment]
                    (let [results (vec (invoke executable environment))]
                      (if (< 1 (count results))
                        (assoc results 1 (double-array 1))
                        results)))}
      (fn [] (is (= :ad-map-runtime-array
                     (reason (fn [] ((:vjp prepared) arguments)))))))
    (let [calls (atom 0)]
      (with-redefs-fn
        {invoke-var (fn [executable environment]
                      (let [results (vec (invoke executable environment))]
                        (if (= 2 (swap! calls inc)) [(double-array 1)] results)))}
        (fn []
          (is (= :ad-map-runtime-array
                 (reason (fn [] ((:jvp prepared) arguments
                                 (double-array [0.25 0.5])))))))))
    (let [calls (atom 0)]
      (with-redefs-fn
        {invoke-var (fn [executable environment]
                      (if (= 2 (swap! calls inc)) [] (invoke executable environment)))}
        (fn []
          (is (= :ad-map-runtime-results
                 (reason (fn [] ((:jvp prepared) arguments
                                 (double-array [0.25 0.5])))))))))))

(deftest delayed-map-pullback-retains-dimensions-and-owned-snapshots-only
  (let [prepared (runtime/prepare (bundle (#'element/map-program :double)))
        x (double-array [1.0 2.0])
        arguments (args prepared {'n (long 2) 'x x})
        [output pullback] ((:vjp prepared) arguments)
        invoke-var #'runtime/invoke
        invoke @invoke-var
        environments (atom [])]
    (with-redefs-fn
      {invoke-var (fn [executable environment]
                    (swap! environments conj environment)
                    (invoke executable environment))}
      #(is (= [1.0 4.0]
              (vec (nth (pullback (double-array [0.5 1.0]))
                        (.indexOf ^java.util.List (:inputs prepared) 'x))))))
    (is (= 1 (count @environments)))
    (let [environment (first @environments)]
      (is (= (long 2) (get environment 'n)))
      (is (not-any? #(or (identical? x %) (identical? output %)) (vals environment)))
      (is (some #(instance? double/1 %) (vals environment))))))

(deftest runtime-does-not-sidestep-bundle-validation
  (let [checked (bundle (#'element/map-program :double))]
    (is (some? (reason #(runtime/prepare (assoc checked :certificate {})))))
    (is (= :ad-program-bundle-scope
           (reason #(runtime/prepare (normalized/linearize-map-element
                                      (#'element/map-program :double)
                                      (the-ns 'raster.ad.normalized-map-runtime-test))))))))

(deftest stale-rules-refuse-before-primal-or-pullback-execution
  (let [checked (bundle (#'element/map-program :double))
        prepared (runtime/prepare checked)
        arguments (args prepared {'n 2 'x (double-array [1.0 2.0])})
        [_ pullback] ((:vjp prepared) arguments)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
        current (revisions)
        invoke (ns-resolve 'raster.ad.normalized-map-runtime 'invoke)
        execution-count (atom 0)]
    (with-redefs-fn {revisions (constantly (update current :template-revision inc))
                    invoke (fn [& _] (swap! execution-count inc))}
      (fn []
        (is (= :ad-normalized-stale-rules (reason #((:vjp prepared) arguments))))
        (is (= :ad-normalized-stale-rules
               (reason #((:jvp prepared) arguments (double-array 2)))))
        (is (= :ad-normalized-stale-rules (reason #(pullback (double-array 2)))))
        (is (zero? @execution-count))))))
