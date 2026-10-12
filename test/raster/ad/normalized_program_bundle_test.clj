(ns raster.ad.normalized-program-bundle-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-excluded-map-test :as excluded]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- recertify [bundle] (assoc bundle :certificate (#'canonical/certificate bundle)))
(defn- recertify-evidence [bundle]
  (recertify (assoc-in bundle [:derivation :bundle :certificate]
                      (#'normalized/certificate (get-in bundle [:derivation :bundle])))))
(defn- execute [bundle phase environment]
  (canonical/validate! bundle)
  (let [projection (route/materialize-host-map (get bundle phase))
        f (binding [*ns* (:source-ns bundle)]
            (eval (list 'fn (:parameters projection) (:body projection))))]
    (apply f (mapv environment (:inputs projection)))))
(defn- simple [dt options]
  (normalized/linearize-map-program (#'element/map-program dt options)
                                    (the-ns 'raster.ad.normalized-program-bundle-test)))

(deftest canonical-programs-execute-through-existing-host-materialization
  (doseq [[v dt array cast] [[#'excluded/weighted-double :double double-array double]
                             [#'excluded/weighted-float :float float-array float]]
          n [0 3] active ['xs 'weights]]
    (let [source (#'excluded/source v dt)
          old (#'excluded/build source active) b (canonical/adapt-map old)
          prepared (runtime/prepare old)
          xs0 (vec (take n [-2.0 0.0 3.0])) ws0 (vec (take n [0.5 2.0 -1.0]))
          xs (array xs0) weights (array ws0) env {'xs xs 'weights weights 'size (long n)}
          dx (array (repeat n 0.25)) dy (array (repeat n -0.5))
          results (execute b :primal env)
          owned (map :id (filter #(= :owned-array (:storage %)) (vals (:residual-bindings b))))
          ;; Only immutable source scalars and the declared owned snapshots enter derivatives.
          saved (merge {'size (long n)} (select-keys (zipmap (soac/outputs (:primal b)) results) owned))
          interface (:interface b)
          tangent-id (get-in interface [:tangent-input :id]) seed-id (get-in interface [:cotangent-input :id])
          tangent (first (execute b :linear (assoc saved tangent-id dx)))
          [old-y old-pb] ((:vjp prepared) (mapv env (:inputs prepared)))
          old-gradient (nth (old-pb dy) (.indexOf ^java.util.List (:inputs prepared) active))
          expected (mapv (fn [x w] (cast (* -0.5 (if (= active 'xs) (* 2.0 x w) (* x x))))) xs0 ws0)]
      (is (identical? source (:source b)))
      (is (identical? b (canonical/validate! b)))
      (doseq [phase [:primal :linear :transpose]]
        (is (identical? (get b phase) (soac/validate! (get b phase)))))
      (is (= :checked-scalar-map (get-in b [:derivation :kind])))
      (is (= (vec (v xs weights n)) (vec (first results)) (vec old-y)))
      (is (= (mapv :id (get-in b [:interface :inputs])) (:inputs prepared)))
      (is (= [active] (get-in b [:interface :active-inputs])))
      (is (empty? (:physical-bindings b)))
      (doseq [[coefficient binding] (:residual-bindings b)]
        (is (= :owned-array (:storage binding)))
        (is (= :owned (get-in binding [:value :ownership])))
        (is (= (:id binding) (get-in binding [:producer :result])))
        (is (= coefficient (get-in binding [:origin :scalar-value])))
        (is (not (identical? xs (get saved (:id binding)))))
        (is (not (identical? weights (get saved (:id binding))))))
      (dotimes [i n] (aset xs i (cast 99.0)) (aset weights i (cast 88.0))
        (aset (first results) i (cast -77.0)))
      (let [g (first (execute b :transpose (assoc saved seed-id dy)))]
        (is (= expected (vec g) (vec old-gradient)))
        (is (= (reduce + 0.0 (map * (seq tangent) (seq dy)))
               (reduce + 0.0 (map * (seq dx) (seq g)))))
        (dotimes [i n] (aset g i (cast 42.0)))
        (is (= expected (vec (first (execute b :transpose (assoc saved seed-id dy))))))))))

(deftest scalar-capture-and-float-checkpoint-contracts-remain-exact
  (let [b (canonical/adapt-map (simple :double {:captures '[scale] :body '(raster.numeric/* element scale)}))
        binding (first (vals (:residual-bindings b)))]
    (is (= :immutable-scalar (:storage binding)))
    (is (= 'scale (get-in binding [:producer :input]) (:id binding)))
    (is (= [] (get-in binding [:value :shape])))
    (is (= [6.0] (vec (first (execute b :primal {'x (double-array [2.0]) 'n 1 'scale 3.0})))))
    (is (= [1.5] (vec (first (execute b :linear {'n 1 'scale 3.0
                                               (get-in b [:interface :tangent-input :id]) (double-array [0.5])}))))))
  (let [b (canonical/adapt-map (simple :float {:body (list 'raster.numeric/+ 'element (float 1.0))}))
        [y] (execute b :primal {'x (float-array [16777216.0]) 'n 1})]
    (is (= (Float/floatToRawIntBits (float 16777216.0)) (Float/floatToRawIntBits (aget ^floats y 0)))))
  (let [old (simple :double {})
        source (get-in old [:map-programs :source])
        facts (reduce (fn [facts id] (assoc-in facts [:values id :memory-space] :device))
                      (soac/facts source) '[x y])
        placed (soac/make facts (soac/equations source) (soac/outputs source))
        b (canonical/adapt-map (normalized/linearize-map-program placed (:source-ns old)))]
    (is (= :device (get-in b [:interface :primal-result :value :memory-space])))
    (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map (:primal b)))))))

(deftest independent-ordinary-primal-finite-difference
  (let [old (#'excluded/build (#'excluded/source #'excluded/weighted-double :double) 'xs)
        b (canonical/adapt-map old) xs (double-array [-2.0 3.0]) ws (double-array [0.5 -1.0])
        seed (double-array [0.25 -0.5]) results (execute b :primal {'xs xs 'weights ws 'size 2})
        residual-ids (map :id (vals (:residual-bindings b)))
        saved (merge {'size 2} (select-keys (zipmap (soac/outputs (:primal b)) results) residual-ids))
        gradient (first (execute b :transpose (assoc saved (get-in b [:interface :cotangent-input :id]) seed)))]
    (doseq [i (range 2)]
      (let [h 1.0e-5
            objective (fn [delta]
                        (let [x (aclone xs)]
                          (aset x i (+ (aget xs i) delta))
                          (reduce + 0.0 (map * (seq seed) (seq (excluded/weighted-double x ws 2))))))
            fd (/ (- (objective h) (objective (- h))) (* 2.0 h))]
        (is (< (Math/abs (- fd (aget ^doubles gradient i))) 1.0e-8))))))

(deftest recertification-cannot-rewrite-producers-interfaces-or-ownership
  (let [b (canonical/adapt-map (simple :double {}))
        coefficient (first (keys (:residual-bindings b)))]
    (doseq [change [#(assoc-in % [:interface :active-inputs] [])
                    #(assoc-in % [:interface :inputs 0 :value :dtype] :float)
                    #(assoc-in % [:interface :cotangent-input :source] 'other)
                    #(assoc-in % [:residual-bindings coefficient :producer :equation] 'other)
                    #(assoc-in % [:residual-bindings coefficient :producer :result] 'other)
                    #(assoc-in % [:residual-bindings coefficient :value :ownership] :borrowed)
                    #(assoc-in % [:physical-bindings 'fake] {:initialization :typed-zero})
                    #(assoc % :source-namespace 'forged.namespace)
                    #(assoc % :linear (:transpose %))]]
      (is (= :ad-program-bundle-correspondence (reason #(canonical/validate! (recertify (change b)))))))
    (is (= :ad-program-bundle-certificate (reason #(canonical/validate! (assoc b :certificate "forged")))))
    (is (= :ad-program-bundle-source-namespace
           (reason #(canonical/validate! (assoc b :source-ns (the-ns 'raster.ad.normalized-scalar))))))
    (is (= :ad-program-bundle-source-namespace
           (reason #(canonical/validate! (assoc b :source-ns (ns-name (:source-ns b)))))))
    (is (= :semantic-fingerprint-unsupported (reason #(recertify (assoc b :opaque-extra (Object.))))))
    (is (= :semantic-fingerprint-unsupported
           (reason #(recertify-evidence (assoc-in b [:derivation :bundle :source-namespace] (Object.))))))
    (is (fingerprint/equivalent? (:primal b) (get-in b [:derivation :bundle :map-programs :primal])))))

(deftest recertified-nested-evidence-still-must-satisfy-the-existing-authority
  (let [b (canonical/adapt-map (simple :double {}))
        residual-id (first (vals (get-in b [:derivation :bundle :map-programs :interface :residual-arrays])))]
    (is (= :ad-normalized-map-active-array
           (reason #(canonical/validate!
                     (recertify-evidence (assoc-in b [:derivation :bundle :map-programs :interface :active-input] 'other))))))
    (is (= :ad-normalized-map-programs
           (reason #(canonical/validate!
                     (recertify-evidence
                      (update-in b [:derivation :bundle :map-programs :primal]
                                 (fn [program]
                                   (apply list (assoc (vec program) 1
                                                      (assoc-in (soac/facts program)
                                                                [:values residual-id :ownership] :borrowed))))))))))))

(deftest existing-rule-staleness-refuses-before-canonical-execution
  (let [b (canonical/adapt-map (simple :double {}))
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions) current (revisions)
        calls (atom 0) original route/materialize-host-map]
    (with-redefs-fn {revisions (constantly (update current :template-revision inc))
                    #'route/materialize-host-map (fn [p] (swap! calls inc) (original p))}
      (fn []
        (is (= :ad-normalized-stale-rules (reason #(execute b :primal {'x (double-array 1) 'n 1}))))
        (is (zero? @calls))))))

(deftest canonical-pilot-does-not-reinterpret-scalar-only-bundles
  (let [source (#'element/map-program :double)
        scalar-only (normalized/linearize-map-element source
                                                     (the-ns 'raster.ad.normalized-program-bundle-test))]
    (is (= :ad-program-bundle-scope (reason #(canonical/adapt-map scalar-only))))))
