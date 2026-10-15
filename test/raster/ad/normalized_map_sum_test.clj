(ns raster.ad.normalized-map-sum-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.core :as core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm weighted-double-sum [xs :- (Array double) weights :- (Array double) size :- Long]
  :- Double
  (let [mapped (par/map [i size] (n/* (n/* (ra/aget xs i) (ra/aget xs i)) (ra/aget weights i)))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))
(deftm weighted-float-sum [xs :- (Array float) weights :- (Array float) size :- Long]
  :- Float
  (let [mapped (par/map [i size] (n/* (n/* (ra/aget xs i) (ra/aget xs i)) (ra/aget weights i)))]
    (par/reduce total (float 0.0) i size (n/+ total (ra/aget mapped i)))))

(defn- surface-source [v dt]
  (let [result (gensym "sum_result_")
        body (first (core/ensure-walked-body! v))
        options {:dtype dt :array-types {'xs dt 'weights dt} :scalar-types {'size :long}}]
    (frontend/form->program
     (frontend/normalize-source
      (inline/flatten-nested-lets (list 'let* [result body] result)) options) options)))

(defn- source [dt]
  (let [zero (if (= dt :float) (float 0.0) 0.0)
        tensor (av/tensor {:dtype dt :shape '[size]})
        scalar (av/tensor {:dtype dt :shape []})
        step '(raster.numeric/+ total element)
        reduction (list '= 'sum-1 '[loss]
                        (list 'reduce {:index 'i :extent 'size :accumulators '[total]
                                       :identities [zero] :dtypes [dt]
                                       :algebra [(scan/certify-reassociation
                                                  {:acc 'total :init zero :lambda step} dt)]}
                              '[mapped] [] (soac/lambda-form '[total element] [step])))]
    ;; Use ordered typed locals, as actual frontend projection does; nested expression source
    ;; is deliberately not a second scalar-parser requirement of this sum slice.
    (let [map-equation (list '= 'map-0 '[mapped]
                             (list 'map {:index 'i :extent 'size} '[xs weights] []
                                   (soac/lambda-form '[x w]
                                                     [(soac/local-value 'square dt '(raster.numeric/* x x))]
                                                     '[(raster.numeric/* square w)])))]
      (soac/make
       (soac/default-program-facts
        {:values {'xs tensor 'weights tensor 'mapped tensor 'loss scalar
                  'size (av/tensor {:dtype :long :shape []})}
         :inputs '[xs weights size]
         :equations {'map-0 (soac/default-equation-facts)
                     'sum-1 (soac/default-equation-facts)}})
       [map-equation reduction] '[loss]))))

(defn- build [source active]
  (normalized/linearize-map-sum-program source (the-ns 'raster.ad.normalized-map-sum-test)
                                        {:active-array active}))
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- recertify [bundle]
  (assoc bundle :certificate (#'normalized/certificate bundle)))
(defn- change-source [source f]
  (soac/make (soac/facts source) (f (soac/equations source)) (soac/outputs source)))
(defn- change-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

(deftest map-sum-explicitly-refuses-multiple-active-inputs
  (let [program (source :double)
        source-ns (the-ns 'raster.ad.normalized-map-sum-test)]
    (is (= :ad-normalized-sum-boundary
           (reason #(normalized/linearize-map-sum-program
                     program source-ns {:active-inputs '[weights xs]}))))
    (let [map-source (#'normalized/sum-source-map program source-ns)
          multi-map (#'normalized/build-map-programs
                     map-source source-ns {:active-inputs '[weights xs]})]
      (is (= :ad-normalized-sum-interface
             (reason #(#'normalized/lift-sum-programs multi-map program {})))))))

(deftest checked-sum-retains-original-ordered-step-and-explicit-scalar-seed
  (doseq [dt [:float :double] active '[xs weights]]
    (let [source (source dt)
          bundle (build source active)
          sums (:sum-programs bundle)
          original-reduction (second (soac/equations source))
          tangent-reduction (second (soac/equations (:linear sums)))
          {:keys [arrays captures lambda]} (soac/operation-parts
                                           (first (soac/equations (:transpose sums))))
          seed (get-in sums [:interface :seed])]
      (is (identical? source (:source sums)))
      (is (identical? bundle (normalized/validate! bundle)))
      (is (fingerprint/equivalent? original-reduction
                                   (second (soac/equations (:primal sums)))))
      (is (fingerprint/equivalent? (soac/operation-parts original-reduction)
                                   (assoc (soac/operation-parts tangent-reduction)
                                          :arrays '[mapped])))
      (is (= dt (:dtype (get-in (soac/facts (:transpose sums)) [:values seed]))))
      (is (= [] (:shape (get-in (soac/facts (:transpose sums)) [:values seed]))))
      (is (= seed (last captures)))
      (is (= (:cotangent-seed bundle) (last (:parameters (soac/lambda-parts lambda)))))
      (is (= (set arrays) (set (vals (get-in bundle [:map-programs :interface :residual-arrays])))))
      (is (= #{:inputs :vjp :jvp} (set (keys (runtime/prepare bundle))))))))

(deftest real-surface-frontend-retains-the-exact-two-operation-objective
  (doseq [[v dt array] [[#'weighted-double-sum :double double-array]
                       [#'weighted-float-sum :float float-array]]]
    (let [program (surface-source v dt)
          bundle (build program 'xs)
          xs (array [-2.0 0.0 3.0]) weights (array [0.5 2.0 -1.0])]
      (is (= 2 (count (soac/equations program))))
      (is (identical? program (get-in bundle [:sum-programs :source])))
      (is (= -7.0 (v xs weights 3)))
      (is (zero? (v (array 0) (array 0) 0)))
      (is (= [-2.0 0.0 3.0] (vec xs)))
      (is (= [0.5 2.0 -1.0] (vec weights)))
      (is (fingerprint/equivalent? (second (soac/equations program))
                                   (second (soac/equations (get-in bundle [:sum-programs :primal]))))))))

(deftest sum-checks-actual-step-identity-and-certificate-separately
  (let [source (source :double)
        ;; Tamper raw producer forms so the refusal occurs inside the checked entry, not
        ;; while constructing test data. Shared SOAC certification may reject first.
        change-reduction (fn [f] (apply list (assoc (vec source) 2
                                                  (assoc (vec (soac/equations source)) 1
                                                         (f (second (soac/equations source)))))))
        altered-step (change-reduction
                      #(apply list (assoc (vec %) 3
                                          (let [op (vec (nth % 3))]
                                            (apply list (assoc op 4
                                                               (soac/lambda-form '[total element]
                                                                                 '[(raster.numeric/* total element)])))))))
        altered-zero (change-reduction
                      #(apply list (update (vec %) 3
                                           (fn [op] (apply list (update (vec op) 1 assoc :identities [1.0]))))))
        altered-algebra (change-reduction
                         #(apply list (update (vec %) 3
                                              (fn [op]
                                                (apply list
                                                       (update (vec op) 1 assoc :algebra
                                                               [(assoc (first (:algebra (second op)))
                                                                       :element 'unrelated)]))))))]
    (is (= :reduction-nonidentity-init (reason #(build altered-step 'xs))))
    (is (= :reduction-nonidentity-init (reason #(build altered-zero 'xs))))
    (is (= :ad-normalized-sum-algebra (reason #(build altered-algebra 'xs))))))

(deftest use-site-binder-metadata-cannot-change-the-sum-carrier
  (let [source (source :double)
        equations (vec (soac/equations source))
        reduction (vec (second equations))
        operation (vec (nth reduction 3))
        total (with-meta 'total {:raster.type/tag 'float})
        body (list 'raster.numeric/+ total 'element)
        operation (apply list (assoc operation 4 (soac/lambda-form '[total element] [body])))
        tampered (apply list (assoc (vec source) 2
                                   (assoc equations 1 (apply list (assoc reduction 3 operation)))))]
    (is (= :ad-normalized-sum-step (reason #(build tampered 'xs))))))

(deftest recertification-cannot-break-source-or-snapshot-correspondence
  (let [bundle (build (source :double) 'xs)
        seed (get-in bundle [:sum-programs :interface :seed])
        [residual] (vals (get-in bundle [:map-programs :interface :residual-arrays]))]
    (is (= :ad-normalized-sum-programs
           (reason #(normalized/validate!
                     (recertify (update-in bundle [:sum-programs :transpose]
                                           change-facts (fn [facts] (assoc-in facts [:values seed :ownership] :borrowed))))))))
    (is (some? (reason #(normalized/validate!
                        (recertify (update-in bundle [:sum-programs :primal]
                                              change-facts (fn [facts] (assoc-in facts [:values residual :ownership] nil))))))))
    (is (some? (reason #(normalized/validate!
                        (recertify (update-in bundle [:sum-programs :transpose]
                                              change-facts (fn [facts] (assoc-in facts [:values seed :dtype] :float))))))))
    (is (some? (reason #(normalized/validate!
                        (recertify (update-in bundle [:sum-programs :primal]
                                              change-facts (fn [facts] (assoc-in facts [:values 'loss :dtype] :float))))))))
    (is (= :ad-normalized-sum-interface
           (reason #(normalized/validate!
                     (recertify (assoc-in bundle [:sum-programs :interface :seed] 'xs))))))
    (let [multi (-> bundle
                    (assoc-in [:map-programs :interface :active-inputs] '[xs weights])
                    (update-in [:map-programs :interface :tangent-inputs] conj 'forged-tangent)
                    (update-in [:map-programs :interface :gradient-outputs] conj 'forged-gradient))]
      (is (= :ad-normalized-map-interface
             (reason #(normalized/validate! (recertify multi))))))))

(deftest frontend-bookkeeping-is-checked-before-map-prefix-projection
  (let [source (source :double)
        facts (-> (soac/facts source)
                  (assoc :attributes {:source-dialect :closed-clojure :source-bindings '[mapped loss]
                                      :host-binding-ids [] :host-read-values [] :host-read-sites [] :allocations []})
                  (assoc :equations
                         (into {} (map-indexed
                                   (fn [index equation]
                                     [(second equation)
                                      (assoc (soac/default-equation-facts
                                              {:front-end :analyzed-source :source-binding-id index})
                                             :attributes {:source-prefix? (zero? index)})])
                                   (soac/equations source)))))
        frontend-source (soac/make facts (soac/equations source) (soac/outputs source))
        bundle (build frontend-source 'xs)]
    (is (= '[mapped loss] (get-in (soac/facts (get-in bundle [:sum-programs :source]))
                                                [:attributes :source-bindings])))
    (is (= '[mapped] (get-in (soac/facts (get-in bundle [:map-programs :source]))
                                            [:attributes :source-bindings])))
    (is (= :ad-normalized-sum-boundary
           (reason #(build (soac/make (assoc-in facts [:attributes :host-binding-ids] '[effect])
                                     (soac/equations source) (soac/outputs source)) 'xs))))
    (is (= :typed-soac-equation-projection
           (reason #(soac/equation-subprogram source
                                              (list '= 'other '[mapped]
                                                    (nth (first (soac/equations source)) 3))))))))

(deftest identity-map-needs-no-residual-array-or-broadcast-seed
  (let [original (source :double)
        source (change-source original
                              #(assoc (vec %) 0
                                      (list '= 'map-0 '[mapped]
                                            (list 'map {:index 'i :extent 'size} '[xs weights] []
                                                  (soac/lambda-form '[x w] '[x])))))
        bundle (build source 'xs)
        transpose (:transpose (:sum-programs bundle))
        {:keys [arrays captures]} (soac/operation-parts (first (soac/equations transpose)))]
    (is (empty? (:residuals bundle)))
    (is (= [] arrays))
    (is (= [(get-in bundle [:sum-programs :interface :seed])] captures))
    (is (identical? transpose (soac/validate! transpose)))))

(deftest reversed-addition-is-preserved-and-negative-zero-is-not-relabelled
  (doseq [dt [:float :double]]
    (let [original (source dt)
          zero (if (= dt :float) (float 0.0) 0.0)
          step '(raster.numeric/+ element total)
          reversed (change-source
                    original
                    (fn [equations]
                      (let [equation (vec (second equations))
                            operation (vec (nth equation 3))
                            attributes (assoc (second operation) :algebra
                                              [(scan/certify-reassociation
                                                {:acc 'total :init zero :lambda step} dt)])
                            operation (apply list (assoc operation 1 attributes
                                                        4 (soac/lambda-form '[total element] [step])))]
                        (assoc (vec equations) 1 (apply list (assoc equation 3 operation))))))
          bundle (build reversed 'xs)
          negative-zero (if (= dt :float) (float -0.0) -0.0)
          wrong-zero (change-source
                      original
                      (fn [equations]
                        (let [equation (vec (second equations))
                              operation (vec (nth equation 3))
                              operation (apply list
                                               (update operation 1 assoc :identities [negative-zero]))]
                          (assoc (vec equations) 1 (apply list (assoc equation 3 operation))))))]
      (is (fingerprint/equivalent? (second (soac/equations reversed))
                                   (second (soac/equations (get-in bundle [:sum-programs :primal])))))
      (is (= :ad-normalized-sum-step (reason #(build wrong-zero 'xs)))))))

(deftest internal-envelope-does-not-repeat-the-public-membership-check
  (let [program (source :double)
        envelope (with-redefs [soac/equation-subprogram
                               (fn [& _] (throw (ex-info "unexpected public membership scan" {})))]
                   (route/program-envelope program))]
    (is (= 2 (count (:equations envelope))))))

(deftest batch-projection-validates-the-whole-program-once
  (let [program (source :double)
        original soac/validate!
        calls (atom 0)
        projected (with-redefs [soac/validate! (fn [candidate]
                                               (when (identical? candidate program) (swap! calls inc))
                                               (original candidate))]
                    (soac/equation-subprograms program))]
    (is (= 1 @calls))
    (is (= 2 (count projected)))
    (is (every? true? (map #(fingerprint/equivalent? %1 (first (soac/equations %2)))
                           (soac/equations program) projected)))))

(deftest scalar-loss-preserves-map-result-physical-facts
  (let [program (source :double)
        bundle (build program 'xs)]
    (doseq [[key value] [[:memory-space :device]
                       [:placement {:device :gpu}]
                       [:sharding {:axis 0}]]]
      ;; A loss-only physical change is not permission to erase or force host placement.
      (let [changed (change-facts program #(assoc-in % [:values 'loss key] value))]
        (is (identical? changed (soac/validate! changed)))
        (is (= :ad-normalized-sum-boundary (reason #(build changed 'xs))))
        (is (= :ad-normalized-sum-boundary
               (reason #(normalized/validate!
                         (recertify (assoc-in bundle [:sum-programs :source] changed)))))))
      (let [matching (change-facts program
                                   #(reduce (fn [facts id] (assoc-in facts [:values id key] value))
                                            % '[xs weights mapped loss]))
            checked (build matching 'xs)]
        (is (= value (get-in (soac/facts (get-in checked [:sum-programs :primal]))
                             [:values 'loss key])))
        (is (= value (get-in (soac/facts (get-in checked [:sum-programs :transpose]))
                             [:values (get-in checked [:sum-programs :interface :seed]) key])))))))
