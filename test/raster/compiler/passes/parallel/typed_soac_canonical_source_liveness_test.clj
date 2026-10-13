(ns raster.compiler.passes.parallel.typed-soac-canonical-source-liveness-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def ^:private scratch
  (with-meta 'scratch {:tag 'floats :raster.type/tag 'floats}))

(def ^:private options
  {:scalar-types {'n :long}
   :values {'n (av/tensor {:dtype :long :shape []})}})

(defn- contract
  [destination]
  (list 'raster.par/contract destination '[[i 2] [j 2]] '[[k 2]]
        '(* (clojure.core/aget a (clojure.core/+ (clojure.core/* i 2) k))
            (clojure.core/aget b (clojure.core/+ (clojure.core/* k 2) j)))))

(defn- source
  [middle body]
  (list 'let*
        (vec (mapcat identity
                     (concat [[scratch '(clojure.core/float-array 4)]
                              ['dead (contract scratch)]]
                             middle
                             [['live '(raster.par/pmap p 4 float
                                                       (clojure.core/aget x p))]])))
        body))

(defn- attempt
  ([form]
   (attempt form {}))
  ([form extra]
   (route/attempt form :float {'a :float 'b :float 'x :float 'C :float}
                  (merge options extra))))

(defn- algorithm-kinds
  [program]
  (mapcat (fn [equation]
            (map dialect/operation-kind (dialect/equations (:algorithm equation))))
          (:equations program)))

(deftest validated-canonical-source-liveness-rebuilds-one-authoritative-program
  (let [{:keys [program stats]} (attempt (source [] 'live))
        algorithm (get-in program [:equations 0 :algorithm])
        facts (dialect/facts algorithm)]
    (is (= 2 (:canonical-source-bindings-removed stats)))
    (is (= '[live] (get-in facts [:attributes :source-bindings])))
    (is (= [0] (mapv second (dialect/equations algorithm))))
    (is (= #{0} (set (keys (:equations facts)))))
    (is (not-any? #{'scratch 'dead} (tree-seq coll? seq (:source program))))))

(deftest validation-precedes-canonical-source-liveness
  (testing "a malformed dead contraction is still rejected"
    (let [bad (list 'raster.par/contract scratch '(i 2) '[[k 2]] 1.0)
          form (list 'let* [scratch '(clojure.core/float-array 4)
                            'dead bad
                            'live '(raster.par/pmap p 4 float
                                                     (clojure.core/aget x p))]
                     'live)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"free-axes must be a vector"
                            (attempt form)))))
  (testing "a dead local output cannot hide contradictory retained value facts"
    (let [form '(let* [unused (raster.par/pmap i 4 float
                                               (clojure.core/aget a i))
                       live (raster.par/pmap p 4 float
                                             (clojure.core/aget x p))]
                      live)]
      (doseq [value [(av/tensor {:dtype :float :shape [5]})
                     (av/tensor {:dtype :double :shape [4]})]]
        (is (= :source-value-conflict
               (get-in (attempt form
                                {:values (assoc (:values options) 'unused value)})
                       [:declined :reason])))))))

(deftest observable-canonical-source-bindings-survive-liveness
  (testing "a write to caller-owned storage remains observable even when its binding is unused"
    (let [form (list 'let* ['dead (contract 'C)
                            'live '(raster.par/pmap p 4 float
                                                     (clojure.core/aget x p))]
                     'live)
          {:keys [program stats]} (attempt form)]
      (is (zero? (:canonical-source-bindings-removed stats)))
      (is (some #{'segmented-reduce} (algorithm-kinds program)))))
  (testing "a returned destination keeps its allocation and writer"
    (let [{:keys [program]} (attempt (source [] scratch))]
      (is (some #{'segmented-reduce} (algorithm-kinds program)))))
  (testing "a downstream read keeps its producer and local storage"
    (let [form (list 'let* [scratch '(clojure.core/float-array 4)
                            'dead (contract scratch)
                            'live '(raster.par/pmap p 4 float
                                                     (clojure.core/aget scratch p))]
                     'live)
          {:keys [program stats]} (attempt form)]
      (is (zero? (:canonical-source-bindings-removed stats)))
      (is (= '#{map segmented-reduce} (set (algorithm-kinds program))))))
  (testing "a source alias of the returned destination does not hide its writer"
    (let [form (list 'let* [scratch '(clojure.core/float-array 4)
                            'alias scratch
                            'dead (contract 'alias)
                            'live '(raster.par/pmap p 4 float
                                                     (clojure.core/aget x p))]
                     scratch)
          {:keys [program]} (attempt form)]
      (is (some #{'segmented-reduce} (algorithm-kinds program)))))
  (testing "an opaque host effect remains ordered even when its value is unused"
    (let [{:keys [program stats]} (attempt (source [['side '(println :observable)]] 'live))]
      (is (= 2 (:canonical-source-bindings-removed stats)))
      (is (some #{'println} (tree-seq coll? seq (:source program))))))
  (testing "a retained checked scalar after device work remains an explicit refusal"
    (let [source-n (with-meta 'n {:tag 'long :raster.type/tag 'long})
          checked (with-meta (list 'clojure.core/int source-n)
                    {:tag 'int :raster.type/tag 'int})
          checked-id (with-meta 'checked {:tag 'int :raster.type/tag 'int})
          result (attempt (source [[checked-id checked]] 'live))]
      (is (= :typed-soac-source-coverage (get-in result [:declined :reason])))
      (is (= {:id 2 :binding 'checked :kind :scalar
              :operation 'clojure.core/int :reason :uncertified-host-scalar}
             (first (get-in result [:declined :bindings])))))))

(deftest eliminating-the-last-admitted-operation-keeps-the-validated-route
  (let [form (list 'let* [scratch '(clojure.core/float-array 4)
                          'dead (contract scratch)]
                   7)
        {:keys [program stats]} (attempt form)]
    (is (zero? (:canonical-source-bindings-removed stats)))
    (is (some #{'segmented-reduce} (algorithm-kinds program)))
    (is (= 7 (last (:source program))))))

(deftest a-reprojection-decline-is-a-compiler-contradiction-not-source-fallback
  (let [project @#'route/source-projection
        calls (atom 0)
        failure
        (try
          (with-redefs-fn
            {#'route/source-projection
             (fn [form frontend-options]
               (if (= 1 (swap! calls inc))
                 (project form frontend-options)
                 (throw (ex-info "injected transformed-source contradiction"
                                 {:reason :source-value-conflict
                                  :witness :reprojection}))))}
            #(attempt (source [] 'live)))
          nil
          (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= 2 @calls))
    (is (= :raster/bug (:reason (ex-data failure))))
    (is (= :canonical-source-liveness-reprojection
           (:invariant (ex-data failure))))
    (is (= {:reason :source-value-conflict :witness :reprojection}
           (:decline (ex-data failure))))
    (is (= :source-value-conflict
           (:reason (ex-data (.getCause failure)))))))
