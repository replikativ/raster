(ns raster.compiler.auto-buffer-cache-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]))

(defn- with-operation [f]
  (let [namespace (create-ns (gensym "raster.auto-buffer-cache-test."))
        operation (symbol (str (ns-name namespace)) "allocate")]
    (try (f namespace operation)
         (finally (remove-ns (ns-name namespace))))))

(defn- publish! [namespace body]
  ;; A retained walked body is the detector's input; no helper execution or
  ;; dispatch registration is needed to exercise this public compiler boundary.
  (let [v (intern namespace 'allocate (fn [_] :not-executed))]
    (alter-meta! v assoc :raster.core/deftm true
                 :raster.core/deftm-params '[n]
                 :raster.core/deftm-walked-body [body])
    v))

(defn- allocation [operation]
  (when-let [[facet _] (descriptor/resolve-buffer-semantics operation)]
    ((:alloc-form facet) '[width] {})))

(deftest negative-lookup-does-not-hide-later-publication
  (with-operation
    (fn [namespace operation]
      (is (nil? (descriptor/resolve-buffer-semantics operation)))
      (publish! namespace '(clojure.core/double-array n))
      (is (= '(clojure.core/double-array width) (allocation operation))))))

(deftest retained-body-replacement-invalidates-positive-and-negative-results
  (with-operation
    (fn [namespace operation]
      (let [v (publish! namespace '(clojure.core/double-array n))]
        (is (= '(clojure.core/double-array width) (allocation operation)))
        (alter-meta! v assoc :raster.core/deftm-walked-body '[n])
        (is (nil? (descriptor/resolve-buffer-semantics operation)))
        (alter-meta! v assoc :raster.core/deftm-walked-body
                     '[(clojure.core/float-array n)])
        (is (= '(clojure.core/float-array width) (allocation operation)))))))

(deftest var-and-root-identity-are-cache-dependencies
  (with-operation
    (fn [namespace operation]
      (let [v (publish! namespace '(clojure.core/double-array n))
            first-result (descriptor/resolve-buffer-semantics operation)]
        (is (identical? first-result (descriptor/resolve-buffer-semantics operation)))
        (alter-var-root v (constantly (fn [_] :new-root)))
        (let [next-result (descriptor/resolve-buffer-semantics operation)]
          (is (not (identical? first-result next-result)))
          (is (= '(clojure.core/double-array width) (allocation operation))))
        (ns-unmap namespace 'allocate)
        (let [replacement (publish! namespace '(clojure.core/float-array n))]
          (is (not (identical? v replacement)))
          (is (= '(clojure.core/float-array width) (allocation operation))))))))

(deftest metadata-only-publication-invalidates-cache
  (with-operation
    (fn [namespace operation]
      (let [v (publish! namespace '(clojure.core/double-array n))
            first-result (descriptor/resolve-buffer-semantics operation)]
        ;; Even equal forms may carry different analysis-relevant metadata.
        (alter-meta! v assoc :raster.core/deftm-walked-body
                     [(with-meta '(clojure.core/double-array n) {:tag 'doubles})])
        (is (not (identical? first-result
                             (descriptor/resolve-buffer-semantics operation))))
        (is (= '(clojure.core/double-array width) (allocation operation)))))))

(deftest analysis-publication-is-not-cached-under-a-newer-snapshot
  (with-operation
    (fn [namespace operation]
      (let [v (publish! namespace '(clojure.core/double-array n))
            detector (ns-resolve 'raster.compiler.core.op-descriptor
                                 'detect-auto-buffer-semantics)
            original @detector
            calls (atom 0)]
        (with-redefs-fn
          {detector (fn [value]
                      (swap! calls inc)
                      (let [result (original value)]
                        (when (= 1 @calls)
                          (alter-meta! v assoc :raster.core/deftm-walked-body
                                       '[(clojure.core/float-array n)]))
                        result))}
          #(do
             (is (= '(clojure.core/double-array width) (allocation operation)))
             (is (= '(clojure.core/float-array width) (allocation operation)))
             (is (= '(clojure.core/float-array width) (allocation operation)))
             (is (= 2 @calls))))))))

(deftest manual-buffer-contract-still-takes-precedence
  (testing "preexisting manual authority is not replaced by auto-detection"
    (with-operation
      (fn [namespace operation]
        (publish! namespace '(clojure.core/double-array n))
        (let [manual {:allocates? false}
              registry @(ns-resolve 'raster.compiler.core.op-descriptor
                                    'descriptor-registry)]
          (try
            (descriptor/register-op-descriptor! operation {:buffer manual})
            (let [[facet base] (descriptor/resolve-buffer-semantics operation)]
              (is (= manual facet))
              (is (= operation base)))
            (finally (swap! registry dissoc operation))))))))
