(ns raster.gpu.benchmark-lifetime-test
  "Fault-test the checked-in benchmark helper without adding all benchmarks to the test classpath."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.ze-runtime :as ze]))

(load-file "bench/native_lifetime.clj")
(alias 'lifetime 'native-lifetime)

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(deftest failed-binding-releases-the-successful-prefix-and-current-kernel
  (let [calls (atom []) next-id (atom 0) primary (ex-info "grid setup failed" {})]
    (with-redefs [ze/create-kernel-fresh
                  (fn [& _]
                    (let [id (swap! next-id inc)]
                      {:handle id ::cleanup/owner
                       (cleanup/owner [{:id :kernel :release #(swap! calls conj id)}])}))
                  ze/destroy-kernel! #(cleanup/release! (::cleanup/owner %))]
      (is (identical? primary
            (error-of #(lifetime/record!
                         (fn []
                           (mapv (fn [i]
                                   (lifetime/bind! :module "k"
                                     (fn [handle]
                                       (when (= i 1) (throw primary))
                                       {:kernel handle})))
                                 (range 3))) {}))))
      (is (= [2 1] @calls))
      (is (= 2 @next-id)))))

(deftest failed-recording-with-known-rollback-releases-kernels
  (let [calls (atom []) primary (ex-info "recording rejected before native contact" {})
        owner (cleanup/owner [{:id :kernel :release #(swap! calls conj :kernel)}])]
    (with-redefs [ze/create-kernel-fresh (fn [& _] {:handle :kernel ::cleanup/owner owner})
                  ze/destroy-kernel! #(cleanup/release! (::cleanup/owner %))
                  ze/record-graph! (fn [& _] (throw primary))]
      (is (identical? primary
            (error-of #(lifetime/record!
                         (fn [] [(lifetime/bind! :module "k" (fn [_] {}))]) {}))))
      (is (= [:kernel] @calls)))))

(deftest unknown-recording-rollback-retains-and-blocks-dependent-kernels
  (let [calls (atom []) primary (ex-info "recording acquisition failed" {})
        destroy-fault (ex-info "list destruction unknown" {})
        kernel-owner (cleanup/owner [{:id :kernel :release #(swap! calls conj :kernel)}])
        graph-owner (cleanup/owner [{:id :list :release #(do (swap! calls conj :graph)
                                                           (throw destroy-fault))}])]
    (with-redefs [ze/create-kernel-fresh (fn [& _] {:handle :kernel ::cleanup/owner kernel-owner})
                  ze/destroy-kernel! #(cleanup/release! (::cleanup/owner %))
                  ze/record-graph! (fn [_ {:keys [adopt-cleanup!]}]
                                    (adopt-cleanup! graph-owner)
                                    (throw primary))]
      (let [error (error-of #(lifetime/record!
                              (fn [] [(lifetime/bind! :module "k" (fn [_] {}))]) {}))
            retained (::cleanup/unresolved (ex-data error))]
        (is (identical? primary (.getCause ^Throwable error)))
        (is (some? retained))
        (is (= [:graph] @calls))
        (is (= :live (:phase (cleanup/status kernel-owner))))
        (is (identical? destroy-fault (error-of #(cleanup/release! retained))))
        (is (= [:graph] @calls))))))

(deftest successful-recording-releases-graph-before-kernels
  (let [calls (atom [])
        kernel-owner (cleanup/owner [{:id :kernel :release #(swap! calls conj :kernel)}])
        graph-owner (cleanup/owner [{:id :list :release #(swap! calls conj :graph)}])]
    (with-redefs [ze/create-kernel-fresh (fn [& _] {:handle :kernel ::cleanup/owner kernel-owner})
                  ze/destroy-kernel! #(cleanup/release! (::cleanup/owner %))
                  ze/record-graph! (fn [& _] {::cleanup/owner graph-owner})]
      (let [graph (lifetime/record! (fn [] [(lifetime/bind! :module "k" (fn [_] {}))]) {})]
        (cleanup/release! (::cleanup/owner graph))
        (cleanup/release! (::cleanup/owner graph))
        (is (= [:graph :kernel] @calls))))))
