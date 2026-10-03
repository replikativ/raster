(ns raster.runtime.display-ownership-test
  (:require [clojure.test :refer [deftest is]]
            [raster.runtime.display :as display]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.ze-runtime :as ze])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(deftest gpu-render-buffer-owns-canonical-child-and-initializes-pixels
  (let [released (atom 0)
        owner (cleanup/owner [{:id :memory :release #(swap! released inc)}])
        storage (int-array [7 7 7 7])
        buffer {:segment (MemorySegment/ofArray storage) ::cleanup/owner owner}]
    (with-redefs [ze/make-buffer
                  (fn [n dtype options]
                    (is (= [4 :int] [n dtype]))
                    ((:retain-owner! options) owner)
                    buffer)]
      (let [render (display/render-buffer-gpu 2 2)]
        (is (identical? buffer (:device-buffer render)))
        (is (= [0 0 0 0] (vec storage)))
        (.set ^MemorySegment (:seg render) ValueLayout/JAVA_INT 0 (int 42))
        (is (zero? (aget ^ints (:pixels render) 0)))
        (is (identical? render (display/sync-from-gpu! render)))
        (is (= 42 (aget ^ints (:pixels render) 0)))
        (display/close-render-buffer! render)
        (display/close-render-buffer! render)
        (is (= 1 @released))
        (is (= :owner-releasing
               (:reason (ex-data (error-of #(display/sync-from-gpu! render))))))))))

(deftest gpu-render-construction-failure-retires-or-retains-the-exact-child
  (doseq [unknown? [false true]]
    (let [primary (ex-info "buffer construction failed" {})
          secondary (ex-info "native release outcome unknown" {})
          attempts (atom 0)
          child (cleanup/owner [{:id :memory
                                 :release #(do (swap! attempts inc)
                                               (when unknown? (throw secondary)))}])]
      (with-redefs [ze/make-buffer
                    (fn [_ _ options]
                      ((:retain-owner! options) child)
                      (throw primary))]
        (let [error (error-of #(display/render-buffer-gpu 2 2))]
          (is (= 1 @attempts))
          (if unknown?
            (let [retained (::cleanup/unresolved (ex-data error))]
              (is (some? retained))
              (is (identical? primary (.getCause ^Throwable error)))
              (is (= [:pixel-allocation] (cleanup/pending retained)))
              (is (identical? secondary (error-of #(cleanup/release! retained))))
              (is (= 1 @attempts) "indeterminate child destruction is not retried"))
            (do (is (identical? primary error))
                (is (empty? (cleanup/pending child))))))))))

(deftest gpu-render-initialization-failure-rolls-back-canonical-allocation
  (let [released (atom 0)
        owner (cleanup/owner [{:id :memory :release #(swap! released inc)}])]
    (with-redefs [ze/make-buffer
                  (fn [_ _ options]
                    ((:retain-owner! options) owner)
                    {:segment (MemorySegment/ofArray (int-array 0)) ::cleanup/owner owner})]
      (is (instance? IndexOutOfBoundsException (error-of #(display/render-buffer-gpu 2 2))))
      (is (= 1 @released))
      (is (empty? (cleanup/pending owner))))))

(deftest gpu-render-retention-cannot-replace-the-first-canonical-owner
  (doseq [first-unknown? [false true]]
    (let [attempts (atom [])
          unknown (ex-info "native retirement unknown" {})
          first-owner (cleanup/owner [{:id :first
                                       :release #(do (swap! attempts conj :first)
                                                     (when first-unknown? (throw unknown)))}])
          second-owner (cleanup/owner [{:id :second
                                        :release #(do (swap! attempts conj :second)
                                                      (throw unknown))}])]
      (with-redefs [ze/make-buffer
                    (fn [_ _ options]
                      ((:retain-owner! options) first-owner)
                      ;; A mismatched constructor must itself preserve its rejected second
                      ;; generation. Use the production transaction's unresolved-error contract.
                      (cleanup/build! second-owner
                                      #(do ((:adopt-cleanup! options) second-owner) {}) nil))]
        (let [error (error-of #(display/render-buffer-gpu 2 2))
              chain (take-while some? (iterate #(.getCause ^Throwable %) error))
              retained (keep #(-> % ex-data ::cleanup/unresolved) chain)]
          (is (some #(= :cleanup-owner-mismatch (:reason (ex-data %))) chain))
          (is (= [:second :first] @attempts) "the first generation was not overwritten")
          (is (some #(identical? second-owner %) retained))
          (if first-unknown?
            (is (some #(= [:pixel-allocation] (cleanup/pending %)) retained)
                "outer authority retains the first generation as well")
            (is (empty? (cleanup/pending first-owner))))
          (doseq [owner retained] (error-of #(cleanup/release! owner)))
          (is (= [:second :first] @attempts) "unknown destruction is never retried"))))))

(deftest gpu-render-dimensions-decline-before-driver-contact
  (let [calls (atom 0)]
    (with-redefs [ze/make-buffer (fn [& _] (swap! calls inc))]
      (is (= :invalid-render-dimensions
             (:reason (ex-data (error-of #(display/render-buffer-gpu -1 2))))))
      (is (instance? ArithmeticException
                     (error-of #(display/render-buffer-gpu Long/MAX_VALUE 2))))
      (is (instance? ArithmeticException
                     (error-of #(display/render-buffer-gpu (inc (long Integer/MAX_VALUE)) 1))))
      (is (instance? ArithmeticException
                     (error-of #(display/render-buffer-gpu Long/MAX_VALUE 0))))
      (is (instance? ArithmeticException
                     (error-of #(display/render-buffer-gpu 0 Long/MAX_VALUE))))
      (is (zero? @calls)))))

(deftest cpu-render-close-preserves-the-zero-copy-array-path
  (let [render (display/render-buffer 2 2)]
    (is (nil? (display/close-render-buffer! render)))
    (.set ^MemorySegment (:seg render) ValueLayout/JAVA_INT 0 (int 9))
    (is (= 9 (aget ^ints (:pixels render) 0)))
    (is (identical? render (display/sync-from-gpu! render)))))
