(ns raster.gpu.compiled-field-intent-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.compiled :as compiled]))

(def projection
  {:physical-parameters
   [{:symbol 'state_positions :binding 'state :field :positions :tag 'floats}
    {:symbol 'state_labels :binding 'state :field :labels :tag 'ints}
    {:symbol 'state_width :binding 'state :field :width :tag 'long}
    {:symbol 'other_positions :binding 'other :field :positions :tag 'floats}
    {:symbol 'weights_0 :binding 'weights :path [:blocks 0 :values] :tag 'floats}]})

(def buffers {'state_positions :p 'state_labels :l 'other_positions :o 'weights_0 :w})

(defn resolve-intent [constants roles]
  (#'compiled/resolve-equation-first-field-intent projection buffers constants roles))

(deftest selectors-use-complete-declared-public-paths
  (is (= {:constants ['state_labels 'state_labels]
          :roles {'state_positions :input 'other_positions :constant 'weights_0 :input}}
         (resolve-intent [[:state :labels] [:state :labels]]
                         {[:state :positions] :input [:other :positions] :constant
                          [:weights :blocks 0 :values] :input})))
  (is (= {:constants ['state_positions 'state_labels] :roles {'state_positions :input 'state_labels :input}}
         (resolve-intent ['state] {'state :input})))
  (is (= {:constants ['plain] :roles {'plain :input}}
         (resolve-intent ['plain] {'plain :input}))))

(deftest absent-scalar-or-malformed-field-selectors-refuse
  (doseq [selector [[:state :width] [:missing :positions] [:state :absent]
                   [:positions] [] ['state :positions] [:weights :values 0]]]
    (doseq [f [#(resolve-intent [selector] {}) #(resolve-intent [] {selector :input})]]
      (let [error (try (f) (catch clojure.lang.ExceptionInfo e e))]
        (is (= :compiled-field-selector (:reason (ex-data error))))
        (is (= selector (:selector (ex-data error)))))))
  (let [error (try (#'compiled/resolve-equation-first-field-intent
                    (update projection :physical-parameters conj (first (:physical-parameters projection)))
                    buffers [[:state :positions]] {})
                   (catch clojure.lang.ExceptionInfo e e))]
    (is (= :compiled-field-selector (:reason (ex-data error))))))

(deftest overlapping-role-intent-is-order-independent
  (doseq [roles [(array-map 'state :input [:state :positions] :constant)
                (array-map [:state :positions] :constant 'state :input)
                (array-map 'state_positions :input [:state :positions] :constant)]]
    (let [error (try (resolve-intent [] roles) (catch clojure.lang.ExceptionInfo e e))]
      (is (= :compiled-field-role-conflict (:reason (ex-data error))))
      (is (= 'state_positions (:symbol (ex-data error))))))
  (is (= {'state_positions :input 'state_labels :input}
         (:roles (resolve-intent [] {'state :input [:state :positions] :input})))))

(deftest field-output-intent-remains-lazily-unsupported
  (doseq [[option from] [[:donate :donated] [:outputs :output] [:taps :tap]]]
    (let [opts {option [[:state :positions]]}
          request (#'compiled/boundary-request (:donate opts) nil (:outputs opts) (:taps opts) {})
          entry-key (case option :donate :donated-entries :outputs :output-entries :taps :tap-entries)
          error (try (vec (get request entry-key)) (catch clojure.lang.ExceptionInfo e e))]
      (is (= :compiled-field-selector-intent (:reason (ex-data error))))
      (is (= from (:intent (ex-data error)))))))
