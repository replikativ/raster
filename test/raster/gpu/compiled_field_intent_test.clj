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

(deftest field-donation-and-tap-intent-remain-lazily-unsupported
  (doseq [[option from] [[:donate :donated] [:taps :tap]]]
    (let [opts {option [[:state :positions]]}
          request (#'compiled/boundary-request (:donate opts) nil (:outputs opts) (:taps opts) {})
          entry-key (case option :donate :donated-entries :outputs :output-entries :taps :tap-entries)
          error (try (vec (get request entry-key)) (catch clojure.lang.ExceptionInfo e e))]
      (is (= :compiled-field-selector-intent (:reason (ex-data error))))
      (is (= from (:intent (ex-data error)))))))

(deftest exact-field-outputs-retain-public-path-keys
  (is (= [{:key [:state :positions] :sym 'state_positions :from :output}
          {:key [:weights :blocks 0 :values] :sym 'weights_0 :from :output}]
         (#'compiled/equation-first-field-output-entries
          projection buffers [[:state :positions] [:weights :blocks 0 :values]])))
  (doseq [[selectors reason]
          [[[[:state :positions] [:state :positions]] :compiled-field-output-duplicates]
           [[[:state :width]] :compiled-field-selector]
           [[[:state :missing]] :compiled-field-selector]]]
    (let [error (try (#'compiled/equation-first-field-output-entries
                      projection buffers selectors)
                     (catch clojure.lang.ExceptionInfo e e))]
      (is (= reason (:reason (ex-data error)))))))

(deftest mutable-field-alias-check-is-cross-public-and-state-scoped
  (let [positions (float-array 2)
        labels (int-array 2)
        other (float-array 2)
        projected {'state_positions positions 'state_labels labels
                   'other_positions other 'weights_0 (float-array 2)}]
    (is (nil? (#'compiled/validate-mutable-aggregate-aliases!
               projection buffers projected
               {'state (take 3 (:physical-parameters projection))}
               {'state_positions :state})))
    (doseq [aliased [(assoc projected 'other_positions positions)
                     (assoc projected 'weights_0 positions)]]
      (let [error (try (#'compiled/validate-mutable-aggregate-aliases!
                        projection buffers aliased
                        {'state (take 3 (:physical-parameters projection))}
                        {'state_positions :state})
                       (catch clojure.lang.ExceptionInfo e e))]
        (is (= :compiled-aggregate-state-alias (:reason (ex-data error))))))
    (is (nil? (#'compiled/validate-mutable-aggregate-aliases!
               projection buffers (assoc projected 'other_positions positions)
               {'state (take 3 (:physical-parameters projection))} {})))))
