(ns raster.compiler.passes.parallel.guarded-store-proof-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(def locals
  '[{:id q :dtype :long :init (clojure.core/quot idx 4)}
    {:id r :dtype :long :init (clojure.core/rem idx 4)}])

(defn store [predicate]
  {:out 'out :index 'q :predicate predicate})

(defn proven [stores]
  (#'frontend/proven-unique-stores stores 'idx 20 locals []))

(deftest fixed-guard-restores-only-proven-cross-item-independence
  (doseq [guard ['(clojure.core/== r 0)
                '(clojure.core/== 0 r)
                '(clojure.core/== (clojure.core/long r) (clojure.core/long 0))]]
    (is (= #{0} (proven [(store guard)]))))
  (doseq [guard [true '(clojure.core/< r 2) '(clojure.core/== q 0)
                '(clojure.core/== (clojure.core/unchecked-int r) 0)
                '(clojure.core/== (clojure.core/int r) 0)
                '(clojure.core/== r (aget values idx))]]
    (is (empty? (proven [(store guard)]))))
  (binding [util/*shadowing-locals* #{'==}]
    (is (empty? (proven [(store '(== r 0))])))))

(deftest separately-injective-guards-cannot-certify-their-overlapping-union
  (is (= #{0 1} (proven [(store '(clojure.core/== r 0))
                         (store '(clojure.core/== r 0))])))
  (is (empty? (proven [(store '(clojure.core/== r 0))
                       (store '(clojure.core/== r 1))]))))
