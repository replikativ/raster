(ns raster.compiler.ir.storage-boundary-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.link-plan :as link]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as runtime]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- request []
  {:id :completed-storage :target :ze:0
   :nodes [(link/node {:id :out :dtype :float :shape [4] :device :ze:0
                      :role :output})]
   :outputs [:out]})

(deftest layout-is-not-an-executable-or-initialization-proof
  (let [boundary (link/make-storage-boundary (request))]
    (is (= [:out] (:outputs boundary)))
    (is (= [4] (get-in boundary [:nodes :out :view :shape])))
    (is (not (link/link-plan? boundary)))
    (is (nil? (:instances boundary)))
    (is (nil? (:initialization boundary)))
    (is (= :link-plan-type (reason #(link/validate! boundary))))
    (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "driver contact")))]
      (is (= :link-plan-type (reason #(runtime/instantiate! boundary)))))))

(deftest storage-boundary-cannot-carry-executable-evidence
  (doseq [[key value] [[:instances []] [:effect-evidence {}] [:initialization #{:out}]]]
    (is (= :link-storage-boundary-contract
           (reason #(link/make-storage-boundary (assoc (request) key value)))))))

(deftest output-layout-and-allocation-rules-are-shared
  (is (= :link-outputs
         (reason #(link/make-storage-boundary (assoc (request) :outputs [:missing])))))
  (is (= :link-output-duplicates
         (reason #(link/make-storage-boundary (assoc (request) :outputs [:out :out])))))
  (let [nodes (mapv (fn [id]
                      (link/node {:id id :dtype :float :shape [4] :device :ze:0
                                  :role :output :allocation-id :shared})) [:a :b])
        request (assoc (request) :nodes nodes :outputs [:a :b])]
    (is (= :link-undeclared-alias (reason #(link/make-storage-boundary request))))
    (is (= #{#{:a :b}}
           (:aliases (link/make-storage-boundary (assoc request :aliases #{#{:a :b}})))))))
