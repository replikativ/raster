(ns raster.compiler.ir.retained-equations-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.emitted-parallel-program :as emitted]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- equation [id literal]
  {:id id :algorithm {:literal literal} :operations [:target-operation]})

(defn- project-alternatives [alternatives suffix]
  ;; Isolate obligations added by the projector, after ordinary program validation.
  ;; Actual validated fused, ordinary, SWR and structured programs are tested separately.
  (with-redefs-fn
    {#'emitted/validate! identity
     (ns-resolve 'raster.compiler.ir.emitted-parallel-program 'equation-candidates)
     (fn [_ caller-options]
       (is (nil? caller-options) "default projection passes no independent math override")
       (mapv #(hash-map :body {:equations %}) alternatives))}
    #(emitted/retained-numerical-equations
      {:equations (into [{:id :compound :operations [:dispatch]
                          :attributes {:emitted-source-equations [:a]}}] suffix)})))

(deftest compound-projection-uses-canonical-floating-literals
  (let [a (equation :a (Double/parseDouble "NaN"))
        b (equation :a (Double/parseDouble "NaN"))]
    (is (= [:a] (mapv :id (project-alternatives [[a] [b]] [])))))
  (is (= :emitted-program-semantic-projection
         (reason #(project-alternatives [[(equation :a 0.0)] [(equation :a -0.0)]] []))))
  (is (= :emitted-program-semantic-projection
         (reason #(project-alternatives [[(equation :a 1.0)] [(equation :a 2.0)]] [])))))

(deftest compound-projection-rejects-expanded-identity-collisions
  (is (= :emitted-program-semantic-identities
         (reason #(project-alternatives [[(equation :a 1.0)]] [(equation :a 1.0)]))))
  (is (= :emitted-program-semantic-projection
         (reason #(project-alternatives [[(equation :b 1.0)]] [])))))
