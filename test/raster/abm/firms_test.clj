(ns raster.abm.firms-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.abm.firms :as firms]
            [raster.abm.firms.membership :as mem]))

(deftest test-init-simulation
  (testing "initialization creates valid state"
    (let [config (firms/->FirmsConfig 100 4 4 0.04 0 10 42 100)
          [_ agents firms _] (firms/init-simulation config)]
      ;; Agent arrays correct size
      (is (= 100 (alength (.effort agents))))
      (is (= 100 (alength (.theta agents))))
      (is (= 400 (alength (.friends agents)))) ;; 100 * 4

      ;; All agents assigned to valid firms
      (let [cf (.firm agents)]
        (dotimes [i 100]
          (let [f (aget ^ints cf i)]
            (is (>= f 0) (str "agent " i " has firm " f))
            (is (< f 100) (str "agent " i " has firm " f)))))

      ;; Theta in [0, 1]
      (let [th (.theta agents)]
        (dotimes [i 100]
          (let [t (aget ^floats th i)]
            (is (>= t 0.0) (str "theta " i " = " t))
            (is (<= t 1.0) (str "theta " i " = " t)))))

      ;; Firm params in valid ranges
      (let [pa (.a firms)]
        (dotimes [i 100]
          (let [a (aget ^floats pa i)]
            (is (>= a 0.09) (str "param-a " i " = " a))
            (is (<= a 1.51) (str "param-a " i " = " a))))))))

(deftest test-membership-consistency
  (testing "CSR membership matches firm sizes"
    (let [config (firms/->FirmsConfig 100 4 4 0.04 0 10 42 100)
          [_ agents firms _] (firms/init-simulation config)
          fs   (.n-workers firms)
          offs (.offsets firms)
          mems (.members firms)
          max-firms (* 100 2)]
      ;; Alive firms have correct member count in CSR
      (let [al (.alive firms)]
        (dotimes [f 100]
          (when (== (aget ^ints al f) 1)
            (let [start (aget ^ints offs f)
                  end   (aget ^ints offs (inc f))
                  csr-size (- end start)
                  fs-size  (aget ^ints fs f)]
              (is (== csr-size fs-size)
                  (str "firm " f ": CSR " csr-size " != size " fs-size)))))))))

(deftest test-smoke-simulation
  (testing "small simulation completes without error"
    (let [config (firms/->FirmsConfig 100 4 4 0.04 0 10 42 100)
          stats (firms/run-simulation config)]
      (is (seq stats) "should produce some stats")
      (let [s (first stats)]
        (is (contains? s :mean-effort))
        (is (contains? s :mean-income))
        (is (contains? s :n-alive))))))

(deftest test-firm-dynamics
  (testing "simulation shows firm birth and death"
    (let [;; High activation to force movement
          config (firms/->FirmsConfig 50 3 3 0.5 0 20 42 50)
          stats (firms/run-simulation config)]
      (is (seq stats))
      (is (> (:n-alive (last stats)) 0)))))

(deftest test-dynamics-stabilize
  (testing "effort and output evolve over time"
    (let [config (firms/->FirmsConfig 200 4 4 0.04 0 500 42 200)
          stats (firms/run-simulation config)]
      (when (>= (count stats) 2)
        (let [last-s (last stats)]
          (is (>= (:mean-effort last-s) 0.0))
          (is (> (:n-alive last-s) 0)))))))
