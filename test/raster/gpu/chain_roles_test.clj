(ns raster.gpu.chain-roles-test
  "Silently-ignored-information family: a chain buffer's residency role is validated at
   bind time. An unknown role (a typo like :ouput) used to be stored verbatim and then
   never matched by the (= r :output) download filter — so the buffer was silently never
   downloaded. chain-roles-of now rejects any role outside the modeled set by name.

   Pure test — exercises the role helper only, no device/session."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.core :as gc]))

(deftest chain-roles-validated
  (testing "known roles pass through; a missing role defaults to :scratch"
    (is (= {:w :constant :x :input :y :scratch}
           (#'gc/chain-roles-of {:w [:float 10 nil :constant]
                                 :x [:float 10 nil :input]
                                 :y [:float 10 nil nil]}))))
  (testing "an unknown/typo'd role is REJECTED by name, not silently stored (never downloaded)"
    (is (thrown-with-msg?
         Exception #"unknown role :ouput"
         (#'gc/chain-roles-of {:z [:float 10 nil :ouput]})))))
