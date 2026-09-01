(ns raster.compiler.passes.scalar.effects-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.scalar.effects :as effects]))

(deftest beichte-analyzes-the-semantic-operation-behind-invk
  (let [pure-call
        (with-meta
          '(.invk raster.numeric/_star__m_long_long-impl rows width)
          {:raster.op/original 'raster.numeric/*
           :raster.type/tag 'long :tag 'long})
        unknown-call '(.invk user/unknown-impl rows width)]
    (is (= {:effect :pure :flags #{}} (effects/descriptor pure-call)))
    (is (effects/removable-expr? pure-call))
    (is (not (effects/removable-expr? unknown-call))
        "an invk without certified semantic metadata remains conservative")))
