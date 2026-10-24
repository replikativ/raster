(ns raster.compiler.support.mangled-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.types :as types]
            [raster.compiler.support.mangled :as mangled]))

(deftest operator-codec-roundtrips-all-one-and-two-character-names
  (is (identical? types/op-name-map mangled/op-name-map))
  (let [characters (sort (keys mangled/op-name-map))
        operators (concat (map str characters)
                          (for [left characters right characters] (str left right)))]
    (doseq [operator operators]
      (let [encoded (mangled/encode-operator-name operator)
            ;; A one-argument `symbol` would parse operator pairs containing
            ;; slash as namespace/name, instead of passing this full base name.
            implementation (symbol "original.source" (str (types/mangle operator '[long long]) "-impl"))]
        (is (= operator (mangled/decode-operator-name encoded)))
        (is (= (symbol "original.source" operator) (mangled/impl->op implementation))))))
  (doseq [[operator encoded] [["==" "_eqeq_"] ["<=" "_lteq_"] [">=" "_gteq_"]
                              ["&&" "_ampamp_"] ["?" "_pred_"] ["!" "_bang_"]]]
    (is (= encoded (mangled/encode-operator-name operator)))
    (is (= operator (mangled/decode-operator-name encoded)))))

(deftest ordinary-helper-names-and-source-namespaces-stay-intact
  (doseq [helper '[ordinary-helper with-hyphen! ready? plus eqeq amp tilde
                  ns_like_name ordinary_pred_name]]
    (is (= (name helper) (mangled/encode-operator-name helper)))
    (is (nil? (mangled/decode-operator-name (name helper))))
    (is (= (symbol "actual.source" (name helper))
           (mangled/impl->op (symbol "actual.source" (str (types/mangle helper '[double]) "-impl"))))))
  (doseq [name ["_unknown_" "_plusunknown_" "__" "_amp_tail" "prefix_amp_"]]
    (is (nil? (mangled/decode-operator-name name))))
  (is (nil? (mangled/impl->op 'unqualified_m_double-impl)))
  (is (nil? (mangled/impl->op 'actual.source/ordinary-helper)))
  (is (= 'original.alias/== (mangled/impl->op 'original.alias/_eqeq__m_long_long-impl))))

(deftest previously-supported-legacy-helper-spellings-remain-compatible
  (doseq [[implementation operation]
          '[[legacy.source/changed_bang_m_long-impl legacy.source/changed!]
            [legacy.source/ready_qmark_m_long-impl legacy.source/ready?]
            [legacy.source/prefix_plus_suffix_m_double-impl legacy.source/prefix+suffix]
            [legacy.source/_lteq__m_long_long-impl legacy.source/<=]
            [legacy.source/_gteq__m_long_long-impl legacy.source/>=]]]
    (is (= operation (mangled/impl->op implementation)))))
