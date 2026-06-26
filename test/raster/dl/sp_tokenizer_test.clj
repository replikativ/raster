(ns raster.dl.sp-tokenizer-test
  "Validates the SentencePiece-style BPE tokenizer against transformers-generated
  gold token ids for gemma-3-270m. Guarded on the tokenizer.json being present
  (set GEMMA_TOKENIZER or place the model at the default path), so it skips cleanly
  in CI without the model."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.sp-tokenizer :as sp]))

(def ^:private tok-path
  (or (System/getenv "GEMMA_TOKENIZER")
      "/home/christian-weilbach/Development/models/gemma-3-270m-it/tokenizer.json"))

(def ^:private present? (.exists (java.io.File. tok-path)))

;; gold from `transformers.AutoTokenizer.encode` (gemma-3-270m), with <bos>=2.
(def ^:private gold
  {"" [2]
   "a" [2 236746]
   "Hello, world!" [2 9259 236764 1902 236888]
   "The quick brown fox." [2 818 3823 8864 37423 236761]
   "café résumé" [2 123125 236859 118515]
   "日本語" [2 94951]
   "multiple   spaces" [2 43819 139 35220]
   "  leading spaces" [2 138 26016 9952]
   "def foo(x):\n  return x+1" [2 2063 46293 236769 236781 1473 107 138 2060 1123 236862 236770]})

(deftest gemma-encode-matches-transformers
  (if-not present?
    (println "  [skip] gemma tokenizer not at" tok-path "— set GEMMA_TOKENIZER")
    (let [t (sp/load-tokenizer tok-path)]
      (testing "byte-exact encode vs transformers gold (metaspace, byte-fallback, BPE)"
        (doseq [[s ids] gold]
          (is (= ids (sp/encode t s)) (str "encode " (pr-str s))))))))

(deftest gemma-decode-roundtrip
  (if-not present?
    (println "  [skip] gemma tokenizer not present")
    (let [t (sp/load-tokenizer tok-path)]
      (testing "decode reverses encode (▁→space, byte-fallback fused to UTF-8)"
        (doseq [s ["Hello, world!" "café résumé" "日本語" "The quick brown fox."]]
          (is (= s (sp/decode t (sp/encode t s))) (str "roundtrip " (pr-str s))))))))
