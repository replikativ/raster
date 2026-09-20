;; Regenerate test/resources/ggml_oracle from llama.cpp's ggml.
;;
;;   LLAMA_CPP=../llama.cpp-new dev/ggml_oracle/build.sh
;;   clojure -M dev/ggml_oracle/generate.clj
;;
;; Inputs are synthetic and seeded; no model weights are committed. Every
;; output file is produced by ggml_oracle, which calls ggml_quantize_chunk
;; (no importance matrix), the type's to_float, and the CPU backend's
;; from_float/vec_dot.
(ns ggml-oracle.generate
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.pprint :as pprint]
            [clojure.string :as str])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.util Random]))

(def ^:private tool "dev/ggml_oracle/ggml_oracle")
(def ^:private out-dir "test/resources/ggml_oracle")

(defn- write-f32! [path ^floats xs]
  (let [bb (.order (ByteBuffer/allocate (* 4 (alength xs))) ByteOrder/LITTLE_ENDIAN)]
    (doseq [x xs] (.putFloat bb x))
    (with-open [o (io/output-stream path)] (.write o (.array bb)))))

(defn- values [n f] (let [a (float-array n)] (dotimes [i n] (aset a i (float (f i)))) a))

(defn- student-t
  "Student-t with 3 degrees of freedom: a heavy tail that yields block outliers."
  [^Random r]
  (let [z (.nextGaussian r)
        chi (+ (Math/pow (.nextGaussian r) 2) (Math/pow (.nextGaussian r) 2)
               (Math/pow (.nextGaussian r) 2))]
    (/ z (Math/sqrt (/ chi 3.0)))))

(def ^:private cases
  "[id n-per-row nrows generator] — generators get the element index and a seeded Random."
  [[:gaussian 1024 4 (fn [_ ^Random r] (* 0.02 (.nextGaussian r)))]
   [:heavy-tail 1024 4 (fn [_ r] (* 0.02 (student-t r)))]
   [:all-positive 512 2 (fn [_ ^Random r] (.nextDouble r))]
   [:all-negative 512 2 (fn [_ ^Random r] (- (.nextDouble r)))]
   [:constant 256 2 (fn [_ _] 0.375)]
   [:zeros 256 2 (fn [_ _] 0.0)]
   [:tiny 512 2 (fn [_ ^Random r] (* 1.0e-7 (.nextGaussian r)))]
   [:large 512 2 (fn [_ ^Random r] (* 1.0e3 (.nextGaussian r)))]
   [:block-outlier 1024 2 (fn [i ^Random r] (if (zero? (mod i 32)) (* 0.5 (.nextGaussian r))
                                               (* 0.01 (.nextGaussian r))))]
   [:gemma-width 640 3 (fn [_ ^Random r] (* 0.03 (student-t r)))]])

(def ^:private formats ["q8_0" "q5_0" "q4_K" "q6_K"])

(defn- block-size [format] (if (#{"q4_K" "q6_K"} format) 256 32))

(defn- oracle! [& args]
  (let [{:keys [exit out err]} (apply sh/sh tool args)]
    (when-not (zero? exit)
      (throw (ex-info "ggml_oracle failed" {:args args :err err})))
    (str/trim out)))

(defn- case-input [[id n nrows generator] seed]
  (let [r (Random. seed)]
    (values (* n nrows) #(generator % r))))

(defn generate! []
  (.mkdirs (io/file out-dir))
  (let [llama (str/trim (:out (sh/sh "git" "-C" (or (System/getenv "LLAMA_CPP") "../llama.cpp-new")
                                     "rev-parse" "--short" "HEAD")))
        quantized
        (vec
         (for [[[id n nrows :as spec] seed] (map vector cases (range 1 100))]
           (let [input (str (name id) ".f32")
                 _ (write-f32! (str out-dir "/" input) (case-input spec seed))]
             {:id id :n-per-row n :nrows nrows :seed seed :input input
              :formats
              (into (sorted-map)
                    (for [format formats
                          :when (zero? (mod n (block-size format)))]
                      (let [blocks (str (name id) "." format ".blocks")
                            dequant (str (name id) "." format ".dequant.f32")
                            [row-bytes] (str/split (oracle! "quantize" format (str n) (str nrows)
                                                         (str out-dir "/" input)
                                                         (str out-dir "/" blocks)
                                                         (str out-dir "/" dequant))
                                                   #" ")]
                        [(keyword format) {:blocks blocks :dequant dequant
                                           :row-bytes (parse-long row-bytes)}])))})))
        dots
        (vec
         (for [format formats
               [n seed ties?] [[1024 101 false] [256 102 false] [256 103 true]]
               :let [n (if (and (= n 256) (= 32 (block-size format))) 640 n)
                     id (str "dot-" format "-" n (when ties? "-ties"))
                     r (Random. (+ seed (.hashCode ^String format)))
                     w (values n (fn [_] (* 0.02 (student-t r))))
                     ;; Ties: every block holds 127 (so d = 1 and id = 1 exactly) and
                     ;; otherwise exact halves, where roundf and round-to-even differ.
                     x (if ties?
                         (values n (fn [i] (if (zero? (mod i 32)) 127.0
                                             (+ 0.5 (- (mod (* 7 i) 200) 100)))))
                         (values n (fn [_] (* 2.0 (.nextGaussian r)))))
                     wf (str id ".w.f32") xf (str id ".x.f32") act (str id ".act.blocks")
                     act-ref (str id ".act-ref.blocks")]]
           (do (write-f32! (str out-dir "/" wf) w)
               (write-f32! (str out-dir "/" xf) x)
               (let [[act-type bits decimal ref-bits ref-decimal generic-bits generic-decimal]
                     (str/split (oracle! "dot" format (str n)
                                         (str out-dir "/" wf) (str out-dir "/" xf)
                                         (str out-dir "/" act) (str out-dir "/" act-ref))
                                #" ")]
                 {:id id :format (keyword format) :n n :weights wf :activations xf
                  :activation-format (keyword act-type)
                  ;; from_float of this CPU's backend (what llama.cpp inference ran)
                  :activation-blocks act :result-bits bits :result (parse-double decimal)
                  ;; the platform-independent reference quantizer
                  :activation-ref-blocks act-ref :ref-result-bits ref-bits
                  :ref-result (parse-double ref-decimal)
                  ;; the scalar generic vec_dot over the reference activations, as this
                  ;; build compiled it (see :contraction in the manifest)
                  :generic-result-bits generic-bits
                  :generic-result (parse-double generic-decimal)}))))
        manifest {:llama-cpp llama
                  :cpu (oracle! "info")
                  ;; FMA contraction the C compiler applied in the generic dot products of
                  ;; this build (libggml-cpu is built with -march=native), from objdump.
                  :contraction {:q4_K :fma :q6_K :none :q5_0 :none :q8_0 :none}
                  :generator "dev/ggml_oracle/generate.clj"
                  :quantize quantized
                  :dot dots}]
    (spit (str out-dir "/manifest.edn") (with-out-str (pprint/pprint manifest)))
    (println "wrote" (count quantized) "quantization cases and" (count dots) "dot cases for llama.cpp" llama)))

(generate!)
(shutdown-agents)
