(ns raster.dl.decoder-layer-gpu-test
  "One full gemma-style decoder layer as a resident GPU command graph (hand-authored op-chain via
   gpu.core/chain-program!), validated end-to-end against a plain-Clojure CPU reference. 25 ops:
   input-norm → q/k/v proj (Q4_K) → q/k-norm → RoPE → kv-append → GQA attention → o-proj →
   post-attn norm → residual → pre-FFN norm → gate/up (Q4_K) → GeGLU → down (Q4_K) → post-FFN norm →
   residual. Weights :constant resident, KV :state, all matmuls Q4_K-quantized. The decode-layer
   substrate for full gemma (the per-token loop + real weights are #32/#36)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.qlinear-k :as qk] [raster.compiler.backend.cpu.quant :as q]
            [raster.dl.nn :as nn] [raster.dl.attention :as attn] [raster.gpu.core :as gpu]))

(defn- gpu-available? []
  (try (require 'raster.gpu.ze-runtime)
       (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices)))) (catch Throwable _ false)))
(defn- gen [n seed] (let [a (float-array n) r (java.util.Random. seed)] (dotimes [i n] (aset a i (float (* 0.3 (- (.nextDouble r) 0.5))))) a))
(defn- b->i [^bytes b] (let [w (quot (alength b) 4) o (int-array w) bb (.order (java.nio.ByteBuffer/wrap b) java.nio.ByteOrder/LITTLE_ENDIAN)] (dotimes [i w] (aset o i (.getInt bb (* i 4)))) o))
(defn- relerr [a b] (/ (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) (seq a) (seq b))) (reduce max 1e-9 (map #(Math/abs (double %)) b))))
(defn- q4k-mm [xv in out qw] (let [{:keys [xq xs bsums]} (q/quantize-act-q8k xv in q/q4-K) {:keys [wq da db aq bq]} qw y (float-array out)] (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq y in out 0 out) y))
(defn- rmsnorm-cpu [x w rows feat eps gain] (let [out (float-array (* rows feat))] (dotimes [r rows] (let [off (* r feat) ms (/ (reduce + (map (fn [i] (let [v (aget x (+ off i))] (* v v))) (range feat))) (double feat)) inv (/ 1.0 (Math/sqrt (+ ms eps)))] (dotimes [i feat] (aset out (+ off i) (float (* (aget x (+ off i)) inv (+ gain (aget w i)))))))) out))
(defn- rope-cpu [x heads hd theta pos] (let [out (float-array (* heads hd)) hdim2 (quot hd 2) lt (Math/log (double theta))] (dotimes [h heads] (let [base (* h hd)] (dotimes [i hdim2] (let [freq (Math/exp (* (/ (* -2.0 i) (double hd)) lt)) ang (* (double pos) freq) c (Math/cos ang) s (Math/sin ang) x0 (aget x (+ base i)) x1 (aget x (+ base i hdim2))] (aset out (+ base i) (float (- (* x0 c) (* x1 s)))) (aset out (+ base i hdim2) (float (+ (* x1 c) (* x0 s)))))))) out))
(defn- attn-cpu [Q K V clen nq grp nkv hd scale] (let [out (float-array (* nq hd)) kvstride (* nkv hd)] (dotimes [hq nq] (let [hkv (quot hq grp) qb (* hq hd) hkvb (* hkv hd) scores (mapv (fn [j] (* scale (reduce + (map (fn [d] (* (aget Q (+ qb d)) (aget K (+ (* j kvstride) hkvb d)))) (range hd))))) (range clen)) mx (reduce max scores) ex (mapv #(Math/exp (- % mx)) scores) sm (reduce + ex) sf (mapv #(/ % sm) ex)] (dotimes [d hd] (aset out (+ qb d) (float (reduce + (map (fn [j] (* (nth sf j) (aget V (+ (* j kvstride) hkvb d)))) (range clen)))))))) out))
(defn- gelu-cpu [g u n] (let [o (float-array n) c 0.7978845608028654] (dotimes [i n] (let [x (aget g i)] (aset o i (float (* (* 0.5 x (+ 1.0 (Math/tanh (* c (+ x (* 0.044715 x x x)))))) (aget u i)))))) o))
(defn- add-cpu [a b n] (let [o (float-array n)] (dotimes [i n] (aset o i (float (+ (aget a i) (aget b i))))) o))

(deftest full-decoder-layer-gpu
  (when (gpu-available?)
    (testing "one full Q4_K decoder layer as a resident command graph matches CPU"
      (let [H 256 IM 512 nq 2 nkv 1 hd 128 grp 2 eps 1.0e-6 theta 1.0e4 scale (/ 1.0 (Math/sqrt (double hd))) kvrow (* nkv hd) p 2 clen (inc p)
            x (gen H 1) win (gen H 2) wqn (gen hd 3) wkn (gen hd 4) wpa (gen H 5) wpf (gen H 6) wpff (gen H 7)
            Wq (q/quantize-weight-q4k (gen (* (* nq hd) H) 11) q/q4-K) Wk (q/quantize-weight-q4k (gen (* kvrow H) 12) q/q4-K) Wv (q/quantize-weight-q4k (gen (* kvrow H) 13) q/q4-K) Wo (q/quantize-weight-q4k (gen (* H (* nq hd)) 33) q/q4-K)
            Wg (q/quantize-weight-q4k (gen (* IM H) 21) q/q4-K) Wu (q/quantize-weight-q4k (gen (* IM H) 22) q/q4-K) Wd (q/quantize-weight-q4k (gen (* H IM) 23) q/q4-K)
            priorK (gen (* p kvrow) 50) priorV (gen (* p kvrow) 51)
            xn (rmsnorm-cpu x win 1 H eps 1.0) Q (q4k-mm xn H (* nq hd) Wq) K (q4k-mm xn H kvrow Wk) V (q4k-mm xn H kvrow Wv)
            Qn (rmsnorm-cpu Q wqn nq hd eps 1.0) Kn (rmsnorm-cpu K wkn nkv hd eps 1.0) Qr (rope-cpu Qn nq hd theta p) Kr (rope-cpu Kn nkv hd theta p)
            cK (float-array (* clen kvrow)) cV (float-array (* clen kvrow))
            _ (System/arraycopy priorK 0 cK 0 (* p kvrow)) _ (System/arraycopy Kr 0 cK (* p kvrow) kvrow) _ (System/arraycopy priorV 0 cV 0 (* p kvrow)) _ (System/arraycopy V 0 cV (* p kvrow) kvrow)
            at (attn-cpu Qr cK cV clen nq grp nkv hd scale) O (q4k-mm at (* nq hd) H Wo)
            O2 (rmsnorm-cpu O wpa 1 H eps 1.0) x1 (add-cpu x O2 H)
            x1n (rmsnorm-cpu x1 wpf 1 H eps 1.0) gate (q4k-mm x1n H IM Wg) up (q4k-mm x1n H IM Wu) hh (gelu-cpu gate up IM)
            down (q4k-mm hh IM H Wd) down2 (rmsnorm-cpu down wpff 1 H eps 1.0) Oref (add-cpu x1 down2 H)
            wq* (fn [qw] {:wp (b->i (:wq qw)) :da (:da qw) :db (:db qw) :aq (:aq qw) :bq (:bq qw)})
            ck0 (float-array (* clen kvrow)) cv0 (float-array (* clen kvrow))
            _ (System/arraycopy priorK 0 ck0 0 (* p kvrow)) _ (System/arraycopy priorV 0 cv0 0 (* p kvrow))
            wbuf (fn [pre w] (let [w (wq* w)] {(keyword (str pre "p")) [:int (alength (:wp w)) (:wp w) :constant] (keyword (str pre "da")) [:float (alength (:da w)) (:da w) :constant] (keyword (str pre "db")) [:float (alength (:db w)) (:db w) :constant] (keyword (str pre "aq")) [:byte (alength (:aq w)) (:aq w) :constant] (keyword (str pre "bq")) [:byte (alength (:bq w)) (:bq w) :constant]}))
            buffers (merge
                     {:x [:float H x :input] :win [:float H win :constant] :wqn [:float hd wqn :constant] :wkn [:float hd wkn :constant] :wpa [:float H wpa :constant] :wpf [:float H wpf :constant] :wpff [:float H wpff :constant]
                      :xn [:float H nil :scratch] :qxp [:int (quot H 4) nil :scratch] :qxs [:float (quot H 256) nil :scratch] :qxb [:int (quot H 32) nil :scratch]
                      :Q [:float (* nq hd) nil :scratch] :K [:float kvrow nil :scratch] :V [:float kvrow nil :scratch] :Qn [:float (* nq hd) nil :scratch] :Kn [:float kvrow nil :scratch] :Qr [:float (* nq hd) nil :scratch] :Kr [:float kvrow nil :scratch]
                      :cK [:float (* clen kvrow) ck0 :state] :cV [:float (* clen kvrow) cv0 :state] :sc [:float (* nq clen) nil :scratch] :at [:float (* nq hd) nil :scratch]
                      :axp [:int (quot (* nq hd) 4) nil :scratch] :axs [:float (quot (* nq hd) 256) nil :scratch] :axb [:int (quot (* nq hd) 32) nil :scratch]
                      :O [:float H nil :scratch] :O2 [:float H nil :scratch] :x1 [:float H nil :scratch]
                      :x1n [:float H nil :scratch] :fxp [:int (quot H 4) nil :scratch] :fxs [:float (quot H 256) nil :scratch] :fxb [:int (quot H 32) nil :scratch]
                      :gate [:float IM nil :scratch] :up [:float IM nil :scratch] :hh [:float IM nil :scratch]
                      :hxp [:int (quot IM 4) nil :scratch] :hxs [:float (quot IM 256) nil :scratch] :hxb [:int (quot IM 32) nil :scratch]
                      :down [:float H nil :scratch] :down2 [:float H nil :scratch] :out [:float H nil :output]}
                     (wbuf "wq" Wq) (wbuf "wk" Wk) (wbuf "wv" Wv) (wbuf "wo" Wo) (wbuf "wg" Wg) (wbuf "wu" Wu) (wbuf "wd" Wd))
            qa (fn [ph xb xpb xsb bsb n] {:op #'qk/quant-act-q8k-gpu! :phase ph :bind {"x" xb "xp" xpb "xs" xsb "bsums" bsb} :scalars {} :n n})
            mm (fn [ph xpb xsb bsb pre yb in n] {:op #'qk/qmatmul-q4k-dp4a! :phase ph :bind {"xp" xpb "xs" xsb "bsums" bsb "wp" (keyword (str pre "p")) "da" (keyword (str pre "da")) "db" (keyword (str pre "db")) "aq" (keyword (str pre "aq")) "bq" (keyword (str pre "bq")) "y" yb} :scalars {"in" in} :n n})
            rms (fn [ph xb wb ob rows feat] {:op #'nn/rms-norm! :phase ph :bind {"x" xb "weight" wb "out" ob} :scalars {"eps" eps "features" feat "gain-offset" 1.0} :n rows})
            radd (fn [ph ab bb ob n] {:op #'nn/residual-add! :phase ph :bind {"a" ab "b" bb "out" ob} :scalars {} :n n})
            steps [(rms :n1 :x :win :xn 1 H) (qa :qx :xn :qxp :qxs :qxb (quot H 256))
                   (mm :mq :qxp :qxs :qxb "wq" :Q H (* nq hd)) (mm :mk :qxp :qxs :qxb "wk" :K H kvrow) (mm :mv :qxp :qxs :qxb "wv" :V H kvrow)
                   (rms :qn :Q :wqn :Qn nq hd) (rms :kn :K :wkn :Kn nkv hd)
                   {:op #'attn/rope-pos-gpu! :phase :rq :bind {"x" :Qn "out" :Qr} :scalars {"head-dim" hd "theta" theta "pos-offset" p} :n nq}
                   {:op #'attn/rope-pos-gpu! :phase :rk :bind {"x" :Kn "out" :Kr} :scalars {"head-dim" hd "theta" theta "pos-offset" p} :n nkv}
                   {:op #'attn/kv-append! :phase :ak :bind {"src" :Kr "cache" :cK} :scalars {"pos" p "kvrow" kvrow} :n kvrow}
                   {:op #'attn/kv-append! :phase :av :bind {"src" :V "cache" :cV} :scalars {"pos" p "kvrow" kvrow} :n kvrow}
                   {:op #'attn/gqa-decode-attention-gpu! :phase :att :bind {"q" :Qr "k" :cK "v" :cV "out" :at "sc" :sc} :scalars {"cache-len" clen "group" grp "head-dim" hd "n-kv" nkv "scale" scale} :n nq}
                   (qa :qaa :at :axp :axs :axb (quot (* nq hd) 256)) (mm :mo :axp :axs :axb "wo" :O (* nq hd) H)
                   (rms :npa :O :wpa :O2 1 H) (radd :r1 :x :O2 :x1 H)
                   (rms :npf :x1 :wpf :x1n 1 H) (qa :qf :x1n :fxp :fxs :fxb (quot H 256))
                   (mm :mg :fxp :fxs :fxb "wg" :gate H IM) (mm :mu :fxp :fxs :fxb "wu" :up H IM)
                   {:op #'nn/gelu-mul! :phase :ge :bind {"gate" :gate "up" :up "out" :hh} :scalars {} :n IM}
                   (qa :qh :hh :hxp :hxs :hxb (quot IM 256)) (mm :md :hxp :hxs :hxb "wd" :down IM H)
                   (rms :npff :down :wpff :down2 1 H) (radd :r2 :x1 :down2 :out H)]
            sess (gpu/make-session :ze:0)]
        (try (gpu/chain-program! sess buffers steps)
          (let [o (get (gpu/run-chain! sess {:x x}) :out)]
            (is (= 25 (count steps)))
            (is (< (relerr o Oref) 2e-2) "full decoder layer matches CPU Q4_K reference"))
          (finally (gpu/close-session! sess)))))))
