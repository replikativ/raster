(ns raster.compiler.backend.wasm.node-util
  "Shared node/V8 runner for wasm test kernels. Chicory can PARSE (validate) a
  v128 module but cannot EXECUTE it, so SIMD execution is asserted on V8 via
  node — guarded, skipping where node is absent.

  The result is emitted via process.stdout.write(String(..)), NEVER console.log:
  node colorizes console.log(number) when it detects color support (e.g. an
  inherited FORCE_COLOR), producing ANSI-wrapped output like \"\\e[33m-4280\\e[39m\"
  that breaks Integer/parseInt. NO_COLOR=1 is set on the subprocess as belt and
  braces."
  (:require [clojure.string :as str]))

(defn node-available? []
  (try
    (let [p (-> (ProcessBuilder. ^java.util.List ["node" "--version"])
                (.redirectErrorStream true) (.start))]
      (.waitFor p) (zero? (.exitValue p)))
    (catch Exception _ false)))

(defn run-wasm-idot-on-node
  "Instantiate `wasm` on node/V8, write int8 vectors xs,ys into linear memory at
  offsets 0 and n, call `export-name`(0, n, n) and return the i32 result."
  [^bytes wasm export-name xs ys n]
  (let [dir (java.nio.file.Files/createTempDirectory
             "wasmnode" (make-array java.nio.file.attribute.FileAttribute 0))
        wp  (str dir "/k.wasm")
        jp  (str dir "/r.mjs")]
    (with-open [o (java.io.FileOutputStream. wp)] (.write o wasm))
    (spit jp (str "import fs from 'fs';\n"
                  "const b=fs.readFileSync('" wp "');\n"
                  "const {instance}=await WebAssembly.instantiate(b,{});\n"
                  "const m=new Int8Array(instance.exports.memory.buffer);\n"
                  "const xs=[" (str/join "," xs) "],ys=[" (str/join "," ys) "];\n"
                  "for(let i=0;i<" n ";i++){m[i]=xs[i];m[" n "+i]=ys[i];}\n"
                  "process.stdout.write(String(instance.exports." export-name "(0," n "," n ")));\n"))
    (let [pb (doto (ProcessBuilder. ^java.util.List ["node" jp])
               (.redirectErrorStream true))
          _  (.put (.environment pb) "NO_COLOR" "1")
          p  (.start pb)
          out (str/trim (slurp (.getInputStream p)))]
      (.waitFor p)
      (Integer/parseInt out))))
