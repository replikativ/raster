(ns raster.tooling.inspect
  "REPL tooling for inspecting compiled bytecode, JIT behavior, and JVMCI.

  Bytecode disassembly (works out of the box):
    (disassemble bytes)                    ; raw class bytes
    (disassemble-deftm 'raster.nn/dense_m_doubles_doubles_doubles)
    (disassemble-var #'my-specialized-fn)

  VM diagnostics (works out of the box):
    (vm-option \"TieredStopAtLevel\")      ; query VM flag
    (set-vm-option! \"PrintCompilation\" \"true\")
    (code-cache)                           ; code cache stats
    (list-diagnostic-commands)             ; all available jcmd commands

  Compiler directives (per-method JIT control, works out of the box):
    (add-compiler-directive! {:match [\"*__BC::*\"] :inline [\"+*::*\"]})
    (print-compiler-directives)
    (clear-compiler-directives!)

  JVMCI (needs --add-modules jdk.internal.vm.ci, included in :valhalla alias):
    (compilation-state #'my-fn)            ; check compilation tiers
    (compilation-state SomeClass \"invoke\") ; check by class + method name
    (native-disassemble #'my-fn)           ; disassemble native code (needs hsdis)

  Native assembly (requires hsdis — see `print-hsdis-build-instructions`):
    -XX:+UnlockDiagnosticVMOptions -XX:+PrintAssembly"
  (:require [clojure.walk]
            [clojure.string]
            [raster.compiler.core.dispatch])
  (:import (java.lang.classfile ClassFile Instruction)
           (java.lang.management ManagementFactory)
           (javax.management ObjectName)))

;; ================================================================
;; Bytecode disassembly via JDK 27 ClassFile API
;; ================================================================

(defn disassemble
  "Disassemble raw class bytes to stdout.
  Prints each method's instructions in a javap-like format.

  Usage:
    (disassemble some-byte-array)
    (disassemble some-byte-array {:method \"invoke\"})  ; filter to one method"
  ([^bytes class-bytes] (disassemble class-bytes {}))
  ([^bytes class-bytes {:keys [method]}]
   (let [cf    (ClassFile/of)
         model (.parse cf class-bytes)]
     ;; Class header
     (println (str "class " (.asInternalName (.thisClass model))
                   "  (version " (.majorVersion model) "." (.minorVersion model) ")"))
     (println)
     ;; Methods
     (doseq [m (.methods model)]
       (let [mname (.stringValue (.methodName m))
             mtype (.stringValue (.methodType m))]
         (when (or (nil? method) (= method mname))
           (println (str "  " mname " " mtype))
           (when-let [code (.orElse (.code m) nil)]
             (doseq [elem code]
               (when (instance? Instruction elem)
                 (println (str "    " elem)))))
           (println)))))))

(defn disassemble-var
  "Disassemble a var's compiled bytecode.
  Works with specialized and deftm-compiled vars.

  Usage:
    (disassemble-var #'raster.nn/dense_m_doubles_doubles_doubles)
    (disassemble-var #'raster.nn/relu_m_doubles)"
  ([v] (disassemble-var v {}))
  ([v opts]
   (let [m (meta v)
         bytes (or (:raster.core/bytecode-bytes m)
                   (:bytes (:raster.core/deftm-bytecode m)))]
     (if bytes
       (disassemble bytes opts)
       (println (str "No compiled bytecode found on " v
                     "\n  Available keys: " (pr-str (keys m))))))))

(defn disassemble-deftm
  "Disassemble a deftm method by its mangled symbol.

  Usage:
    (disassemble-deftm 'raster.nn/dense_m_doubles_doubles_doubles)
    (disassemble-deftm 'dense_m_doubles_doubles_doubles 'raster.nn)"
  ([sym]
   (let [ns-name (namespace sym)
         name-part (name sym)]
     (if ns-name
       (disassemble-deftm (symbol name-part) (the-ns (symbol ns-name)))
       (disassemble-deftm sym *ns*))))
  ([sym ns-obj]
   (if-let [v (ns-resolve ns-obj sym)]
     (disassemble-var v)
     (println (str "Cannot resolve: " sym " in " ns-obj)))))

;; ================================================================
;; ClassPrinter-based disassembly (richer output, needs module exports)
;; ================================================================

(defn disassemble-yaml
  "Disassemble class bytes to YAML format using ClassPrinter.
  Requires JVM flag: --add-exports=java.base/jdk.internal.classfile.components=ALL-UNNAMED

  Usage:
    (disassemble-yaml some-byte-array)
    (disassemble-yaml some-byte-array :trace-all)"
  ([^bytes class-bytes] (disassemble-yaml class-bytes :critical))
  ([^bytes class-bytes verbosity]
   (let [cf    (ClassFile/of)
         model (.parse cf class-bytes)
         printer-class (Class/forName "jdk.internal.classfile.components.ClassPrinter")
         verbosity-class (Class/forName "jdk.internal.classfile.components.ClassPrinter$Verbosity")
         verbosity-val (case verbosity
                         :members (aget (.getEnumConstants verbosity-class) 0)
                         :critical (aget (.getEnumConstants verbosity-class) 1)
                         :trace-all (aget (.getEnumConstants verbosity-class) 2))
         to-yaml (.getMethod printer-class "toYaml"
                             (into-array Class [(Class/forName "java.lang.classfile.CompoundElement")
                                                verbosity-class
                                                java.util.function.Consumer]))
         sb (StringBuilder.)]
     (.invoke to-yaml nil (object-array [model verbosity-val
                                         (reify java.util.function.Consumer
                                           (accept [_ s] (.append sb s)))]))
     (println (str sb)))))

;; ================================================================
;; VM Diagnostics (public API, no JVMCI needed)
;; ================================================================

(defn vm-option
  "Get a VM option value by name.

  Usage:
    (vm-option \"TieredStopAtLevel\")    ; => {:name ... :value \"4\" :writeable true}
    (vm-option \"PrintCompilation\")"
  [name]
  (let [bean (ManagementFactory/getPlatformMXBean
              (Class/forName "com.sun.management.HotSpotDiagnosticMXBean"))
        opt (.getVMOption bean name)]
    {:name (.getName opt)
     :value (.getValue opt)
     :writeable (.isWriteable opt)
     :origin (str (.getOrigin opt))}))

(defn set-vm-option!
  "Set a writeable VM option at runtime. Value must be a string.

  Usage:
    (set-vm-option! \"PrintCompilation\" \"true\")
    (set-vm-option! \"TieredStopAtLevel\" \"3\")   ; limit to C1 only
    (set-vm-option! \"PrintInlining\" \"true\")"
  [name value]
  (let [bean (ManagementFactory/getPlatformMXBean
              (Class/forName "com.sun.management.HotSpotDiagnosticMXBean"))]
    (.setVMOption bean name (str value))
    (vm-option name)))

;; ================================================================
;; Diagnostic Commands (jcmd equivalent from REPL)
;; ================================================================

(def ^:private ^ObjectName diag-cmd-object-name
  (ObjectName. "com.sun.management:type=DiagnosticCommand"))

(defn- mbean-server []
  (ManagementFactory/getPlatformMBeanServer))

(defn list-diagnostic-commands
  "List all available diagnostic commands (jcmd equivalents).
  Returns a vector of {:name ... :description ...} maps."
  []
  (let [server (mbean-server)
        info (.getMBeanInfo server diag-cmd-object-name)]
    (mapv (fn [op]
            {:name (.getName op)
             :description (.getDescription op)})
          (.getOperations info))))

(defn diagnostic-command
  "Execute a diagnostic command (jcmd equivalent) by name.
  Operation names use camelCase (e.g. \"vmVersion\", \"compilerCodecache\").
  Use (list-diagnostic-commands) to discover available commands.

  Usage:
    (diagnostic-command \"vmVersion\")
    (diagnostic-command \"compilerCodecache\")
    (diagnostic-command \"compilerCodelist\")"
  ([cmd-name]
   (let [server (mbean-server)]
     (.invoke server diag-cmd-object-name cmd-name
              (object-array [(into-array String [])])
              (into-array String ["[Ljava.lang.String;"]))))
  ([cmd-name & args]
   (let [server (mbean-server)]
     (.invoke server diag-cmd-object-name cmd-name
              (object-array [(into-array String (map str args))])
              (into-array String ["[Ljava.lang.String;"])))))

(defn code-cache
  "Print code cache statistics."
  []
  (println (diagnostic-command "compilerCodecache")))

(defn compiled-methods
  "List compiled methods in the code cache. Can be very long."
  []
  (diagnostic-command "compilerCodelist"))

(defn compile-queue
  "Show current compilation queue."
  []
  (println (diagnostic-command "compilerQueue")))

;; ================================================================
;; Compiler Directives (per-method JIT control)
;; ================================================================

(defn add-compiler-directive!
  "Add a compiler directive for per-method JIT control.
  Writes the directive to a temp file, then loads it via diagnostic command.

  Directive is a map with:
    :match    - vector of method patterns (\"pkg.Class::method\")
    :inline   - vector of inline patterns (\"+pkg.Class::method\" to force, \"-...\" to prevent)
    :PrintAssembly - true to print native assembly for matched methods
    :c1 / :c2 - tier-specific overrides (maps)

  Usage:
    ;; Force inline all methods in our bytecode classes:
    (add-compiler-directive! {:match [\"*__BC::*\"] :inline [\"+*::*\"]})

    ;; Print assembly for a specific method:
    (add-compiler-directive! {:match [\"*loss_mse*__BC::invoke\"]
                              :PrintAssembly true})

    ;; Disable C1 compilation (go straight to C2):
    (add-compiler-directive! {:match [\"*__BC::*\"]
                              :c1 {:Exclude true}})"
  [directive]
  (let [json-str (letfn [(to-json [v]
                           (cond
                             (map? v) (str "{"
                                           (clojure.string/join ","
                                                                (map (fn [[k v]]
                                                                       (str "\"" (name k) "\": " (to-json v)))
                                                                     v))
                                           "}")
                             (vector? v) (str "["
                                              (clojure.string/join ","
                                                                   (map to-json v))
                                              "]")
                             (string? v) (str "\"" v "\"")
                             (boolean? v) (str v)
                             (number? v) (str v)
                             :else (str "\"" v "\"")))]
                   (str "[" (to-json directive) "]"))
        tmp-file (java.io.File/createTempFile "compiler-directive" ".json")]
    (spit tmp-file json-str)
    (try
      (let [result (diagnostic-command "compilerDirectivesAdd" (.getAbsolutePath tmp-file))]
        (println result)
        result)
      (finally
        (.delete tmp-file)))))

(defn clear-compiler-directives!
  "Remove all custom compiler directives."
  []
  (println (diagnostic-command "compilerDirectivesClear")))

(defn print-compiler-directives
  "Print all active compiler directives."
  []
  (println (diagnostic-command "compilerDirectivesPrint")))

;; ================================================================
;; JVMCI — Method compilation state & native code
;; ================================================================

(defn- ensure-jvmci!
  "Check that JVMCI module is available. Throws with helpful message if not."
  []
  (try
    (Class/forName "jdk.vm.ci.hotspot.HotSpotJVMCIRuntime")
    true
    (catch ClassNotFoundException _
      (throw (ex-info
              (str "JVMCI not available. Start REPL with :valhalla alias, or add:\n"
                   "  --add-modules jdk.internal.vm.ci\n"
                   "  --add-exports jdk.internal.vm.ci/jdk.vm.ci.hotspot=ALL-UNNAMED\n"
                   "  --add-exports jdk.internal.vm.ci/jdk.vm.ci.meta=ALL-UNNAMED\n"
                   "  --add-exports jdk.internal.vm.ci/jdk.vm.ci.runtime=ALL-UNNAMED")
              {:type :jvmci-not-available})))))

(def ^:private jvmci-state (atom nil))

(defn- get-jvmci!
  "Lazily initialize and return JVMCI runtime components.
  Returns {:runtime ... :meta-access ... :c2vm ...}."
  []
  (ensure-jvmci!)
  (or @jvmci-state
      (let [rt-class (Class/forName "jdk.vm.ci.hotspot.HotSpotJVMCIRuntime")
            runtime (.invoke (.getMethod rt-class "runtime" (into-array Class []))
                             nil (object-array []))
            ;; HotSpotJVMCIRuntime → getHostJVMCIBackend() → JVMCIBackend
            backend (.invoke (.getMethod (class runtime) "getHostJVMCIBackend"
                                         (into-array Class []))
                             runtime (object-array []))
            ;; JVMCIBackend → getMetaAccess() → MetaAccessProvider
            meta-access (.invoke (.getMethod (class backend) "getMetaAccess"
                                             (into-array Class []))
                                 backend (object-array []))
            ;; HotSpotJVMCIRuntime → getCompilerToVM() (public but class is package-private)
            c2vm (.invoke (.getMethod (class runtime) "getCompilerToVM"
                                      (into-array Class []))
                          runtime (object-array []))
            state {:runtime runtime :backend backend
                   :meta-access meta-access :c2vm c2vm}]
        (reset! jvmci-state state)
        state)))

(defn- ->jvmci-method
  "Convert a java.lang.reflect.Method or Constructor to a JVMCI ResolvedJavaMethod."
  [^java.lang.reflect.Executable method]
  (let [{:keys [meta-access]} (get-jvmci!)
        lookup-method (.getMethod (class meta-access) "lookupJavaMethod"
                                  (into-array Class [java.lang.reflect.Executable]))]
    (.invoke lookup-method meta-access (object-array [method]))))

(defn- jvmci-invoke
  "Invoke a method by name on a JVMCI object, using the interface class to avoid
  access restrictions on the impl class."
  ([obj method-name]
   (jvmci-invoke obj method-name (into-array Class []) (object-array [])))
  ([obj method-name param-types args]
   (let [;; Try interface first (HotSpotResolvedJavaMethod), then impl class
         iface (Class/forName "jdk.vm.ci.hotspot.HotSpotResolvedJavaMethod")
         m (try (.getMethod iface method-name param-types)
                (catch NoSuchMethodException _
                  ;; Try meta interface
                  (try (.getMethod (Class/forName "jdk.vm.ci.meta.ResolvedJavaMethod")
                                   method-name param-types)
                       (catch NoSuchMethodException _
                         ;; Fall back to impl class with setAccessible
                         (doto (.getMethod (class obj) method-name param-types)
                           (.setAccessible true))))))]
     (.invoke m obj args))))

(defn- jvmci-method-info
  "Extract compilation info from a JVMCI ResolvedJavaMethod."
  [jmethod]
  {:name (jvmci-invoke jmethod "getName")
   :compiled? (jvmci-invoke jmethod "hasCompiledCode")
   :c1? (jvmci-invoke jmethod "hasCompiledCodeAtLevel"
                      (into-array Class [Integer/TYPE]) (object-array [(int 3)]))
   :c2? (jvmci-invoke jmethod "hasCompiledCodeAtLevel"
                      (into-array Class [Integer/TYPE]) (object-array [(int 4)]))
   :force-inline? (jvmci-invoke jmethod "isForceInline")
   :intrinsic-candidate? (jvmci-invoke jmethod "isIntrinsicCandidate")})

(defn compilation-state
  "Check the JIT compilation state of a method.
  Returns a map with :compiled?, :c1?, :c2?, :force-inline?, etc.

  Usage:
    ;; By var (for bytecode-compiled methods):
    (compilation-state #'raster.nn/dense_m_doubles_doubles_doubles)

    ;; By class + method name:
    (compilation-state SomeClass \"invoke\")

    ;; By reflection Method:
    (compilation-state (.getDeclaredMethod String \"valueOf\" (into-array Class [Object])))"
  ([v-or-class]
   (cond
     (var? v-or-class)
     (let [m (meta v-or-class)
           cls (or (:raster.core/bytecode-class m)
                   (:class (:raster.core/deftm-bytecode m)))]
       (if cls
         (let [methods (.getDeclaredMethods ^Class cls)]
           (mapv (fn [^java.lang.reflect.Method m]
                   (jvmci-method-info (->jvmci-method m)))
                 methods))
         (throw (ex-info (str "No bytecode class on var " v-or-class
                              ". Available keys: " (pr-str (keys m)))
                         {:var v-or-class}))))

     (instance? java.lang.reflect.Executable v-or-class)
     (jvmci-method-info (->jvmci-method v-or-class))

     :else
     (throw (ex-info "Expected var, Class, or java.lang.reflect.Method"
                     {:got (type v-or-class)}))))
  ([^Class cls method-name]
   (let [methods (->> (.getDeclaredMethods cls)
                      (filter #(= (.getName ^java.lang.reflect.Method %) method-name)))]
     (if (seq methods)
       (mapv (fn [^java.lang.reflect.Method m]
               (jvmci-method-info (->jvmci-method m)))
             methods)
       (throw (ex-info (str "No method '" method-name "' on " cls)
                       {:class cls :method method-name
                        :available (mapv #(.getName ^java.lang.reflect.Method %)
                                         (.getDeclaredMethods cls))}))))))

(defn- get-unsafe
  "Get sun.misc.Unsafe instance for low-level field access."
  []
  (let [f (doto (.getDeclaredField (Class/forName "sun.misc.Unsafe") "theUnsafe")
            (.setAccessible true))]
    (.get f nil)))

(defn- disassemble-jmethod
  "Disassemble a JVMCI ResolvedJavaMethod's compiled native code.
  Returns the disassembly string, or nil if not compiled or hsdis unavailable."
  [jmethod]
  (let [{:keys [c2vm]} (get-jvmci!)
        get-cc (doto (.getDeclaredMethod (class jmethod) "getCompiledCode" (into-array Class []))
                 (.setAccessible true))
        nmethod-ptr (long (.invoke get-cc jmethod (object-array [])))]
    (when (pos? nmethod-ptr)
      (let [nmethod-class (Class/forName "jdk.vm.ci.hotspot.HotSpotNmethod")
            ctor (doto (.getDeclaredConstructor nmethod-class
                                                (into-array Class [(class jmethod) String Boolean/TYPE Boolean/TYPE Long/TYPE]))
                   (.setAccessible true))
            nmethod-obj (.newInstance ctor (object-array [jmethod (jvmci-invoke jmethod "getName")
                                                          false false (long 0)]))
            unsafe (get-unsafe)
            installed-code-class (Class/forName "jdk.vm.ci.code.InstalledCode")
            addr-offset (.objectFieldOffset ^sun.misc.Unsafe unsafe
                                            (.getDeclaredField installed-code-class "address"))]
        (.putLong ^sun.misc.Unsafe unsafe nmethod-obj addr-offset nmethod-ptr)
        (let [disasm-m (doto (.getDeclaredMethod (class c2vm) "disassembleCodeBlob"
                                                 (into-array Class [installed-code-class]))
                         (.setAccessible true))]
          (str (.invoke disasm-m c2vm (object-array [nmethod-obj]))))))))

(defn native-disassemble
  "Disassemble the native (JIT-compiled) code of a method.
  Requires hsdis plugin installed (see print-hsdis-build-instructions).
  Method must be JIT-compiled first (use force-jit! if needed).

  Usage:
    (force-jit! #(my-fn 1.0 2.0) 20000)
    (native-disassemble #'my-fn)
    (native-disassemble SomeClass \"invoke\")"
  ([v-or-class]
   (cond
     (var? v-or-class)
     (let [m (meta v-or-class)
           cls (or (:raster.core/bytecode-class m)
                   (:class (:raster.core/deftm-bytecode m)))]
       (if cls
         (doseq [^java.lang.reflect.Method method (.getDeclaredMethods ^Class cls)]
           (native-disassemble cls (.getName method)))
         (throw (ex-info (str "No bytecode class on var " v-or-class
                              ". Available keys: " (pr-str (keys m)))
                         {:var v-or-class}))))

     (instance? Class v-or-class)
     (doseq [^java.lang.reflect.Method method (.getDeclaredMethods ^Class v-or-class)]
       (native-disassemble v-or-class (.getName method)))

     :else
     (throw (ex-info "Expected var or Class" {:got (type v-or-class)}))))
  ([^Class cls method-name]
   (let [methods (->> (.getDeclaredMethods cls)
                      (filter #(= (.getName ^java.lang.reflect.Method %) method-name)))]
     (doseq [^java.lang.reflect.Method method methods]
       (let [jmethod (->jvmci-method method)
             info (jvmci-method-info jmethod)]
         (println (str "=== " method-name " " (.toGenericString method) " ==="))
         (println (str "  compiled: " (:compiled? info)
                       " c1: " (:c1? info) " c2: " (:c2? info)))
         (if (:compiled? info)
           (if-let [asm (disassemble-jmethod jmethod)]
             (println asm)
             (println "  disassembleCodeBlob returned nil — is hsdis installed?"))
           (println "  Not yet compiled. Use (force-jit! thunk) first.")))))))

(defn native-disassemble-deftm
  "Disassemble a deftm method's JIT-compiled native code by its mangled symbol.
  JIT-compiles the method first if needed by calling the warmup thunk.

  Usage:
    ;; Disassemble the kepler Hamiltonian:
    (native-disassemble-deftm
      'user/test-kepler_m_double_double_double_double
      #(test-kepler 1.0 0.0 0.0 1.0))

    ;; Just the mangled symbol (must already be JIT-compiled):
    (native-disassemble-deftm 'raster.numeric/_plus__m_double_double)"
  ([mangled-sym]
   (native-disassemble-deftm mangled-sym nil))
  ([mangled-sym warmup-thunk]
   (when warmup-thunk
     (dotimes [_ 50000] (warmup-thunk)))
   (if-let [v (resolve mangled-sym)]
     (let [m (meta v)
           cls (or (:raster.core/bytecode-class m)
                   (:class (:raster.core/deftm-bytecode m)))]
       (if cls
         (native-disassemble cls "invk")
         (println (str "No bytecode class on " mangled-sym
                       ". Keys: " (pr-str (keys m))))))
     (println (str "Cannot resolve: " mangled-sym)))))

(defn set-not-inlinable!
  "Mark a method as not inlinable or compilable via JVMCI.
  This is a per-method setting that survives across compilations.

  Usage:
    (set-not-inlinable! SomeClass \"someMethod\")"
  [^Class cls method-name]
  (let [methods (->> (.getDeclaredMethods cls)
                     (filter #(= (.getName ^java.lang.reflect.Method %) method-name)))]
    (doseq [^java.lang.reflect.Method method methods]
      (let [jmethod (->jvmci-method method)]
        (jvmci-invoke jmethod "setNotInlinableOrCompilable")
        (println (str "Marked " method-name " as not inlinable/compilable"))))))

;; ================================================================
;; JVMCI Profiling — HotSpot type/method profiles
;; ================================================================

(defn profiling-info
  "Get HotSpot profiling data for a method via JVMCI ProfilingInfo.
  Returns the raw ProfilingInfo object for a given var, class+method, or reflect Method.

  Usage:
    (profiling-info #'my-fn)
    (profiling-info SomeClass \"invoke\")"
  ([v-or-class]
   (cond
     (var? v-or-class)
     (let [m (meta v-or-class)
           cls (or (:raster.core/bytecode-class m)
                   (:class (:raster.core/deftm-bytecode m)))]
       (if cls
         (mapv (fn [^java.lang.reflect.Method method]
                 (let [jm (->jvmci-method method)]
                   {:method (.getName method)
                    :descriptor (.toGenericString method)
                    :profiling-info (jvmci-invoke jm "getProfilingInfo")}))
               (.getDeclaredMethods ^Class cls))
         (throw (ex-info (str "No bytecode class on var " v-or-class) {:var v-or-class}))))

     (instance? java.lang.reflect.Executable v-or-class)
     (let [jm (->jvmci-method v-or-class)]
       {:method (.getName ^java.lang.reflect.Executable v-or-class)
        :profiling-info (jvmci-invoke jm "getProfilingInfo")})

     :else
     (throw (ex-info "Expected var, Class, or java.lang.reflect.Method"
                     {:got (type v-or-class)}))))
  ([^Class cls method-name]
   (let [methods (->> (.getDeclaredMethods cls)
                      (filter #(= (.getName ^java.lang.reflect.Method %) method-name)))]
     (if (seq methods)
       (mapv (fn [^java.lang.reflect.Method method]
               (let [jm (->jvmci-method method)]
                 {:method method-name
                  :descriptor (.toGenericString method)
                  :profiling-info (jvmci-invoke jm "getProfilingInfo")}))
             methods)
       (throw (ex-info (str "No method '" method-name "' on " cls)
                       {:class cls :method method-name}))))))

(defn type-profile
  "Extract type profile at a specific bytecode index from a ProfilingInfo object.
  Returns {:types [{:type <name> :probability <double>}] :null-seen <tristate> :mature? bool}
  or nil if no profile at that BCI.

  Usage:
    (let [pi (:profiling-info (first (profiling-info #'my-fn)))]
      (type-profile pi 42))"
  [prof-info ^long bci]
  (let [tp (.getTypeProfile prof-info bci)]
    (when tp
      {:types (mapv (fn [pt]
                      {:type (str (.getType pt))
                       :probability (.getProbability pt)})
                    (.getTypes tp))
       :null-seen (str (.getNullSeen tp))
       :not-recorded (.getNotRecordedProbability tp)
       :monomorphic (when-let [single (.asSingleType tp)]
                      (str single))})))

(defn method-profile
  "Extract method/call-target profile at a specific bytecode index.
  Returns {:methods [{:method <name> :probability <double>}] :not-recorded <double>}
  or nil if no profile at that BCI.

  Usage:
    (let [pi (:profiling-info (first (profiling-info #'my-fn)))]
      (method-profile pi 42))"
  [prof-info ^long bci]
  (let [mp (.getMethodProfile prof-info bci)]
    (when mp
      {:methods (mapv (fn [pm]
                        {:method (str (.getMethod pm))
                         :probability (.getProbability pm)})
                      (.getMethods mp))
       :not-recorded (.getNotRecordedProbability mp)})))

(defn scan-profiles
  "Scan all BCIs in a ProfilingInfo for type and method profiles.
  Returns a map of {bci {:type-profile ... :method-profile ... :exec-count ...}}
  for BCIs that have profiling data.

  Usage:
    (let [pi (:profiling-info (first (profiling-info #'my-fn)))]
      (scan-profiles pi))"
  [prof-info]
  (let [code-size (.getCodeSize prof-info)]
    (into (sorted-map)
          (for [bci (range code-size)
                :let [tp (type-profile prof-info bci)
                      mp (method-profile prof-info bci)
                      exec (.getExecutionCount prof-info bci)
                      branch (.getBranchTakenProbability prof-info bci)]
                :when (or tp mp (pos? exec) (>= branch 0.0))]
            [bci (cond-> {}
                   tp (assoc :type-profile tp)
                   mp (assoc :method-profile mp)
                   (pos? exec) (assoc :exec-count exec)
                   (>= branch 0.0) (assoc :branch-probability branch))]))))

(defn dispatch-profile
  "High-level: for a compiled deftm or bytecode-compiled var, show what HotSpot
  has observed at each dispatch/call site. Prints type profiles, method profiles,
  and execution counts.

  The method must be JIT-compiled first (use force-jit!).

  Usage:
    (force-jit! #(my-fn 1.0 2.0) 20000)
    (dispatch-profile #'my-fn)"
  [v-or-class & [method-name]]
  (let [infos (if method-name
                (profiling-info v-or-class method-name)
                (profiling-info v-or-class))]
    (doseq [{:keys [method descriptor profiling-info]} (if (map? infos) [infos] infos)]
      (println (str "=== " method " " descriptor " ==="))
      (let [mature? (.isMature profiling-info)
            profiles (scan-profiles profiling-info)]
        (println (str "  mature: " mature? "  code-size: " (.getCodeSize profiling-info)
                      "  profiled-sites: " (count profiles)))
        (doseq [[bci data] profiles]
          (print (str "  @" bci ":"))
          (when-let [ec (:exec-count data)]
            (print (str " exec=" ec)))
          (when-let [bp (:branch-probability data)]
            (print (str " branch=" (format "%.3f" bp))))
          (println)
          (when-let [tp (:type-profile data)]
            (if (:monomorphic tp)
              (println (str "    types: MONO " (:monomorphic tp)))
              (do
                (doseq [t (:types tp)]
                  (println (str "    type: " (:type t) " p=" (format "%.3f" (:probability t)))))
                (when (pos? (:not-recorded tp))
                  (println (str "    not-recorded: " (format "%.3f" (:not-recorded tp))))))))
          (when-let [mp (:method-profile data)]
            (doseq [m (:methods mp)]
              (println (str "    method: " (:method m) " p=" (format "%.3f" (:probability m)))))
            (when (pos? (:not-recorded mp))
              (println (str "    not-recorded: " (format "%.3f" (:not-recorded mp)))))))
        (println)))))

;; ================================================================
;; Devirtualization analysis
;; ================================================================

(defn analyze-devirtualization
  "Analyze a walked body form for devirtualization status.
  Counts devirtualized (.invk) calls vs still-dispatched (raster.numeric/*, etc.) calls.
  Returns {:devirtualized n :dispatched n :dispatch-ops [op-names] :pct <double>}.

  Usage:
    ;; Analyze a deftm var's walked body:
    (analyze-devirtualization (first (::raster.core/deftm-walked-body (meta (resolve 'my-fn_m_double_double)))))

    ;; Analyze a pipeline stage:
    (analyze-devirtualization some-form)"
  [form]
  (let [devirt (atom 0) disp (atom 0) disp-ops (atom [])]
    (clojure.walk/postwalk
     (fn [f]
       (when (and (seq? f) (symbol? (first f)))
         (let [op (str (first f))]
           (cond
             (.startsWith op ".invk") (swap! devirt inc)
             (or (.startsWith op "raster.numeric/")
                 (.startsWith op "raster.math/"))
             (do (swap! disp inc) (swap! disp-ops conj op)))))
       f)
     form)
    (let [d @devirt s @disp total (+ d s)]
      {:devirtualized d
       :dispatched s
       :dispatch-ops (vec (distinct @disp-ops))
       :pct (if (pos? total) (* 100.0 (/ (double d) total)) 100.0)})))

(defn analyze-pipeline-devirtualization
  "Analyze devirtualization across all pipeline stages for a deftm var.
  Requires raster.compiler.pipeline namespace.

  Usage:
    (analyze-pipeline-devirtualization #'raster.numeric/+ 'raster.numeric/_plus__m_double_double :gradient)"
  [dispatch-var mangled-sym pipeline-type]
  (require 'raster.compiler.pipeline)
  (let [resolve-fn (resolve 'raster.compiler.pipeline/resolve-deftm-var)
        show-fn (resolve 'raster.compiler.pipeline/show-pipeline)
        result (show-fn (resolve mangled-sym) pipeline-type)]
    (println (str "=== Devirtualization for " mangled-sym " (" (name pipeline-type) ") ==="))
    (doseq [[stage form] result
            :when form]
      (let [{:keys [devirtualized dispatched pct dispatch-ops]} (analyze-devirtualization form)]
        (println (str "  " (name stage) ": "
                      devirtualized " devirt, " dispatched " dispatched ("
                      (format "%.0f%%" pct) ")"
                      (when (seq dispatch-ops)
                        (str " — " (clojure.string/join ", " dispatch-ops)))))))))

;; ================================================================
;; JIT diagnostics helpers
;; ================================================================

(defn print-jit-flags
  "Print JVM flags for JIT diagnostics.

  Add these to your REPL startup command or deps.edn :jvm-opts to see
  what the JIT compiler is doing."
  []
  (println "=== JIT Compilation Monitoring ===")
  (println "Add to :jvm-opts in deps.edn or REPL startup:")
  (println)
  (println "  ;; See which methods get JIT-compiled:")
  (println "  \"-XX:+UnlockDiagnosticVMOptions\"")
  (println "  \"-XX:+PrintCompilation\"")
  (println)
  (println "  ;; See inlining decisions (very verbose):")
  (println "  \"-XX:+PrintInlining\"")
  (println)
  (println "  ;; Log compilation to XML for offline analysis:")
  (println "  \"-XX:+LogCompilation\"")
  (println "  \"-XX:LogFile=compilation.log\"")
  (println)
  (println "  ;; Filter to specific class:")
  (println "  \"-XX:CompileCommand=print,raster.nn*::*\"")
  (println)
  (println "=== Runtime VM options (no restart needed) ===")
  (println "  (set-vm-option! \"PrintCompilation\" \"true\")")
  (println "  (set-vm-option! \"PrintInlining\" \"true\")")
  (println)
  (println "=== Compiler Directives (per-method control) ===")
  (println "  (add-compiler-directive! {:match [\"*__BC::*\"] :inline [\"+*::*\"]})")
  (println "  (print-compiler-directives)")
  (println "  (clear-compiler-directives!)")
  (println)
  (println "=== JVMCI (compilation state) ===")
  (println "  (compilation-state #'my-fn)        ; check if JIT-compiled")
  (println "  (compilation-state MyClass \"invoke\")  ; by class + method")
  (println)
  (println "=== Native Assembly (requires hsdis) ===")
  (println "  \"-XX:+PrintAssembly\"")
  (println "  \"-XX:PrintAssemblyOptions=intel\"")
  (println)
  (println "Run (raster.tooling.inspect/print-hsdis-build-instructions) for hsdis setup."))

(defn print-hsdis-build-instructions
  "Print instructions for building the hsdis disassembler plugin,
  which enables -XX:+PrintAssembly for native code inspection."
  []
  (println "=== Building hsdis (HotSpot Disassembler) ===")
  (println)
  (println "hsdis enables -XX:+PrintAssembly to show JIT-compiled native assembly.")
  (println "Like Julia's @code_native but at JVM level.")
  (println)
  (println "1. Install capstone (disassembler backend):")
  (println "   sudo apt install libcapstone-dev")
  (println)
  (println "2. Configure and build hsdis in the Valhalla JDK source:")
  (println "   cd ~/Development/valhalla-jdk")
  (println "   bash configure --with-hsdis=capstone")
  (println "   make build-hsdis")
  (println "   make install-hsdis")
  (println)
  (println "3. Or for the mainline JDK:")
  (println "   cd ~/Development/jdk")
  (println "   bash configure --with-hsdis=capstone")
  (println "   make build-hsdis")
  (println "   make install-hsdis")
  (println)
  (println "4. Then add to JVM flags:")
  (println "   -XX:+UnlockDiagnosticVMOptions -XX:+PrintAssembly")
  (println "   -XX:PrintAssemblyOptions=intel")
  (println "   -XX:CompileCommand=print,<classname>::<method>")
  (println)
  (println "The hsdis shared library goes to: $JAVA_HOME/lib/hsdis-amd64.so"))

(defn force-jit!
  "Force JIT compilation of a function by calling it many times.
  If JVMCI is available, verifies compilation state afterwards.

  Usage:
    (force-jit! #(my-fn arg1 arg2) 20000)"
  ([thunk] (force-jit! thunk 20000))
  ([thunk n]
   (dotimes [_ n] (thunk))
   (println (str "Invoked " n " times — should be JIT-compiled now."
                 "\nUse (compilation-state ...) to verify."))))

;; ================================================================
;; Dispatch tracing and introspection
;; ================================================================

(defn methods-for
  "List all registered methods for a deftm var with their type signatures.
  Returns a map of {arity -> [{:tags [...] :specificity n} ...]}.

  Usage:
    (methods-for #'raster.numeric/+)
    (methods-for 'raster.numeric/+)"
  [fn-name-or-var]
  (require 'raster.core)
  (let [methods-of (resolve 'raster.core/methods-of)]
    (when-let [table (methods-of fn-name-or-var)]
      (into (sorted-map)
            (map (fn [[arity entries]]
                   [arity (mapv (fn [e]
                                  {:tags (:tags e)
                                   :specificity (:specificity e)
                                   :typed? (boolean (:typed-impl e))})
                                entries)]))
            table))))

(defn type-hierarchy
  "Print the subtype registration hierarchy.
  Shows all registered subtype relationships.

  Usage:
    (type-hierarchy)"
  []
  (require 'raster.core)
  (let [subtypes @@(resolve 'raster.compiler.core.dispatch/subtype-registry)]
    (println "Subtype registrations:")
    (println "======================")
    (if (empty? subtypes)
      (println "  (none)")
      (doseq [[child parents] (sort-by #(.getSimpleName ^Class (key %)) subtypes)]
        (println (str "  " (.getSimpleName ^Class child)
                      " <: " (clojure.string/join ", "
                                                  (map #(.getSimpleName ^Class %) parents))))))))

(defmacro trace-dispatch
  "Execute body with dispatch tracing enabled.
  Prints each deftm dispatch: function name, arg types, and selected method.

  Usage:
    (trace-dispatch
      (raster.numeric/+ 1.0 2.0)
      (raster.numeric/* 3 4))"
  [& body]
  `(binding [raster.compiler.core.dispatch/*trace-dispatch* true]
     ~@body))

;; ================================================================
;; Compiled pipeline diagnostics
;; ================================================================

(defn compiled-diagnostics
  "Return diagnostics for a compiled deftm var, or nil if not compiled.
   Diagnostics are stored by compile-aot and include:
     :helper-map    — {name → {:binding sym, :expr s-expr, :params [...], :return-tag tag}}
     :statics-bytes — byte[] of the compute class (for disassembly)
     :wrapper-bytes — byte[] of the wrapper class
     :compute-form  — the S-expression fed to the bytecode compiler
     :alloc-form    — the allocation function (Clojure-eval'd)
     :hoisted-count — number of hoisted buffer allocations

  Usage:
    (compiled-diagnostics #'raster.mnist-bench/train-step)"
  [f-var]
  (:raster.compiler.pipeline/compiled-diagnostics (meta f-var)))

(defn disassemble-helper
  "Disassemble a specific helper method from a compiled deftm var.
  Shows the source S-expression, bytecode, and optionally native assembly.

  Usage:
    (disassemble-helper #'my-fn \"helper_6\")
    (disassemble-helper #'my-fn \"helper_6\" :native? true)"
  [f-var helper-name & {:keys [native?] :or {native? false}}]
  (if-let [diag (compiled-diagnostics f-var)]
    (do
      ;; Source expression
      (when-let [info (get (:helper-map diag) helper-name)]
        (println (str "=== " helper-name " ==="))
        (println "Binding:" (:binding info))
        (println "Return:" (:return-tag info))
        (println "Params:" (pr-str (mapv #(vector % (:tag (meta %))) (:params info))))
        (println "\nSource expression:")
        (clojure.pprint/pprint (:expr info))
        (println))
      ;; Bytecode
      (when-let [bytes (:statics-bytes diag)]
        (println "Bytecode:")
        (disassemble bytes {:method helper-name})
        (println))
      ;; Native assembly (if requested and available)
      (when (and native? (:statics-bytes diag))
        (let [cf-name (clojure.string/replace (.getName (class @f-var)) "$W" "")]
          (try
            (let [cls (Class/forName cf-name)]
              (println "Native assembly:")
              (native-disassemble cls helper-name))
            (catch Exception e
              (println "Native disassembly not available:" (.getMessage e)))))))
    (println "No compiled diagnostics found on" f-var
             "\n  Compile with (pipeline/compile-aot" f-var ") first.")))

(defn list-helpers
  "List all helper methods in a compiled deftm var with their signatures.

  Usage:
    (list-helpers #'raster.mnist-bench/train-step)"
  [f-var]
  (if-let [diag (compiled-diagnostics f-var)]
    (let [hm (:helper-map diag)]
      (println (str (count hm) " helpers:"))
      (doseq [[name info] (sort-by key hm)]
        (println (format "  %s: binding=%s return=%s params=%d"
                         name (:binding info) (:return-tag info)
                         (count (:params info)))))
      (println (str "\n" (:hoisted-count diag) " hoisted buffers")))
    (println "No compiled diagnostics found.")))

(defn disassemble-compute
  "Disassemble the main compute method of a compiled deftm var.

  Usage:
    (disassemble-compute #'raster.mnist-bench/train-step)"
  [f-var]
  (if-let [diag (compiled-diagnostics f-var)]
    (when-let [bytes (:statics-bytes diag)]
      (disassemble bytes {:method "compute"}))
    (println "No compiled diagnostics found.")))
