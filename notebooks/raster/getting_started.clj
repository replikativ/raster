;; # Getting Started with Raster

;; authors: Christian Weilbach

;; Raster is a typed multiple dispatch system for Clojure — think
;; Julia-style polymorphic arithmetic with JVM bytecode compilation.
;; You write math with `deftm`, and Raster compiles it to code that
;; runs at Java speed while keeping full REPL interactivity.

;; ## Setup

(ns raster.getting-started
  (:require [raster.core :refer [deftm ftm defvalue]]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength aclone]]
            [scicloj.kindly.v4.kind :as kind]))

;; ## Typed Functions: `deftm` and `ftm`
;;
;; `deftm` is Raster's replacement for `defn` in numerical code.
;; It looks like a normal function with type annotations:

(deftm add-scaled [x :- Double, y :- Double, alpha :- Double] :- Double
  (+ x (* alpha y)))

(add-scaled 1.0 2.0 3.0)  ;; => 7.0

;; The `:- Double` annotations tell Raster the parameter and return types.
;; Behind the scenes, `deftm` does much more than `defn`:
;;
;; - Generates a **typed JVM interface** with primitive parameters (no boxing)
;; - Stores a **walked body** — the compiler-processed form used for
;;   inlining, automatic differentiation, and `compile-aot`
;; - Registers in a **dispatch table** for multiple dispatch (Julia-style)
;; - Queues a **lazy JIT spec** — on first call, the bytecode compiler
;;   generates optimized JVM bytecode (branch folding, int loops,
;;   typed field access)
;; - Sets up **invokedynamic** call sites so HotSpot C2 can inline across
;;   deftm boundaries
;;
;; None of this happens with plain `defn` — which boxes all arguments
;; through `Object` and is invisible to the Raster compiler.

;; ## Multiple Dispatch
;;
;; `deftm` supports Julia-style multiple dispatch. Define the same function
;; name with different type signatures, and Raster picks the most specific
;; match at runtime:

(deftm dot-product
  [a :- (Array double), b :- (Array double)] :- Double
  (let [n (alength a)]
    (loop [i (int 0), acc 0.0]
      (if (>= i n)
        acc
        (recur (unchecked-inc-int i)
               (+ acc (* (aget a i) (aget b i))))))))

(dot-product (double-array [1 2 3]) (double-array [4 5 6]))  ;; => 32.0

;; You can add a Float32 version later — the dispatch system routes
;; calls to the right implementation automatically:

(deftm dot-product
  [a :- (Array float), b :- (Array float)] :- Float
  (let [n (alength a)]
    (loop [i (int 0), acc (float 0)]
      (if (>= i n)
        acc
        (recur (unchecked-inc-int i)
               (+ acc (* (aget a i) (aget b i))))))))

(dot-product (float-array [1 2 3]) (float-array [4 5 6]))  ;; => 32.0 (float)

;; ## Anonymous Typed Functions: `ftm`
;;
;; `ftm` is to `deftm` what `fn` is to `defn` — an anonymous typed
;; function. Use it for inline lambdas passed as arguments:

(let [scale (ftm [x :- Double, factor :- Double] :- Double
              (* x factor))]
  (scale 3.14 2.0))  ;; => 6.28

;; `ftm` generates the typed interface (zero-boxing calls), but does
;; **not** register a dispatch table entry, store a walked body, or
;; queue for lazy JIT. This is intentional — anonymous functions don't
;; need compiler integration since they're not referenced by name.
;;
;; **Rule of thumb:** If it has a name, use `deftm`. If it's a
;; throwaway lambda passed to a solver or combinator, use `ftm`.
;; Never write `(def x (ftm ...))` — that gives you the worst of
;; both worlds (named but invisible to the compiler).

;; ## Value Types: `defvalue`
;;
;; `defvalue` creates immutable value types with typed fields —
;; similar to Java records but with Valhalla value semantics
;; (when running on a Valhalla JDK):

(defvalue Point2D [x :- Double, y :- Double])

(let [p (->Point2D 3.0 4.0)]
  {:x (.x p)
   :y (.y p)
   :dist (n/sqrt (+ (* (.x p) (.x p)) (* (.y p) (.y p))))})

;; ## Polymorphic Arithmetic: `raster.numeric`
;;
;; `raster.numeric` provides `+`, `-`, `*`, `/`, `sqrt`, `pow`, etc.
;; that dispatch on argument types. They work on doubles, floats,
;; complex numbers, symbolic expressions, dual numbers (for AD),
;; and any custom type you register:

(n/+ 1.0 2.0)         ;; Double + Double → 3.0
(n/sqrt 2.0)           ;; → 1.4142135623730951
(n/pow 2.0 10.0)       ;; → 1024.0

;; These operators are what `deftm` bodies use for arithmetic.
;; The walker resolves them to the optimal concrete implementation
;; at compile time.

;; ## Array Operations
;;
;; `raster.arrays` provides typed `aget`, `aset`, `alength`, `aclone`
;; that work with any array type (double[], float[], long[], etc.)
;; and compile to the correct JVM array instructions:

(let [a (double-array [10 20 30 40 50])]
  {:length (alength a)
   :third (aget a 2)
   :sum (loop [i (int 0), s 0.0]
          (if (>= i (alength a)) s
              (recur (unchecked-inc-int i) (+ s (aget a i)))))})

;; ## The Compilation Pipeline
;;
;; When you call a `deftm` function, three things can happen:
;;
;; 1. **First call:** The lazy JIT compiles the function body to
;;    optimized JVM bytecode (branch folding, int-counted loops,
;;    typed field access). This is a one-time cost, then subsequent
;;    calls run at Java speed.
;;
;; 2. **Subsequent calls:** The compiled bytecode runs directly.
;;    C2 (the HotSpot JIT) can further optimize and inline.
;;
;; 3. **`compile-aot`:** For production hot paths, you can
;;    AOT-compile an entire call chain into one method. This inlines
;;    all deftm calls, eliminates dispatch overhead, and typically
;;    matches hand-written Java.
;;
;; The key insight: **the same code works at all three levels.** You
;; develop interactively at the REPL, and the compiler handles the rest.

;; ## What's Next?
;;
;; - **[ODE Solvers](ode_solvers.html)** — Solve differential equations
;;   (Lorenz, Lotka-Volterra, heat equation, stochastic DEs)
;; - **[Geometric Algebra](ga_ode_rotors.html)** — Rotor kinematics
;;   with native Multivector state
;; - **Automatic Differentiation** — Forward and reverse mode AD,
;;   gradients through ODE solvers
;; - **Linear Algebra** — Dense/sparse operations, LAPACK via Panama FFI
