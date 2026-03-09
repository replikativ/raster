# Julia benchmark for Lotka-Volterra parameter estimation
# Equivalent to raster.sensitivity Clojure implementation
#
# Usage:
#   julia bench/julia_sensitivity_bench.jl
#
# Requires:
#   ] add OrdinaryDiffEqLowOrderRK ForwardDiff BenchmarkTools

using OrdinaryDiffEqLowOrderRK
using ForwardDiff
using BenchmarkTools
using Printf

# ================================================================
# Lotka-Volterra ODE
# ================================================================

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    xy = x * y
    du[1] = α * x - β * xy
    du[2] = δ * xy - γ * y
    nothing
end

# Out-of-place version for ForwardDiff (works with Dual numbers)
function lotka_volterra(u, p, t)
    x, y = u
    α, β, δ, γ = p
    xy = x * y
    [α * x - β * xy,
     δ * xy - γ * y]
end

# ================================================================
# Fixed-step RK4 (matches raster implementation exactly)
# ================================================================

function rk4_step(f, u, p, t, dt)
    half_dt = 0.5 * dt
    k1 = f(u, p, t)
    k2 = f(u .+ half_dt .* k1, p, t + half_dt)
    k3 = f(u .+ half_dt .* k2, p, t + half_dt)
    k4 = f(u .+ dt .* k3, p, t + dt)
    u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function rk4_solve_segments(f, u0, p, t0, save_times, dt)
    u = u0 .+ zero(eltype(p))  # promote to Dual if needed
    t = t0
    results = typeof(u)[]
    for t_target in save_times
        while t < t_target - 1e-12
            h = min(dt, t_target - t)
            u = rk4_step(f, u, p, t, h)
            t += h
        end
        push!(results, copy(u))
    end
    results
end

# ================================================================
# Loss function
# ================================================================

function loss_mse(predicted, observed)
    n = length(predicted)
    dim = length(predicted[1])
    s = zero(eltype(predicted[1]))
    for i in 1:n
        for j in 1:dim
            diff = predicted[i][j] - observed[i][j]
            s += diff * diff
        end
    end
    s / (n * dim)
end

# ================================================================
# Full loss: params -> scalar
# ================================================================

function lv_loss(params, u0, observed, save_times, dt)
    traj = rk4_solve_segments(lotka_volterra, u0, params, 0.0, save_times, dt)
    loss_mse(traj, observed)
end

# ================================================================
# Gradient descent
# ================================================================

function gradient_descent(loss_fn, params0, lr, n_iters)
    params = copy(params0)
    losses = Float64[]
    for _ in 1:n_iters
        g = ForwardDiff.gradient(loss_fn, params)
        loss = loss_fn(params)
        push!(losses, loss)
        params .-= lr .* g
    end
    params, losses
end

# ================================================================
# Main benchmark
# ================================================================

function main()
    # Problem setup (matches Clojure test exactly)
    u0 = [1.0, 1.0]
    true_params = [1.5, 1.0, 3.0, 1.0]
    initial_guess = [1.2, 0.8, 2.5, 0.8]
    save_times = [0.1 * i for i in 1:20]
    dt = 0.02
    lr = 0.001
    n_iters = 200

    # Generate observed data
    observed = rk4_solve_segments(lotka_volterra, u0, true_params, 0.0, save_times, dt)

    # Loss function (closed over data)
    loss_fn(p) = lv_loss(p, u0, observed, save_times, dt)

    # Warmup
    println("Warming up...")
    ForwardDiff.gradient(loss_fn, initial_guess)
    gradient_descent(loss_fn, initial_guess, lr, 10)

    # Benchmark single gradient evaluation
    println("\n=== Single gradient evaluation ===")
    b_grad = @benchmark ForwardDiff.gradient($loss_fn, $initial_guess)
    display(b_grad)

    # Benchmark full 200-iteration gradient descent
    println("\n=== 200-iteration gradient descent ===")
    b_gd = @benchmark gradient_descent($loss_fn, $initial_guess, $lr, $n_iters)
    display(b_gd)

    # Verify convergence
    println("\n=== Convergence check ===")
    final_params, losses = gradient_descent(loss_fn, initial_guess, lr, n_iters)
    @printf "Initial loss: %.6e\n" losses[1]
    @printf "Final loss:   %.6e\n" losses[end]
    @printf "True params:    [%.3f, %.3f, %.3f, %.3f]\n" true_params...
    @printf "Initial guess:  [%.3f, %.3f, %.3f, %.3f]\n" initial_guess...
    @printf "Final params:   [%.3f, %.3f, %.3f, %.3f]\n" final_params...

    # Parameter errors
    println("\nParameter errors:")
    for i in 1:4
        init_err = abs(initial_guess[i] - true_params[i])
        final_err = abs(final_params[i] - true_params[i])
        @printf "  p%d: initial=%.4f  final=%.4f  (%.1fx improvement)\n" i init_err final_err init_err/max(final_err, 1e-15)
    end

    # Timing summary
    println("\n=== Timing summary ===")
    grad_median_us = median(b_grad.times) / 1e3
    gd_median_ms = median(b_gd.times) / 1e6
    @printf "Single gradient:     %.1f μs\n" grad_median_us
    @printf "200-iter GD:         %.1f ms\n" gd_median_ms
    @printf "Per-iteration:       %.1f μs\n" (gd_median_ms * 1000 / n_iters)
end

main()
