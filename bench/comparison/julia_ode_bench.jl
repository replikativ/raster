using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5
using BenchmarkTools

# ================================================================
# Problem definitions (in-place)
# ================================================================

function lorenz!(du, u, p, t)
    x, y, z = u
    sigma = 10.0; rho = 28.0; beta = 8.0/3.0
    du[1] = sigma * (y - x)
    du[2] = x * (rho - z) - y
    du[3] = x * y - beta * z
    nothing
end

function exponential_decay!(du, u, p, t)
    for i in eachindex(u)
        du[i] = -u[i]
    end
    nothing
end

# ================================================================
# Benchmark: Lorenz 3D with DP5
# ================================================================

println("="^60)
println("Lorenz 3D — DP5 adaptive (atol=1e-8, rtol=1e-6)")
println("="^60)

u0_lorenz = [1.0, 1.0, 1.0]
tspan_lorenz = (0.0, 100.0)
prob_lorenz = ODEProblem(lorenz!, u0_lorenz, tspan_lorenz)

# Warmup
sol = solve(prob_lorenz, DP5(), abstol=1e-8, reltol=1e-6,
            save_everystep=false, save_start=false, save_end=true)
println("  Final state: ", sol.u[end])
println("  Stats: nf=", sol.stats.nf, " naccept=", sol.stats.naccept, " nreject=", sol.stats.nreject)

b = @benchmark solve($prob_lorenz, DP5(), abstol=1e-8, reltol=1e-6,
                     save_everystep=false, save_start=false, save_end=true)
println("  Benchmark: ", b)
println()

# ================================================================
# Benchmark: Lorenz 3D with Tsit5
# ================================================================

println("="^60)
println("Lorenz 3D — Tsit5 adaptive (atol=1e-8, rtol=1e-6)")
println("="^60)

sol = solve(prob_lorenz, Tsit5(), abstol=1e-8, reltol=1e-6,
            save_everystep=false, save_start=false, save_end=true)
println("  Final state: ", sol.u[end])
println("  Stats: nf=", sol.stats.nf, " naccept=", sol.stats.naccept, " nreject=", sol.stats.nreject)

b = @benchmark solve($prob_lorenz, Tsit5(), abstol=1e-8, reltol=1e-6,
                     save_everystep=false, save_start=false, save_end=true)
println("  Benchmark: ", b)
println()

# ================================================================
# Benchmark: Lorenz 3D with DP5, save_everystep=true
# ================================================================

println("="^60)
println("Lorenz 3D — DP5 adaptive (atol=1e-8, rtol=1e-6) save_everystep=true")
println("="^60)

sol = solve(prob_lorenz, DP5(), abstol=1e-8, reltol=1e-6,
            save_everystep=true)
println("  Final state: ", sol.u[end])
println("  Num saved: ", length(sol.t))
println("  Stats: nf=", sol.stats.nf, " naccept=", sol.stats.naccept, " nreject=", sol.stats.nreject)

b = @benchmark solve($prob_lorenz, DP5(), abstol=1e-8, reltol=1e-6,
                     save_everystep=true)
println("  Benchmark: ", b)
println()

# ================================================================
# Benchmark: Exponential decay 1D
# ================================================================

println("="^60)
println("Exponential decay 1D — DP5 adaptive (atol=1e-8, rtol=1e-6)")
println("="^60)

u0_decay = [1.0]
tspan_decay = (0.0, 10.0)
prob_decay = ODEProblem(exponential_decay!, u0_decay, tspan_decay)

sol = solve(prob_decay, DP5(), abstol=1e-8, reltol=1e-6,
            save_everystep=false, save_start=false, save_end=true)
println("  Final state: ", sol.u[end])
println("  Exact: ", exp(-10.0))
println("  Error: ", abs(sol.u[end][1] - exp(-10.0)))
println("  Stats: nf=", sol.stats.nf, " naccept=", sol.stats.naccept, " nreject=", sol.stats.nreject)

b = @benchmark solve($prob_decay, DP5(), abstol=1e-8, reltol=1e-6,
                     save_everystep=false, save_start=false, save_end=true)
println("  Benchmark: ", b)
println()

# ================================================================
# Benchmark: 100D linear ODE (array throughput)
# ================================================================

println("="^60)
println("100D linear decay — DP5 adaptive (atol=1e-8, rtol=1e-6)")
println("="^60)

function linear_decay_100!(du, u, p, t)
    for i in eachindex(u)
        du[i] = -Float64(i) * u[i]
    end
    nothing
end

u0_100 = ones(100)
tspan_100 = (0.0, 1.0)
prob_100 = ODEProblem(linear_decay_100!, u0_100, tspan_100)

sol = solve(prob_100, DP5(), abstol=1e-8, reltol=1e-6,
            save_everystep=false, save_start=false, save_end=true)
println("  Final u[1]: ", sol.u[end][1], " exact: ", exp(-1.0))
println("  Final u[100]: ", sol.u[end][100], " exact: ", exp(-100.0))
println("  Stats: nf=", sol.stats.nf, " naccept=", sol.stats.naccept, " nreject=", sol.stats.nreject)

b = @benchmark solve($prob_100, DP5(), abstol=1e-8, reltol=1e-6,
                     save_everystep=false, save_start=false, save_end=true)
println("  Benchmark: ", b)
println()

# ================================================================
# Benchmark: RK4 fixed step on Lorenz (no adaptive overhead)
# ================================================================

println("="^60)
println("Lorenz 3D — RK4 fixed step dt=0.01 (10000 steps)")
println("="^60)

sol = solve(prob_lorenz, RK4(), dt=0.01, adaptive=false,
            save_everystep=false, save_start=false, save_end=true)
println("  Final state: ", sol.u[end])
println("  Stats: nf=", sol.stats.nf)

b = @benchmark solve($prob_lorenz, RK4(), dt=0.01, adaptive=false,
                     save_everystep=false, save_start=false, save_end=true)
println("  Benchmark: ", b)
println()

println("All benchmarks complete.")
