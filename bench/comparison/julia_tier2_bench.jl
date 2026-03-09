using BenchmarkTools
using LinearAlgebra

# ================================================================
# Tier 2 Benchmarks
# ================================================================

println("="^60)
println("SOR / Jacobi-2D Relaxation")
println("="^60)

function jacobi_2d!(u, tmp, n, niter)
    for _ in 1:niter
        for j in 2:n-1
            for i in 2:n-1
                tmp[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
            end
        end
        for j in 2:n-1
            for i in 2:n-1
                u[i,j] = tmp[i,j]
            end
        end
    end
    return u
end

for (n, niter) in [(100, 100), (256, 100), (512, 50)]
    u = zeros(n, n)
    # Boundary: top row = 1.0
    u[1, :] .= 1.0
    tmp = copy(u)
    # Warmup
    jacobi_2d!(copy(u), copy(tmp), n, 1)
    b = @benchmark jacobi_2d!(u2, t2, $n, $niter) setup=(u2=copy($u); t2=copy($tmp))
    med = median(b).time / 1e6
    println("  N=$n, iter=$niter: $(round(med, digits=2)) ms")
end

println()

# ================================================================
# Lotka-Volterra ODE (non-stiff)
# ================================================================

println("="^60)
println("Lotka-Volterra ODE (RK4 fixed-step)")
println("="^60)

function lotka_volterra!(du, u, p, t)
    a, b, c, d = p
    du[1] = a*u[1] - b*u[1]*u[2]
    du[2] = -c*u[2] + d*u[1]*u[2]
    nothing
end

function rk4_step!(f!, u, du, k2, k3, k4, tmp, p, t, dt)
    n = length(u)
    f!(du, u, p, t)
    for i in 1:n; tmp[i] = u[i] + 0.5*dt*du[i]; end
    f!(k2, tmp, p, t + 0.5*dt)
    for i in 1:n; tmp[i] = u[i] + 0.5*dt*k2[i]; end
    f!(k3, tmp, p, t + 0.5*dt)
    for i in 1:n; tmp[i] = u[i] + dt*k3[i]; end
    f!(k4, tmp, p, t + dt)
    for i in 1:n
        u[i] += (dt/6.0)*(du[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
    end
end

function solve_rk4(f!, u0, p, tspan, dt)
    u = copy(u0)
    t = tspan[1]
    tf = tspan[2]
    n = length(u0)
    du = similar(u); k2 = similar(u); k3 = similar(u); k4 = similar(u); tmp = similar(u)
    while t < tf
        dt_actual = min(dt, tf - t)
        rk4_step!(f!, u, du, k2, k3, k4, tmp, p, t, dt_actual)
        t += dt_actual
    end
    return u
end

p = (1.5, 1.0, 3.0, 1.0)
u0 = [1.0, 1.0]

# Correctness
result = solve_rk4(lotka_volterra!, u0, p, (0.0, 10.0), 0.01)
println("  u(10) = ", result)

# Benchmark
b = @benchmark solve_rk4(lotka_volterra!, $u0, $p, (0.0, 10.0), 0.01)
med = median(b).time / 1e6
println("  RK4 dt=0.01, tspan=[0,10]: $(round(med, digits=3)) ms")

b = @benchmark solve_rk4(lotka_volterra!, $u0, $p, (0.0, 10.0), 0.001)
med = median(b).time / 1e6
println("  RK4 dt=0.001, tspan=[0,10]: $(round(med, digits=3)) ms")

println()

# ================================================================
# FFT (Radix-2 Cooley-Tukey)
# ================================================================

println("="^60)
println("FFT (Radix-2 Cooley-Tukey)")
println("="^60)

function fft_radix2!(x::Vector{ComplexF64})
    n = length(x)
    if n <= 1; return x; end

    # Bit-reversal permutation
    j = 0
    for i in 0:n-2
        if i < j
            x[i+1], x[j+1] = x[j+1], x[i+1]
        end
        m = n >> 1
        while m >= 1 && j >= m
            j -= m
            m >>= 1
        end
        j += m
    end

    # Butterfly
    len = 2
    while len <= n
        half = len >> 1
        w = exp(-2π*im/len)
        for start in 0:len:n-1
            wk = ComplexF64(1.0, 0.0)
            for k in 0:half-1
                u = x[start + k + 1]
                v = wk * x[start + k + half + 1]
                x[start + k + 1] = u + v
                x[start + k + half + 1] = u - v
                wk *= w
            end
        end
        len <<= 1
    end
    return x
end

for logn in [10, 14, 20]
    n = 1 << logn
    x = randn(ComplexF64, n)
    b = @benchmark fft_radix2!(y) setup=(y=copy($x))
    med = median(b).time / 1e6
    println("  N=2^$logn ($n): $(round(med, digits=3)) ms")
end

println()

# ================================================================
# DGEMM (naive matmul — no BLAS)
# ================================================================

println("="^60)
println("DGEMM (naive matmul, no BLAS)")
println("="^60)

function naive_matmul!(C, A, B, n)
    for i in 1:n
        for j in 1:n
            s = 0.0
            for k in 1:n
                s += A[i, k] * B[k, j]
            end
            C[i, j] = s
        end
    end
end

for n in [100, 500, 1000]
    A = randn(n, n)
    B = randn(n, n)
    C = zeros(n, n)
    b = @benchmark naive_matmul!($C, $A, $B, $n)
    med = median(b).time / 1e6
    println("  N=$n: $(round(med, digits=3)) ms")
end

println()

# ================================================================
# DGEMM via BLAS
# ================================================================

println("="^60)
println("DGEMM (dense matmul via BLAS)")
println("="^60)

for n in [100, 500, 1000]
    A = randn(n, n)
    B = randn(n, n)
    b = @benchmark ($A * $B)
    med = median(b).time / 1e6
    println("  N=$n: $(round(med, digits=3)) ms")
end

println()
println("All Tier 2 benchmarks complete.")
