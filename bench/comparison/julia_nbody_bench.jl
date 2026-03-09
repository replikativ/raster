using BenchmarkTools

# N-body simulation from the Computer Language Benchmarks Game
# https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/nbody.html
#
# 5 bodies (Sun + Jupiter + Saturn + Uranus + Neptune)
# Symplectic leapfrog integrator, dt=0.01
# Benchmark: advance N steps, verify energy conservation

const SOLAR_MASS = 4π^2
const DAYS_PER_YEAR = 365.24

mutable struct Body
    x::Float64; y::Float64; z::Float64
    vx::Float64; vy::Float64; vz::Float64
    mass::Float64
end

function make_bodies()
    # Sun
    sun = Body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS)

    # Jupiter
    jupiter = Body(
        4.84143144246472090e+00,
        -1.16032004402742839e+00,
        -1.03622044471123109e-01,
        1.66007664274403694e-03 * DAYS_PER_YEAR,
        7.69901118419740425e-03 * DAYS_PER_YEAR,
        -6.90460016972063023e-05 * DAYS_PER_YEAR,
        9.54791938424326609e-04 * SOLAR_MASS)

    # Saturn
    saturn = Body(
        8.34336671824457987e+00,
        4.12479856412430479e+00,
        -4.03523417114321381e-01,
        -2.76742510726862411e-03 * DAYS_PER_YEAR,
        4.99852801234917238e-03 * DAYS_PER_YEAR,
        2.30417297573763929e-05 * DAYS_PER_YEAR,
        2.85885980666130812e-04 * SOLAR_MASS)

    # Uranus
    uranus = Body(
        1.28943695621391310e+01,
        -1.51111514016986312e+01,
        -2.23307578892655734e-01,
        2.96460137564761618e-03 * DAYS_PER_YEAR,
        2.37847173959480950e-03 * DAYS_PER_YEAR,
        -2.96589568540237556e-05 * DAYS_PER_YEAR,
        4.36624404335156298e-05 * SOLAR_MASS)

    # Neptune
    neptune = Body(
        1.53796971148509165e+01,
        -2.59193146099879641e+01,
        1.79258772950371181e-01,
        2.68067772490389322e-03 * DAYS_PER_YEAR,
        1.62824170038242295e-03 * DAYS_PER_YEAR,
        -9.51592254519715870e-05 * DAYS_PER_YEAR,
        5.15138902046611451e-05 * SOLAR_MASS)

    bodies = [sun, jupiter, saturn, uranus, neptune]

    # Offset momentum: sun's velocity compensates for others
    px, py, pz = 0.0, 0.0, 0.0
    for b in bodies
        px += b.vx * b.mass
        py += b.vy * b.mass
        pz += b.vz * b.mass
    end
    sun.vx = -px / SOLAR_MASS
    sun.vy = -py / SOLAR_MASS
    sun.vz = -pz / SOLAR_MASS

    return bodies
end

function energy(bodies)
    e = 0.0
    n = length(bodies)
    for i in 1:n
        bi = bodies[i]
        e += 0.5 * bi.mass * (bi.vx^2 + bi.vy^2 + bi.vz^2)
        for j in (i+1):n
            bj = bodies[j]
            dx = bi.x - bj.x
            dy = bi.y - bj.y
            dz = bi.z - bj.z
            dist = sqrt(dx^2 + dy^2 + dz^2)
            e -= bi.mass * bj.mass / dist
        end
    end
    return e
end

function advance!(bodies, dt, n_steps)
    n = length(bodies)
    for _ in 1:n_steps
        # Update velocities (all pairs)
        for i in 1:n
            bi = bodies[i]
            for j in (i+1):n
                bj = bodies[j]
                dx = bi.x - bj.x
                dy = bi.y - bj.y
                dz = bi.z - bj.z
                dsq = dx^2 + dy^2 + dz^2
                dist = sqrt(dsq)
                mag = dt / (dsq * dist)
                bi.vx -= dx * bj.mass * mag
                bi.vy -= dy * bj.mass * mag
                bi.vz -= dz * bj.mass * mag
                bj.vx += dx * bi.mass * mag
                bj.vy += dy * bi.mass * mag
                bj.vz += dz * bi.mass * mag
            end
        end
        # Update positions
        for b in bodies
            b.x += dt * b.vx
            b.y += dt * b.vy
            b.z += dt * b.vz
        end
    end
end

# ================================================================
# Benchmark
# ================================================================

println("="^60)
println("N-body (5 planets, Benchmarks Game)")
println("="^60)

# Correctness check
bodies = make_bodies()
println("  Initial energy: ", energy(bodies))
advance!(bodies, 0.01, 1000)
println("  After 1000 steps: ", energy(bodies))

# Warmup
bodies = make_bodies()
advance!(bodies, 0.01, 1000)

# Benchmark: 50M steps (the Benchmarks Game size)
println("\n  Benchmark: 50,000,000 steps")
bodies = make_bodies()
b = @benchmark advance!($bodies, 0.01, 50_000_000) setup=(bodies = make_bodies()) evals=1 samples=5
println("  ", b)

# Also benchmark smaller sizes for comparison
for n in [1_000, 100_000, 1_000_000]
    bodies = make_bodies()
    b = @benchmark advance!($bodies, 0.01, $n) setup=(bodies = make_bodies()) evals=1 samples=10
    med = median(b).time / 1e6  # ms
    println("  $n steps: $(round(med, digits=2)) ms")
end

println()

# ================================================================
# Spectral Norm (Benchmarks Game)
# ================================================================

println("="^60)
println("Spectral Norm (Benchmarks Game)")
println("="^60)

function A(i, j)
    return 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1)
end

function Av!(out, v, n)
    for i in 0:n-1
        s = 0.0
        for j in 0:n-1
            s += A(i, j) * v[j+1]
        end
        out[i+1] = s
    end
end

function Atv!(out, v, n)
    for i in 0:n-1
        s = 0.0
        for j in 0:n-1
            s += A(j, i) * v[j+1]
        end
        out[i+1] = s
    end
end

function AtAv!(out, v, tmp, n)
    Av!(tmp, v, n)
    Atv!(out, tmp, n)
end

function spectral_norm(n)
    u = ones(n)
    v = similar(u)
    tmp = similar(u)
    for _ in 1:10
        AtAv!(v, u, tmp, n)
        AtAv!(u, v, tmp, n)
    end
    vBv = 0.0
    vv = 0.0
    for i in 1:n
        vBv += u[i] * v[i]
        vv += v[i] * v[i]
    end
    return sqrt(vBv / vv)
end

# Correctness
println("  spectral_norm(100) = ", spectral_norm(100))
println("  Expected: 1.274219991...")

# Benchmark
for n in [100, 1000, 5500]
    b = @benchmark spectral_norm($n)
    med = median(b).time / 1e6
    println("  N=$n: $(round(med, digits=3)) ms")
end

println()

# ================================================================
# Rosenbrock gradient (forward-mode AD test)
# ================================================================

println("="^60)
println("Rosenbrock gradient (forward AD)")
println("="^60)

# Rosenbrock: f(x) = sum_{i=1}^{N-1} [100(x_{i+1} - x_i^2)^2 + (1-x_i)^2]
function rosenbrock(x)
    s = 0.0
    for i in 1:length(x)-1
        s += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return s
end

using ForwardDiff

for n in [10, 100, 1000]
    x = randn(n)
    g = similar(x)
    # Warmup
    ForwardDiff.gradient!(g, rosenbrock, x)
    b = @benchmark ForwardDiff.gradient!($g, rosenbrock, $x)
    med = median(b).time / 1e6
    println("  N=$n: $(round(med, digits=4)) ms (gradient)")
end

println()

# ================================================================
# Monte Carlo Pi
# ================================================================

println("="^60)
println("Monte Carlo Pi (10M samples)")
println("="^60)

function monte_carlo_pi(n)
    count = 0
    for _ in 1:n
        x = rand()
        y = rand()
        if x*x + y*y <= 1.0
            count += 1
        end
    end
    return 4.0 * count / n
end

println("  pi ≈ ", monte_carlo_pi(1_000_000))

for n in [1_000_000, 10_000_000]
    b = @benchmark monte_carlo_pi($n)
    med = median(b).time / 1e6
    println("  N=$n: $(round(med, digits=2)) ms")
end

println()
println("All benchmarks complete.")
