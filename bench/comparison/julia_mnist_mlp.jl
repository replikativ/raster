# bench/julia_mnist_mlp.jl
# Fair comparison with Raster MNIST: 784-128-10 MLP, single-sample SGD
#
# Run: OPENBLAS_NUM_THREADS=1 julia --project=../Flux.jl bench/julia_mnist_mlp.jl
# Or without local Flux: OPENBLAS_NUM_THREADS=1 julia bench/julia_mnist_mlp.jl

using Flux, MLDatasets, Statistics, Printf, Random

println("=== Flux.jl MNIST MLP Benchmark ===")
println("Julia $(VERSION), Flux $(pkgversion(Flux))")
println("OPENBLAS_NUM_THREADS = $(get(ENV, "OPENBLAS_NUM_THREADS", "not set"))")
println()

# Load MNIST (old-style MLDatasets API)
println("Loading MNIST...")
train_x_raw, train_y_int = MLDatasets.MNIST.traindata()
test_x_raw, test_y_int = MLDatasets.MNIST.testdata()

train_x = reshape(Float64.(train_x_raw), 784, :)
train_y = Flux.onehotbatch(train_y_int, 0:9) .|> Float64
test_x = reshape(Float64.(test_x_raw), 784, :)

n_train = size(train_x, 2)
n_test = size(test_x, 2)
println("Train: $n_train, Test: $n_test")

# ================================================================
# Benchmark 1: Single-sample SGD (matches Raster exactly)
# Architecture: 784-128-10, SGD lr=0.01, Float64
# ================================================================

println("\n--- Benchmark 1: Single-sample SGD (Float64) ---")
println("Architecture: 784-128-10, SGD lr=0.01")

Random.seed!(42)
model_f64 = Chain(
    Dense(784 => 128, relu),
    Dense(128 => 10),
    softmax
) |> f64

opt_f64 = Flux.setup(Descent(0.01), model_f64)

for epoch in 1:5
    indices = shuffle(1:n_train)
    t0 = time()
    for i in indices
        x_i = train_x[:, i:i]
        y_i = train_y[:, i:i]
        grads = Flux.gradient(model_f64) do m
            Flux.crossentropy(m(x_i), y_i)
        end
        Flux.update!(opt_f64, model_f64, grads[1])
    end
    dt = time() - t0
    preds = Flux.onecold(model_f64(test_x)) .- 1
    acc = mean(preds .== test_y_int) * 100.0
    @printf("Epoch %d/5  acc=%.2f%%  time=%.1fs\n", epoch, acc, dt)
end

# ================================================================
# Benchmark 2: Single-sample SGD (Float32)
# ================================================================

println("\n--- Benchmark 2: Single-sample SGD (Float32) ---")
println("Architecture: 784-128-10, SGD lr=0.01")

train_x32 = Float32.(train_x)
train_y32 = Float32.(train_y)
test_x32 = Float32.(test_x)

Random.seed!(42)
model_f32 = Chain(
    Dense(784 => 128, relu),
    Dense(128 => 10),
    softmax
) |> f32

opt_f32 = Flux.setup(Descent(0.01f0), model_f32)

for epoch in 1:5
    indices = shuffle(1:n_train)
    t0 = time()
    for i in indices
        x_i = train_x32[:, i:i]
        y_i = train_y32[:, i:i]
        grads = Flux.gradient(model_f32) do m
            Flux.crossentropy(m(x_i), y_i)
        end
        Flux.update!(opt_f32, model_f32, grads[1])
    end
    dt = time() - t0
    preds = Flux.onecold(model_f32(test_x32)) .- 1
    acc = mean(preds .== test_y_int) * 100.0
    @printf("Epoch %d/5  acc=%.2f%%  time=%.1fs\n", epoch, acc, dt)
end

# ================================================================
# Benchmark 3: Batched training (more typical Flux usage)
# ================================================================

println("\n--- Benchmark 3: Batched SGD (Float32, batch=128) ---")
println("Architecture: 784-128-10, SGD lr=0.01, batch=128")

Random.seed!(42)
model_batch = Chain(
    Dense(784 => 128, relu),
    Dense(128 => 10),
    softmax
) |> f32

opt_batch = Flux.setup(Descent(0.01f0), model_batch)
data_loader = Flux.DataLoader((train_x32, train_y32), batchsize=128, shuffle=true)

for epoch in 1:5
    t0 = time()
    for (x_batch, y_batch) in data_loader
        grads = Flux.gradient(model_batch) do m
            Flux.crossentropy(m(x_batch), y_batch)
        end
        Flux.update!(opt_batch, model_batch, grads[1])
    end
    dt = time() - t0
    preds = Flux.onecold(model_batch(test_x32)) .- 1
    acc = mean(preds .== test_y_int) * 100.0
    @printf("Epoch %d/5  acc=%.2f%%  time=%.1fs\n", epoch, acc, dt)
end

println("\nDone.")
