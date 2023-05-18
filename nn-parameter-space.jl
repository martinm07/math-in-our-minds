using GLMakie
using Distributions
using Statistics

w1, w2 = 0.1, 0.1
function predict(X; return_steps=false, w1_ = nothing, w2_ = nothing)
    c_w1 = w1_ !== nothing ? w1_ : w1
    c_w2 = w2_ !== nothing ? w2_ : w2

    h1 = c_w1.*X
    h2 = c_w2.*h1
    if return_steps
        h1, h2
    else
        h2
    end
end

X = collect(-5:0.1:5)
y = X.^2 .+ X.*1.4 .- 9 .+ rand(Normal(0, 7), length(X))

scatter(X, y)
lines!(X, predict(X))

function cost(y, ŷ)
    (1/length(y)) * sum(abs2, ŷ .- y)
end
cost(y, predict(X))

function ∂w2(X, y)
    h1, h2 = predict(X; return_steps = true)
    (1/length(y)) * sum(2 .* (h2 .- y) .* h1)
end
function ∂w1(X, y)
    h1, h2 = predict(X; return_steps = true)
    (1/length(y)) * sum(2 .* (h2 .- y) .* w2 .* X)
end
∂w2(X, y)

learning_rate = 0.0005

cost_hist = Float64[cost(y, predict(X))]
w1_hist, w2_hist = [w1], [w2]
for i in 1:500
    global w1, w2
    new_w2 = w2 - learning_rate*∂w2(X, y)
    new_w1 = w1 - learning_rate*∂w1(X, y)
    w1, w2 = new_w1, new_w2
    push!(cost_hist, cost(y, predict(X)))
    push!(w1_hist, w1); push!(w2_hist, w2)
end
cost_hist[end]
lines(0.:500., cost_hist)

# construct error surface
w1s = collect(-3:0.01:3)
w2s = collect(-3:0.01:3)
errors = [cost(y, predict(X, w1_=w1, w2_=w2)) for w1 in w1s, w2 in w2s]

surface(w1s, w2s, errors, axis=(type=Axis3, xlabel="w1", ylabel="w2", zlabel="Cost", title="Error Surface of Linear NN with 1 Hidden Layer"), visible = false)
contour3d!(w1s, w2s, errors; overdraw = true, levels = 200)
# Plot path we took
scatter!(w1_hist, w2_hist, [cost(y, predict(X, w1_=w1, w2_=w2))+10 for (w1, w2) in zip(w1_hist, w2_hist)], markersize=5)
scatter!([w1], [w2], [cost(y, predict(X))+10], markersize=15, color=:red)

########################### TRUE MULTILAYER PERCEPTRON

# 2 input neurons, 4 hidden neurons, 1 output neuron

X = transpose([0 0; 1 0; 0 1; 1 1])
y = [0 1 1 0;]
X

W1 = randn((4, 2))
b1 = randn(4)
W2 = randn((1, 4))
b2 = rand(1)

function predict(X; return_steps = false)
    h1 = W1*X .+ b1
    a1 = convert(Matrix{Float64}, h1 .> 0)
    h2 = W2*a1 .+ b2
    a2 = convert(Matrix{Float64}, h2 .> 0)
    if return_steps
        h1, a1, h2, a2
    else
        a2
    end
end

predict(X)

ϵ = 0.1
function ∂W2(X, y) # (1, 4)
    h1, a1, h2, a2 = predict(X, return_steps = true)
    sign_ = [(y_ ≤ 0 && a2_ > 0) ? -ϵ : ((y_ > 0 && a2_ ≤ 0) ? ϵ : 0) for (y_, a2_) in zip(y, a2)]

    transpose(mean(repeat(sign_, 4) .* a1, dims=2))
end
function ∂b2(X, y) # (1,)
    h1, a1, h2, a2 = predict(X, return_steps = true)
    sign_ = [(y_ ≤ 0 && a2_ > 0) ? -ϵ : ((y_ > 0 && a2_ ≤ 0) ? ϵ : 0) for (y_, a2_) in zip(y, a2)]

    mean(sign_, dims = 2)[:, 1]
end

function ∂W1(X, y) # (4, 2)
    h1, a1, h2, a2 = predict(X, return_steps = true)
    sign_ = [(y_ ≤ 0 && a2_ > 0) ? -ϵ : ((y_ > 0 && a2_ ≤ 0) ? ϵ : 0) for (y_, a2_) in zip(y, a2)]

    dw1s = [mean(repeat(sign_, 2) .* X .* W2[1, i], dims = 2) for i in axes(W1, 1)]
    transpose(hcat(dw1s...))
end
function ∂b1(X, y) # (4,)
    h1, a1, h2, a2 = predict(X, return_steps = true)
    sign_ = [(y_ ≤ 0 && a2_ > 0) ? -ϵ : ((y_ > 0 && a2_ ≤ 0) ? ϵ : 0) for (y_, a2_) in zip(y, a2)]

    mean(repeat(transpose(W2), 1, size(X)[2]) .* repeat(sign_, 4), dims=2)[:, 1]
end
# Batch gradient descent
for i in 1:20000
    global W1, b1, W2, b2
    new_W1 = W1 + ∂W1(X, y)
    new_b1 = b1 + ∂b1(X, y)
    new_W2 = W2 + ∂W2(X, y)
    new_b2 = b2 + ∂b2(X, y)
    W1, b1, W2, b2 = new_W1, new_b1, new_W2, new_b2
end
predict(X)
