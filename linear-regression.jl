using CairoMakie
using Statistics

CairoMakie.activate!()

X = [0.2, 1,   1.1, 1.85, 2.3, 2.7,  3.2]
y = [0.7, 0.3, 0.6, 1.3,  0.8, 1.95, 2. ]

X̅ = sum(X) / length(X)
y̅ = sum(y) / length(y)

set_theme!(theme_black())

# Rectangle areas
fig, ax, scat_ = scatter(X, y; color = :cyan)

xlims!(ax, ax.xaxis.attributes.limits[])
ylims!(ax, ax.yaxis.attributes.limits[])

linesegments!([X̅, X̅, -10, 10], [-10, 10, y̅, y̅]; linestyle = :dash, color = :white)

for i in eachindex(X)
    poly!(Point2f[(X[i], y[i]), (X̅, y[i]), (X̅, y̅), (X[i], y̅)]; 
        color = (:cyan, 0.1), strokewidth = 0)
end
fig

# Square areas
fig, ax, scat_ = scatter(X, X; color = :cyan)

xlims!(ax, ax.xaxis.attributes.limits[])
ylims!(ax, ax.yaxis.attributes.limits[])

linesegments!([X̅, X̅], [-10, 10]; linestyle = :dash, color = :white)
linesegments!([-10, 10], [X̅, X̅]; linestyle = :dash, color = :white)
linesegments!([-10, 10], [-10, 10]; color = (:white, 0.6), label = "\$y = x\$")

for i in eachindex(X)
    poly!(Point2f[(X[i], X[i]), (X̅, X[i]), (X̅, X̅), (X[i], X̅)]; 
        color = (:cyan, 0.1), strokewidth = 0)
end

fig

# Linear regression
β = sum((X .- X̅).*(y .- y̅)) / sum((X .- X̅).^2)
sum(X .* y - y̅ .* X) / sum(X.^2 - X̅ .* X)
(sum(X .* y) - sum(β0 .* X)) / sum(X.^2)

β0 = y̅ - β*X̅
(y̅ - X̅ * (sum(X .* y) / sum(X.^2))) / (1 + sum(X) / sum(X.^2))

f(x) = β * x + β0

fig, ax, scat_ = scatter(X, y; color = :cyan)

xlims!(ax, ax.xaxis.attributes.limits[])
ylims!(ax, ax.yaxis.attributes.limits[])

lines!([-100, 100], f.([-100, 100]))
fig

####################################

weight, bias = 0, 0
n = length(X)
f(x) = weight * x + bias
alpha = 0.1

weight_hist = Float32[]
for i in 1:100000
    weight += alpha * 2/n * sum((y .- f.(X)).*X)
    bias += alpha * 2/n * sum(y .- f.(X))
    push!(weight_hist, weight)
end
weight
bias
weight_hist

lines(collect(1:1000), weight_hist)
lines([1, 2, 3], [2, 1, 3])

################################

fig, ax, _ = linesegments([0, 0, -100, 100], [-100, 100, 0, 0]; color = :white)
xlims!(-10, 10)
ylims!(-10, 10)
fig

function line(m, b)
    [-100, 100], [-100, 100] .* m .+ b
end
lines!(line(1, 0)...; label = "m=1, b=0")
lines!(line(0.6, 2)...; label = "m=0.6, b=2")
lines!(line(-2.3, 3)...; label = "m=-2.3, b=3")
lines!(line(15, -20)...; label = "m=15, b=-20")
fig
axislegend()

#################################

fig, _, _ = lines([-100, 100], [-100, 100]; axis = (; limits = (0, 10, 0, 10)), color = :tomato)
linesegments!([3, 3, 7, 7], [3, 5, 7, 9]; color = :white, linestyle = :dash)
scatter!([3, 3, 7, 7], [3, 5, 7, 9]; color = :tomato)
# (3, 5), (7, 9)
5 / 3
9 / 7
lines!([0, 100], [0, 100 * (5 / 3)]; color = (:blue, 0.5))
lines!([0, 100], [0, 100 * (9 / 7)]; color = (:blue, 0.5))

fig

###############################

using LaTeXStrings

fig, ax, _ = linesegments([0, 0, -100, 100], [-100, 100, 0, 0]; color = :white,
    axis = (; limits = (-10, 10, -10, 10), xticks = -10:2:10, yticks = -10:2:10)
)
hidespines!(ax)

linesegments!([-6, -6], [-6, -3]; color = (:white, 0.8), linestyle = :dash, 
    label = L"$y_i - \hat{y}_i = -3 - (-6) = 3$ which is positive..."
)
lines!([-100, 100], [-100, 100]; linewidth = 3, label = L"$m=1$ ...which means that we'd want to increase $m$ from here...")
lines!([-100, 100], 1/2 .* [-100, 100]; color = (:dodgerblue, 0.5), label = L"$m=0.5$ ...but actually we should decrease $m$.")
scatter!([-6], [-3]; color = :dodgerblue, markersize = 15)

axislegend()

fig
