using LaTeXStrings
using CairoMakie

############### DERIVATIVES #################

f(x) = -3*exp(-x-3) + 3 - 3*exp(-x^2)
df(x) = 6*exp(-x^2)*x + exp(-x-3)
function tangent_approx(x, a, h)
    slope = (f(a + h) - f(a))/h
    slope*(x - a) + f(a)
end

xs = collect(-5:0.05:5)
ys = f.(xs)

fig = Figure();
ax = Axis(fig[1, 1]; aspect = 1, xticks = WilkinsonTicks(8))
xlims!(-4, 4)
ylims!(-3, 4)
linesegments!([0, 0, -100, 100], [-100, 100, 0, 0]; color = :black)

a, h = -0.75, 1.25
lines!(-5:0.05:5, x -> tangent_approx(x, a, h), color = :black)
lines!(-5:0.05:5, f, label = L"f(x)");

scatter!([a], [f(a)]; color = :dodgerblue)
scatter!([a+h], [f(a+h)]; color = :black)
linesegments!([a, a+h], [f(a+h), f(a+h)]; linestyle = :dash, color = :grey35)
text!(h/2+a, f(a+h); text = L"h", align = (:center, :top))

axislegend()
fig

########### SIGMOID ###############

fig, ax, plot = lines(-10:0.05:10, x -> 1/(1+exp(-x)), label=L"\frac{1}{1+e^{-x}}", color = :dodgerblue, axis=
    (; xticks = -5:5, title = "Sigmoid Function as Compromise Between Linear Activation and Heaviside Step Function"));
xlims!(-5, 5); ylims!(-0.5, 1.5)
linesegments!([0, 0, -100, 100], [-100, 100, 0, 0]; color = :black)
lines!(-10:0.05:10, x -> x*1/4 + 0.5; color = :grey35, linestyle = :dash)
lines!(-10:0.05:10, x -> x > 0.005 ? 1 : 0.01; color = :grey35, linestyle = :dash)
lines!(-10:0.05:10, x -> 1/(1+exp(-x)); color = :dodgerblue, linewidth = 2)

axislegend(; labelsize = 40)
fig
