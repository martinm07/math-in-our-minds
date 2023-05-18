# 'The Math In Our Minds' Independent project
Hosted at: https://legoboy7.pythonanywhere.com/independent_project

I used the Julia programming language to ultimately create figures and animations for this project.
- `explain-maths.jl` creates two figures. One explaining linear functions, and the other, derivatives.
- `hanoi.jl` creates the Tower of Hanoi problem and a search algorithm for solutions, both breadth-first search and A* with a few different heuristics. It then creates two animations, one showing the full solution and the other showing the first few steps of (breadth-first) search.
- `hanoi-heuristic.jl` implements the "perfect heuristic" for Tower of Hanoi, which exactly determines the minimum number of moves (to the goal) from ANY position.
- `icecream.jl` first wrangles the data (found in "data" folder), then plots catplots of the features by month, then a scatterplot, then performs a linear regression which is added to the scatterplot. It also plots distributions.
- `linear-regression.jl` creates some (unused) plots for understanding the analytical solution to linear regression. It also implements the gradient descent approach.
- `nn-parameter-space.jl` creates and trains a basic neural network for demonstrating redundancy in extra layers with linear activation. It then implements my training solution for a "true" multilayer perceptron (i.e. using heaviside step function activation) that's just a logical extension of Rosenblatt's perceptron learning rule. It successfully solves the XOR problem.

The "figures" folder contains the figures generated from all these files, but also two figures I created myself using GIMP. \
If you're looking to run these Julia files, then note they probably won't run as is&mdash; they were executed in a dynamic order using VSCode and the Julia extension.
