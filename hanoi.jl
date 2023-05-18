using Combinatorics

const State = Tuple{Vector{UInt8}, Vector{UInt8}, Vector{UInt8}}

struct Action
    from::UInt8
    to::UInt8
end

struct Node
    state::State
    parent::Union{Node, Nothing}
    action::Union{Action, Nothing}
    path_cost::Float64
end
Node(s::State) = Node(s, nothing, nothing, 0)
Node(s::State, p::Node, a::Action) = Node(s, p, a, p.path_cost + 1)

struct Problem
    initial::State
    goals::Vector{State}
    actions::Function
    result::Function
    actions_cost::Function
end
Problem(initial::State, goals::Vector{State}) = 
    Problem(initial, goals, () -> (), () -> (), () -> ())

function actions(problem::Problem, s::State)

end
function result(s::State, a::Action)::State
    state = deepcopy(s)
    disk = pop!(state[a.from])
    push!(state[a.to], disk)
    state
end
function action_cost(problem::Problem, s::State, a::Action, new_s::State)
end

function isgoal(problem::Problem, s::State)::Bool
    if s ∈ problem.goals
        return true
    else
        return false
    end
end

function expand(node::Node)::Vector{Node}
    stack_tops = [!isempty(node.state[i]) ? node.state[i][end] : Inf for i in 1:3]

    # ([7, 6, 5, 4, 3, 2, 1], [], [])
    # [1, Inf, Inf]

    # stack_tops = [1, Inf, Inf]
    # Combinations of two 2-tuples of (index, value)
    # combs = [(1, 1), (2, Inf)], [(1, 1), (3, Inf)], [(2, Inf), (3, Inf)]
    combs = stack_tops |> enumerate |> collect |> (x -> combinations(x, 2)) |> collect

    actions = [comb[1][2] > comb[2][2] ? 
        Action(comb[2][1], comb[1][1]) : Action(comb[1][1], comb[2][1]) 
        for comb in combs if !isequal(comb[1][2], comb[2][2])
    ]
    
    [Node(result(node.state, action), node, action) for action in actions]
end

function breadth_search(problem)
    global depth
    depth = 1

    node = Node(problem.initial)
    if isgoal(problem, node.state)
        return node
    end

    frontier = [node] # a FIFO queue
    reached = Set([problem.initial])

    depth_num_nodes = 1
    while !isempty(frontier)
        node = pop!(frontier)
        children = expand(node)

        for child in children
            s = child.state
            if isgoal(problem, s)
                return child
            end
            if s ∉ reached
                push!(reached, s)
                pushfirst!(frontier, child)
            end
        end

        depth_num_nodes -= 1
        if depth_num_nodes == 0
            depth += 1
            depth_num_nodes = length(frontier)
        end
    end
    error("Failure to find a solution.")
end

depth = 1
initial = convert(State, ([8, 7, 6, 5, 4, 3, 2, 1], [], []))
goals = convert(Vector{State}, [([], [], [8, 7, 6, 5, 4, 3, 2, 1])])
problem = Problem(initial, goals)

solution = breadth_search(problem)
depth

function solution_state(solution, depth)::Vector{State}
    solution
    nodes = [eval(Meta.parse("solution" * ".parent"^i)) for i in 0:depth]
    [node.state for node in nodes]
end
solution_steps = solution_state(solution, depth)


##### ---------------------- ------------------------ ##### INFORMED SEARCH


using DataStructures

function action_cost(s::State, a::Action, new_s::State)
    return 1
end
function expand(node::Node)::Vector{Node}
    stack_tops = [!isempty(node.state[i]) ? node.state[i][end] : Inf for i in 1:3]
    combs = stack_tops |> enumerate |> collect |> (x -> combinations(x, 2)) |> collect
    actions = [comb[1][2] > comb[2][2] ? 
        Action(comb[2][1], comb[1][1]) : Action(comb[1][1], comb[2][1]) 
        for comb in combs if !isequal(comb[1][2], comb[2][2])
    ]
    
    costs = [node.path_cost + action_cost(node.state, action, result(node.state, action)) for action in actions]
    [Node(result(node.state, action), node, action, cost) for (action, cost) in zip(actions, costs)]
end

getfirst(f::Function, x) = getindex(x, findfirst(f, x))
other(x...) = getfirst(rod -> !(rod ∈ x), [:A, :B, :C])
function perfect_heuristic(disks, num_disks, goal)
    wrongful = false
    moves_left = 0
    for disk in range(num_disks, 1; step = -1)
        current = findfirst(x -> disk ∈ x, disks)
        if (current == goal) && wrongful
            moves_left -= 2^disk - 1
            wrongful = false
        elseif (current != goal) && !wrongful
            moves_left += 2^disk - 1
            wrongful = true
        end
        goal = wrongful ? other(current, goal) : goal
    end
    moves_left
end

h(n::Node) = begin # Number of disks on left peg
    length(n.state[1])
end
h(n::Node) = begin # "No. disks not right" + 2 × ("No. disks right" - "No. biggest disks on right")
    s = n.state
    n_disks = length(vcat(s...))
    n_biggest_right = !isempty(s[3]) && s[3][1] == n_disks ? 
        length([i for i in 2:length(s[3]) if s[3][i-1] == s[3][i]+1])+1 : 0
    length(vcat(s[1:2]...)) + 2*(length(s[3]) - n_biggest_right)
end
h(n::Node) = begin # Perfect heuristic
    disks = Dict([:A, :B, :C] .=> n.state)
    perfect_heuristic(disks, length(vcat(n.state...)), :C)
end

f(n::Node) = n.path_cost# + h(n)
function bfs(problem, f)
    node = Node(problem.initial)
    frontier = PriorityQueue(Base.Order.Forward, node => f(node))
    reached = Dict{State, Node}(problem.initial => node)

    N = 0
    while !isempty(frontier)
        node = dequeue!(frontier)
        if isgoal(problem, node.state)
            return node, N
        end
        for child in expand(node)
            N += 1
            s = child.state
            if (s ∉ keys(reached)) || (child.path_cost < reached[s].path_cost)
                reached[s] = child
                enqueue!(frontier, child => f(child))
            end
        end
    end
    error("Failure to find a solution.")
end

solution, N = bfs(problem, f)
function solution_state(solution)::Vector{State}
    nodes = []; i = 0
    while true
        try
            node = eval(Meta.parse("solution" * ".parent"^i))
            push!(nodes, node)
        catch 
            break 
        end
        i += 1
    end
    [node.state for node in nodes[1:end-1]]
end
solution_state(solution)

# Calculate effective branching factor
d = length(solution_state(solution)) - 1
N
pow = newton(ebf, 1.2, ebfprime)
newton(ebf, 2., ebfprime)
newton(ebf, 2., ebfprime)
newton(ebf, 2., ebfprime)

sum([pow^(n) for n in 0:d])

#=
Heuristic function                                           | Effective Branching Factor | Nodes Expanded
-----------------------------------------------------------------------------------------------------------
No heuristic                                                                   --- 1.159 --- 706
"No. disks left"                                                               --- 1.154 --- 626
"No. disks not right" + 2 × ("No. disks right" - "No. biggest disks on right") --- 1.142 --- 482
Perfect heuristic                                                              --- 1.061 --- 92
=#

# This maths is beyond my head (particularly for "ebf(Bstar)")
# https://stackoverflow.com/questions/71514682/how-to-calculate-effective-branching-factor
# https://mmas.github.io/newton-julia

function newton(f::Function, x0::Number, fprime::Function, args::Tuple=();
    tol::AbstractFloat=1e-8, maxiter::Integer=50, eps::AbstractFloat=1e-10)
    for _ in 1:maxiter
        yprime = fprime(x0, args...)
        if abs(yprime) < eps
            warn("First derivative is zero")
            return x0
        end
        y = f(x0, args...)
        x1 = x0 - y/yprime
        if abs(x1-x0) < tol
            return x1
        end
        x0 = x1
    end
    error("Max iteration exceeded")
end
ebf = Bstar -> (Bstar^(d+1) - 1) / (Bstar - 1) - (N+1)
ebfprime = Bstar -> (-d*Bstar^d - Bstar^d + d*Bstar^(d+1) + 1) / (Bstar - 1)^2

newton(ebf, 2., ebfprime)

##### ---------------------- ------------------------ ##### CREATE VISUALISATIONS OF SEARCH

using Javis
using Colors
using ColorSchemes
using Animations

# Make the hanoi towers

GROUND, GROUND_WIDTH = 50, 400
ROD_TOP_MARGIN, ROD_WIDTH = 10, 10
DISK_MARGIN, MIN_DISK_WIDTH, DISK_HEIGHT = 5, 20, 15
function draw_position(disks::State)
    Luxor.sethue("black")
    setdash("solid")

    disks_flat = vcat(disks...)

    rods_x = [-GROUND_WIDTH/2 + (GROUND_WIDTH*i)/4 for i in 1:3] # [-100, 0, 100]
    max_disk_width = abs(rods_x[1] - rods_x[2]) - DISK_MARGIN

    disk_widths = range(MIN_DISK_WIDTH, max_disk_width; length = maximum(disks_flat)) |> collect
    disk_widths = disk_widths[disks_flat] # order disk widths properly
    disk_rods = vcat([repeat([index], length(list)) for (index, list) in enumerate(disks)]...) # [1, 1, 1, 3, 3]
    disk_rod_heights = vcat([[i for (i, _) in enumerate(list)] for list in disks]...) # [1, 2, 3, 1, 2]

    # Draw ground
    A, B = [Point(x, GROUND) for x in [(max_disk_width/2)-GROUND_WIDTH/2, GROUND_WIDTH/2-(max_disk_width/2)]]
    line(A, B; action = :stroke)
    
    # Draw rods
    rod_height = length(disks_flat)*DISK_HEIGHT + ROD_TOP_MARGIN
    [begin
        [box(Point(rod-(ROD_WIDTH/2), GROUND), Point(rod+(ROD_WIDTH/2), GROUND - rod_height); action = action) 
            for action in [:stroke, :clip]]

        current_linewidth = getline()
        setline(1)

        [rule(Point(0, i*5), -pi/4) for i in -200:50]
        clipreset()

        setline(current_linewidth)
    end for rod in rods_x]
    
    # Draw disks
    color_scheme = ColorSchemes.Spectral
    [begin
        # Ground is centered around 0 with width GROUND_WIDTH
        # i.e. it spans from -GROUND_WIDTH/2 to GROUND_WIDTH/2
        # Starting from the start of the ground, the rods are 1/4, 2/4 and 3/4 along the length respectively
        get_rod_x(j) = -GROUND_WIDTH/2 + (GROUND_WIDTH*j)/4
        offset_x = get_rod_x(disk_rods[i])
        offset_y = GROUND - DISK_HEIGHT * disk_rod_heights[i]

        width = disk_widths[i]

        disk_box_path(action) = box(Point(-width/2 + offset_x, offset_y), Point(width/2 + offset_x, DISK_HEIGHT + offset_y), action)
        Luxor.sethue(color_scheme[disks_flat[i]])
        disk_box_path(:fill)
        Luxor.sethue("black")
        disk_box_path(:stroke)
    end for i in eachindex(disk_widths)]
end
@svg begin
    disks = convert(State, ([5, 4, 3, 2, 1], [], []))
    draw_position(disks)
end

# Split up draw_position function such that parts can be animated

function draw_rods(num_disks::Int)
    Luxor.sethue("black")
    setdash("solid")

    rods_x = [-GROUND_WIDTH/2 + (GROUND_WIDTH*i)/4 for i in 1:3] # [-100, 0, 100]
    max_disk_width = abs(rods_x[1] - rods_x[2]) - DISK_MARGIN

    # Draw ground
    A, B = [Point(x, GROUND) for x in [(max_disk_width/2)-GROUND_WIDTH/2, GROUND_WIDTH/2-(max_disk_width/2)]]
    line(A, B; action = :stroke)

    # Draw rods
    rod_height = num_disks*DISK_HEIGHT + ROD_TOP_MARGIN
    [begin
        [box(Point(rod-(ROD_WIDTH/2), GROUND), Point(rod+(ROD_WIDTH/2), GROUND - rod_height); action = action) 
            for action in [:stroke, :clip]]

        current_linewidth = getline()
        setline(1)

        [rule(Point(0, i*5), -pi/4) for i in -200:50]
        clipreset()

        setline(current_linewidth)
    end for rod in rods_x]
end

function draw_disk(disks::State, disk_id::Int)
    Luxor.sethue("black")
    setdash("solid")

    disks_flat = vcat(disks...)

    rods_x = [-GROUND_WIDTH/2 + (GROUND_WIDTH*i)/4 for i in 1:3] # [-100, 0, 100]
    max_disk_width = abs(rods_x[1] - rods_x[2]) - DISK_MARGIN

    disk_widths = range(MIN_DISK_WIDTH, max_disk_width; length = maximum(disks_flat)) |> collect
    disk_widths = disk_widths[disks_flat] # order disk widths properly

    # Draw disks
    color_scheme = ColorSchemes.Spectral

    i = findfirst(disks_flat .== disk_id)
    width = disk_widths[i]

    disk_box_path(action) = box(Point(-width/2, 0), Point(width/2, DISK_HEIGHT), action)
    Luxor.sethue(color_scheme[disks_flat[i]])
    disk_box_path(:fill)
    Luxor.sethue("black")
    disk_box_path(:stroke)
end

function position_disk(disks::State, disk_id::Union{UInt8, Int})::Tuple{Float64, Float64}
    disks_flat = vcat(disks...)
    disk_rods = vcat([repeat([index], length(list)) for (index, list) in enumerate(disks)]...)
    disk_rod_heights = vcat([[i for (i, _) in enumerate(list)] for list in disks]...)
    get_rod_x(j) = -GROUND_WIDTH/2 + (GROUND_WIDTH*j)/4

    i = findfirst(disks_flat .== disk_id)
    offset_x = get_rod_x(disk_rods[i])
    offset_y = GROUND - DISK_HEIGHT * disk_rod_heights[i]
    return offset_x, offset_y
end

# Animate solution_steps of breadth-first search

FRAMES_PER_MOVE = 20
begin
    solution_vid = Video(500, 500)
    Background(1:700, background_)
    disks = convert(State, ([5, 4, 3, 2, 1], [], []))
    rods = Object((args...) -> draw_rods(5), O)
    disk_objs = [
        Object((args...) -> draw_disk(disks, i), Point(position_disk(disks, i)))
    for i in 1:5]

    prev_step = disks
    disk_obj_poss = [obj.start_pos for obj in disk_objs] # "poss" means "positions"
    for (i, step) in enumerate(reverse(solution_steps)[2:end])
        disk_id = [prev_step[i][end] for i in 1:3 if length(step[i]) < length(prev_step[i])][1]
        
        relative_move = Point(position_disk(step, disk_id)) - disk_obj_poss[disk_id]
        frames = (i-1)*FRAMES_PER_MOVE+1:i*FRAMES_PER_MOVE
        act!(disk_objs[disk_id], Javis.Action(frames, anim_translate(relative_move)))

        prev_step = step
        disk_obj_poss[disk_id] += relative_move
    end
end

render(
    solution_vid;
    pathname="tower.gif",
    framerate=20
)

# Animate first few steps of breadth-first search

#=
Well, it looks like Javis DOES NOT support nested layers...
It's just a pool of Objects at the bottom, with a pool of Layers on top, and the overall video at the top.
How do I do this then?

...create my own layer macro?
A CLayer (Composable Layer) is a data type that you can act!() on, like a normal object.
The CLayer stores the code 
=#

initial = convert(State, ([5, 4, 3, 2, 1], [], []))
goals = convert(Vector{State}, [([], [], [5, 4, 3, 2, 1])])
problem = Problem(initial, goals)

node = Node(problem.initial)

const SearchSteps = Vector{Tuple{Int, Union{State, Nothing}, Vector{State}}}

search_steps = convert(SearchSteps, [(0, nothing, [node.state])])
frontier = [node] # a FIFO queue
reached = Set([problem.initial])
depth = 1

depth_num_nodes = 1
for _ in 1:9
    node = pop!(frontier)
    children = expand(node)
    push!(search_steps, (depth, node.state, [child.state for child in children]))

    for child in children
        s = child.state
        if isgoal(problem, s)
            return child
        end
        # if s ∉ reached
        push!(reached, s)
        pushfirst!(frontier, child)
        # end
    end

    depth_num_nodes -= 1
    if depth_num_nodes == 0
        depth += 1
        depth_num_nodes = length(frontier)
    end
end

# search_steps -> bunches sets of states that should be revealed together in the animation
# 1 -> depth of state/s
# 2 -> parent of state/s
# 3/end -> the state/s themselves
search_steps

function ground(args...)
    background("white")
    Luxor.sethue("black")
end
end_frame = 660
video_width, video_height = 1122, 500
tree_vid = Video(video_width, video_height)
Background(1:end_frame, ground)

states_at_d(depth::Int) = vcat([item[3] for item in search_steps if depth == item[1]]...)

states = vcat(last.(search_steps)...)
num_disks = convert(Int, maximum(vcat(states[1]...)))
positions_disks = Vector{Object}[]
positions = [@JLayer 1:end_frame begin
    rods = Object((args...) -> draw_rods(num_disks), O)
    disk_objs = [
        Object((args...) -> draw_disk(state, i), Point(position_disk(state, i)))
    for i in 1:num_disks]
    push!(positions_disks, disk_objs)
end for state in states]

disks_flat = vcat(problem.initial...)
rods_x = [-GROUND_WIDTH/2 + (GROUND_WIDTH*i)/4 for i in 1:3]
max_disk_width = abs(rods_x[1] - rods_x[2]) - DISK_MARGIN
pos_height = length(disks_flat)*DISK_HEIGHT + ROD_TOP_MARGIN
pos_width = GROUND_WIDTH - max_disk_width

BOTTOM_ROW_GAPS = 25
ROW_MARGINS = 135
OFFSET_X, OFFSET_Y = 0, -200
SCALE = 0.4

#=
    i = 1
    for d_ in 0:depth
        depth_states = states_at_d(d_)
        row_width = (length(depth_states)*(pos_width + BOTTOM_ROW_GAPS) - BOTTOM_ROW_GAPS)
        ypos = (d_*(pos_height + ROW_MARGINS))

        for j in eachindex(depth_states)
            # We want to calculate bottom row positions first, then go up from there
            xpos = (j-1)*(pos_width + BOTTOM_ROW_GAPS) - (row_width - pos_width)/2

            positions[i].position = Point(xpos*SCALE + OFFSET_X, ypos*SCALE + OFFSET_Y)
            act!(positions[i], Javis.Action(1:1, anim_scale(SCALE)))

            i += 1
        end
    end
=#

"Variable dependencies: `pos_width`, `BOTTOM_ROW_GAPS`, `SCALE`, `OFFSET_X`"
function get_x_poss(search_steps::SearchSteps, bottom_filler::Int=0)::Vector{Float64}
    states_at_d(d_::Int) = vcat([item[3] for item in search_steps if d_ == item[1]]...)

    depth = maximum(first.(search_steps))

    # Calculate x positions of bottom row
    num_bottom = length(states_at_d(depth))
    row_width = ((num_bottom+bottom_filler)*(pos_width + BOTTOM_ROW_GAPS) - BOTTOM_ROW_GAPS)

    obj_x_poss = Vector{Float64}[[]]
    for i in 1:num_bottom
        xpos = (i-1)*(pos_width + BOTTOM_ROW_GAPS) - (row_width - pos_width)/2
        xpos = xpos*SCALE + OFFSET_X
        push!(obj_x_poss[1], xpos)
    end

    # Go back up calculating rest of rows by centering nodes on their children
    for d_ in range(depth-1, 0; step=-1)
        states_s_p = [step for step in search_steps if step[1] == d_+1] # states at current depth are parent here
        n_overall_childs, poss = 0, []
        for (i, ps_states) in enumerate(states_s_p)
            n_childs = length(ps_states[3])
            push!(poss, sum(obj_x_poss[1][n_overall_childs+1:n_overall_childs+n_childs]) / n_childs)
            n_overall_childs += n_childs
        end
        for _ in 1:length(states_at_d(d_)) - length(states_s_p) # if a depth has been left unexpanded
            push!(poss, poss[end] + (pos_width + BOTTOM_ROW_GAPS)*SCALE)
        end
    
        pushfirst!(obj_x_poss, poss)
    end
    vcat(obj_x_poss...)
end
get_x_poss(search_steps)

node_index_bundles = UnitRange{Int64}[]
[begin
    start_index = !isempty(node_index_bundles) ? collect.(node_index_bundles)[end][end] : 0
    push!(node_index_bundles, start_index+1:start_index+length(bundle[3]))
end for bundle in search_steps]
node_index_bundles

id_to_depth(id::Int)::Int = 
    vcat([fill(bundle[1], length(bundle[3])) for bundle in search_steps]...)[id]
calc_ypos(d_::Int) = (d_*(pos_height + ROW_MARGINS)) * SCALE + OFFSET_Y
get_parent_id(id::Int)::Int = begin
    d_ = id_to_depth(id)
    ss_id = vcat([fill(i, length(bundle[3])) for (i, bundle) in enumerate(search_steps)]...)[id]
    depth_bundle_num = sum(first.(search_steps)[1:ss_id-1] .== d_) + 1
    reduce(+, [length(states_at_d(i)) for i in 0:d_-2]; init=0) + depth_bundle_num
end
get_children_ids(id::Int)::Vector{Int} = begin
    d_ = id_to_depth(id)
    depth_id = id - reduce(+, [length(bundle[3]) for bundle in search_steps if bundle[1] < d_]; init=0)
    child_depth_bundles = filter(x -> x[1] == d_+1, search_steps)
    if isempty(child_depth_bundles)
        return []
    elseif length(child_depth_bundles) < depth_id
        return []
    end
    num_sofar = sum([length(states_at_d(i)) for i in 0:d_]) + reduce(+, length.(last.(child_depth_bundles[1:depth_id-1])); init=0)
    [i+num_sofar for i in eachindex(child_depth_bundles[depth_id][3])]
end
get_bottom_filler(search_steps_::SearchSteps)::Int = begin
    d_ = search_steps_[end][1]
    depth_done = reduce(+, length.(last.(filter(x -> x[1] == d_, search_steps_))); init=0)
    length(states_at_d(d_)) - depth_done
end
id_to_bundleid(id::Int) = begin

end

EXPAND_STEP_TIME = 60
NODE_APPEAR_TIME = 10
DISK_MOVE_TIME = 40
DISK_MOVE_OVERLAP = 20
ADJUST_POSITIONS_TIME = 19

bounding_angle = atan(pos_height + 36, pos_width)

function draw_arrow(loc, ploc)
    arrow_angle = atan(abs(ploc.y - loc.y), abs(ploc.x - loc.x))
    xoffset = SCALE * pos_width/2
    yoffset = SCALE * (pos_height/2 + 18)

    if bounding_angle > arrow_angle
        yoffset = tan(arrow_angle)*xoffset
    else
        xoffset = tan(π/2 - arrow_angle)*yoffset
    end
    if ploc.x - loc.x > 0
        new_loc = Point(loc.x + xoffset, loc.y + yoffset)
        new_ploc = Point(ploc.x - xoffset, ploc.y - yoffset)
    else
        new_loc = Point(loc.x - xoffset, loc.y + yoffset)
        new_ploc = Point(ploc.x + xoffset, ploc.y - yoffset)
    end
    arrow(new_loc, new_ploc)
end

[act!(pos, Javis.Action(1:1, disappear(:fade))) for pos in positions]
[act!(pos, Javis.Action(1:1, anim_scale(SCALE))) for pos in positions]

arrows = Dict{Int, Object}()
for (i, bundle_id) in enumerate(node_index_bundles)
    fstart = (i-1)*EXPAND_STEP_TIME + 1

    xposs = get_x_poss(search_steps[1:i], get_bottom_filler(search_steps[1:i]))
    for j in bundle_id
        act!(positions[j], Javis.Action(fstart:fstart+NODE_APPEAR_TIME, appear(:fade)))
        xpos, ypos = xposs[j], calc_ypos(search_steps[i][1])
        positions[j].position = Point(xpos, ypos)
        # Visualize action by moving disk
        if search_steps[i][2] !== nothing
            step, prev_step = states[j], search_steps[i][2]
            disk_id = [prev_step[rod][end] for rod in 1:3 if length(step[rod]) < length(prev_step[rod])][1]
            disk_prev_pos = Point(position_disk(prev_step, disk_id))
            disk_move = Point(position_disk(step, disk_id)) - disk_prev_pos

            disks = positions_disks[j]
            disks[disk_id].start_pos = Point(position_disk(prev_step, disk_id))

            num_states_before = reduce(+, length.(last.(search_steps[1:i-1])); init=0)
            # 40 frames to complete all moves, pretend there's 20 more frames and overlap accordingly
            movetime = convert(Int, floor((DISK_MOVE_TIME+DISK_MOVE_OVERLAP)/length(search_steps[i][3])))
            # TODO: Divide by 0 error if there's only 1 successor
            starttime = convert(Int, floor(DISK_MOVE_OVERLAP / (length(search_steps[i][3]) - 1)))
            
            disk_frames_start = (j - num_states_before -1)*starttime + 1
            disk_frames_end = disk_frames_start + movetime - 1
            disk_frames = fstart+NODE_APPEAR_TIME+disk_frames_start:fstart+NODE_APPEAR_TIME+disk_frames_end
            disk_anim = Animation(
                [0, 1],
                [O, disk_move],
                [expout(8)]
            )
            act!(disks[disk_id], Javis.Action(disk_frames, disk_anim, translate()))
        end
        # Create arrow
        if search_steps[i][2] !== nothing
            pos_, ppos = positions[j].position, positions[get_parent_id(j)].position
            arrow_ = Object(1:end_frame, (args...; x1=ppos.x, y1=ppos.y, x2=pos_.x, y2=pos_.y) ->
                draw_arrow(Point(x1, y1), Point(x2, y2))
            )
            act!(arrow_, Javis.Action(1:1, disappear(:fade)))
            act!(arrow_, Javis.Action(fstart:fstart+NODE_APPEAR_TIME, appear(:fade)))
            arrows[j] = arrow_
        end
    end
end

current_xposs = get_x_poss(search_steps[1:1])
for (i, bundle_id) in enumerate(node_index_bundles)
    fstart = (i-1)*EXPAND_STEP_TIME + 1

    for j in bundle_id
        positions[j].position += Point(OFFSET_X, 0)
        push!(current_xposs, positions[j].position.x)
    end
    OFFSET_X = 0
    xposs = get_x_poss(search_steps[1:i], get_bottom_filler(search_steps[1:i]))
    current_xstart, current_xend = xposs[end-length(search_steps[i][3])+1], xposs[end]
    if current_xstart < -video_width/2
        OFFSET_X = -(video_width/2 + (current_xstart - SCALE*pos_width/2 - 20))
    elseif current_xend > video_width/2
        OFFSET_X = -((current_xend + SCALE*pos_width/2 + 20) - video_width/2)
    end
    xposs = get_x_poss(search_steps[1:i], get_bottom_filler(search_steps[1:i]))

    for j_ in 1:bundle_id.stop
        xdir = xposs[j_] - current_xposs[j_]
        state_translate_anim = Animation(
            [0, 1],
            [O, Point(xdir, 0)],
            [sineio()]
        )
        act!(positions[j_], Javis.Action(fstart:fstart+ADJUST_POSITIONS_TIME, state_translate_anim, translate()))
        
        arrow_move_anim = Animation(
            [0, 1],
            [current_xposs[j_], xposs[j_]],
            [sineio()]
        )
        if haskey(arrows, j_)
            act!(arrows[j_], Javis.Action(fstart:fstart+ADJUST_POSITIONS_TIME, arrow_move_anim, change(:x2)))
        end
        for child_id in get_children_ids(j_)
            act!(arrows[child_id], Javis.Action(fstart:fstart+ADJUST_POSITIONS_TIME, arrow_move_anim, change(:x1)))
        end
    end
    current_xposs = copy(xposs)
end



render(
    tree_vid;
    pathname="tree4.gif",
    framerate=20
)

# ============--------------- Learning Javis & Luxor ------------------===============

function path!(points, pos, color)
    Luxor.sethue(color)
    push!(points, pos) # add pos to points
    circle.(points, 2, :fill) # draws a circle for each point using broadcasting
end

function dancing_circles(c1, c2, start_pos = O)
    global red_ball
    path_of_red = Point[]
    path_of_blue = Point[]

    red_ball = Object(JCircle(O, 25, color = c1, action = :fill), start_pos + (100, 0))
    act!(red_ball, Javis.Action(anim_rotate_around(2π, start_pos)))
    blue_ball = Object(JCircle(O, 25, color = c2, action = :fill), start_pos + (200, 80))
    act!(blue_ball, Javis.Action(anim_rotate_around(2π, 0.0, red_ball)))
    Object(@JShape begin
        path!(path_of_red, pos(red_ball), c1)
    end)
    Object(@JShape begin
        path!(path_of_blue, pos(blue_ball), c2)
    end)
end

myvideo = Video(500, 500)
Background(1:140, ground)

l1 = @JLayer 1:140 begin
    dancing_circles("green", "orange")
end
Javis.Layer
l1.layer_objects
Javis.Object
# l1.position = Point(-100, -200)
Javis.get
act!(l1, Javis.Action(1:1, anim_scale(0.4)))

animation_point = Point(-75, -75)
anim_back_and_forth = Animation(
		[0, 1/2, 1],
		[animation_point, -animation_point, animation_point],
		[sineio(), sineio()]
	)
	
act!(l1, Javis.Action(1:140, anim_back_and_forth, translate()))

test_arrow = Object(1:140, (args...) -> begin
    arrow(Point(-250, 250), pos(red_ball))
end, O)

render(myvideo; pathname="dancing_circles_layer.gif")

#########################

function ground(args...)
    background("white")
    Luxor.sethue("black")
end
