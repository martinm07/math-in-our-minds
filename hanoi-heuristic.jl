other(x...) = [:A, :B, :C][findfirst(x_ -> !in(x_, x), [:A, :B, :C])]
moves_left = 0

# 127 steps
disks = Dict(:A => Int[], :B => [7], :C => [6, 5, 4, 3, 2, 1])
total_disks = 7

current_disk = 7
current = findfirst(x -> current_disk ∈ x, disks)
goal = :C
if current == goal
    # goal = other([current, goal])
    # !goal stays the same
else
    moves_left += 2^current_disk - 1
    goal = other([current, goal])
end

current_disk -= 1
old_current = current
current = findfirst(x -> current_disk ∈ x, disks)

if current == goal
else
    if current != old_current
        # current isn't goal, and current changed... sign we're not on optimal path?
    end
end

##################################

getfirst(f::Function, x) = getindex(x, findfirst(f, x))


other(x...) = getfirst(rod -> !(rod ∈ x), [:A, :B, :C])
begin
    disks = Dict(:A => Int[6, 5], :B => [3, 2], :C => [4, 1])
    num_disks = 6
    goal = :C

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
2^num_disks - 1 - moves_left

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

disks = Dict(:A => Int[1], :B => [], :C => [8, 7, 6, 5, 4, 3, 2])
perfect_heuristic(disks, 8, :C)
