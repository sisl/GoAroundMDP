#module GoAround

#*******************************************************************************
# PACKAGES
#*******************************************************************************
using Random
using Distributions
using LinearAlgebra
using POMDPs
using StaticArrays
using Parameters
using GridInterpolations
using POMDPModelTools
using POMDPModels
using Plots
using LocalFunctionApproximation
using LocalApproximationValueIteration 

#*******************************************************************************
# SETUP
#*******************************************************************************
export
    GoAroundMDP,
    GAState

# Define a struct for the Go Around MDP state
mutable struct GAState
    x::Float64
    v::Float64
    t::Int64
end

# Define a struct for the Go Around MDP
@with_kw struct GoAroundMDP <: MDP{GAState, Symbol}
    pos_lim::Tuple{Float64, Float64}                = (0.0, 720.0)
    vel_lim::Tuple{Float64, Float64}                = (0.0, 54.0)
    discount::Float64                               = 0.95
end

#*******************************************************************************
# MDP FORMULATION
#*******************************************************************************
go_around_actions = [:continue, :go_around]
POMDPs.actions(ga::GoAroundMDP) = go_around_actions
POMDPs.discount(ga::GoAroundMDP) = ga.discount

# Define the actionindex function
function POMDPs.actionindex(ga::GoAroundMDP, a::Symbol)
    if a==:continue
        return 2
    elseif a==:go_around
        return 1
    end
    error("invalid GoAroundMDP action: $a")
end

# Define the generative model
function POMDPs.gen(ga::GoAroundMDP, s::GAState, a::Symbol, rng::AbstractRNG)
    if a==:continue
        v = s.v + rand(Normal(0,1))
        x = s.x - (10/150)*v
        t = s.t - 1
        sp = GAState(x, v, t)
        r = reward(ga, s, a, sp)
        return (sp=sp, r=r)
    else
        x = 360
        v = 0
        t = 0
        sp = GAState(x, v, t)
        r = reward(ga, s, a, sp)
        return (sp=sp, r=r)
    end
end

# Define the reward function
function POMDPs.reward(ga::GoAroundMDP, s::GAState, a::Symbol, sp::GAState)
    rew = 0.0
    if a==:continue
        distance = abs(s.x-360)
        if sp.t==0 && distance < 10
            rew = -1
        else
            rew = 0
        end
    elseif a==:go_around
        rew = -5*(0.8)^s.t - 0.1
    end
    return rew
end

# Define the isterminal function
function POMDPs.isterminal(ga::GoAroundMDP, a::Symbol, s::GAState)
    return (a==:go_around || s.t==0)
end

# Define the initialstate function
function POMDPs.initialstate(ga::GoAroundMDP)
    x = rand(360:720)
    v = rand(0:54)
    t = 47
    sp = GAState(x,v,t)
    r = reward(ga, s, a, sp)
    return (sp=sp, r=r)
end

# Define conversion functions for LocalApproximationValueIteration
function POMDPs.convert_s(::Type{GAState}, v::AbstractVector{Float64}, 
    ga::GoAroundMDP)
    s = GAState(convert(Float64,v[1]), convert(Float64,v[2]), convert(Int64, v[3]))
    return s
end

function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, 
    s::GAState, ga::GoAroundMDP)
    v = SVector{3,Float64}(convert(Float64, s.x), convert(Float64, s.v), convert(Float64, s.t))
    return v
end

#include("runtest.jl")

#end # module

# Instantiate the Go Around MDP and the grid
ga = GoAroundMDP()
nx = 300; ny = 300
x_spacing = range(first(ga.pos_lim), stop=last(ga.pos_lim), length=nx)
y_spacing = range(first(ga.vel_lim), stop=last(ga.vel_lim), length=ny)
grid = RectangleGrid(x_spacing, y_spacing, 0:1:47)

# Solve using LocalApproximationValueIteration
interp = LocalGIFunctionApproximator(grid)
approx_solver = LocalApproximationValueIterationSolver(interp, max_iterations=1, verbose=true, is_mdp_generative=true, n_generative_samples=10)
approx_policy = solve(approx_solver, ga)

# Extract the policy and value function
t_arr = 1:1:48
policy_evolution = zeros(Int8, nx, ny, length(t_arr))
value_function_evolution = zeros(Float64, nx, ny, length(t_arr))
for t in t_arr
    for (i,vel) in enumerate(y_spacing)
        for (j,pos) in enumerate(x_spacing)
            state = GAState(pos,vel,t)
            policy_evolution[i,j,t] = actionindex(ga, action(approx_policy, state))
            value_function_evolution[i,j,t] = value(approx_policy, state)
        end
    end

end

x_ticks = (x_spacing.-360)./240
y_ticks = y_spacing

policy_animation = @animate for i in 1:48
    heatmap(x_ticks, y_ticks, policy_evolution[:,:,i],color=:viridis,
    xlabel="Ground Vehicle Distance from Runway Centerline (miles)", 
    ylabel="Ground Vehicle Velocity (mph)",
    title="Time to Landing: $(48-i) Seconds", colorbar=false)
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), color="#440154", label="Go-Around")
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), color="#FDE725", label="Continue")
end
gif(policy_animation, "policy.gif", fps = 4)

value_function_animation = @animate for i in 1:48
    heatmap(x_ticks, y_ticks, value_function_evolution[:,:,i],color=:viridis,
    xlabel="Ground Vehicle Distance from Runway Centerline (miles)", 
    ylabel="Ground Vehicle Velocity (mph)",
    title="Time to Landing: $(48-i) Seconds")
end
gif(value_function_animation, "value_function.gif", fps = 4)
