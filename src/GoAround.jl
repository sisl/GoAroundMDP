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
#plotly()

export
    GoAroundMDP,
    GAState

mutable struct GAState
    x::Float64
    v::Float64
    t::Int64
end

@with_kw struct GoAroundMDP <: MDP{GAState, Symbol}
    xlim::Tuple{Float64, Float64}                   = (0.0, 50.0)
    ylim::Tuple{Float64, Float64}                   = (0.0, 50.0)
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
        return 1
    elseif a==:go_around
        return 2
    end
    error("invalid GoAroundMDP action: $a")
end


function POMDPs.gen(ga::GoAroundMDP, s::GAState, a::Symbol, rng::AbstractRNG)
    if a==:continue
        x = s.x + s.v
        v = s.v + rand(Normal(0,1))
        t = s.t - 1
        sp = GAState(x,v,t)
        r = -1
        return (sp=sp, r=r)
    else
        x = 0
        v = 0
        t = 0
        r = -2
        sp = GAState(x,v,t)
        return (sp=sp, r=r)
    end
end
# Define the reward function
function POMDPs.reward(ga::GoAroundMDP, s::GAState, a::Symbol, sp::GAState)
    rew = 0.0
    if a==:continue
        if s.t==1 && s.x==40
            rew = -2
        else
            rew = 0
        end
    elseif a==:go_around
        rew = -10*(0.2)^s.t - 0.5
    end
    return rew
end

# Define the isterminal function
function POMDPs.isterminal(ga::GoAroundMDP, a::Symbol, s::GAState)
    if a==:go_around || s.t==0 
        return true 
    end
    return false
end

# Define the initialstate function
function POMDPs.initialstate(ga::GoAroundMDP)
    x = 0
    v = 1
    t = 47
    sp = GAState(x,v,t)
    return (sp=sp)
end

# Define conversion functions for LocalApproximationValueIteration
function POMDPs.convert_s(::Type{GAState}, v::AbstractVector{Float64}, 
    ga::GoAroundMDP)
    s = GAState(convert(Float64,v[1]),convert(Float64,v[2]),convert(Int64, v[3]))
    return s
end

function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, 
    s::GAState, ga::GoAroundMDP)
    v = SVector{3,Float64}(convert(Float64, s.x), convert(Float64, s.v), convert(Float64, s.t))
    return v
end



#include("runtest.jl")

#end # module


ga = GoAroundMDP()
nx = 150; ny = 150
x_spacing = range(first(ga.xlim), stop=last(ga.xlim), length=nx)
y_spacing = range(first(ga.ylim), stop=last(ga.ylim), length=ny)
grid = RectangleGrid(x_spacing, y_spacing, 0 : 1 : 47)

interp = LocalGIFunctionApproximator(grid)
approx_solver = LocalApproximationValueIterationSolver(interp, max_iterations=1, verbose=true, is_mdp_generative=true, n_generative_samples=10)
approx_policy = solve(approx_solver, ga)

all_interp_values = get_all_interpolating_values(approx_policy.interp)
all_interp_states = get_all_interpolating_points(approx_policy.interp)

s = GAState(40,20,0)
v = value(approx_policy, s)
a = action(approx_policy, s)