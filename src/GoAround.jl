module GoAround

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
using Plots

#*******************************************************************************
# SETUP
#*******************************************************************************
plotly()

export
    GoAroundMDP,
    GoAroundMDPVis,
    GAState

mutable struct GAState
    x::Float64
    v::Float64
    t::Float64
end

@with_kw struct GoAroundMDP <: MDP{GAState, Symbol}
    xlim::Tuple{Float64, Float64}                   = (0.0, 50.0)
    ylim::Tuple{Float64, Float64}                   = (0.0, 50.0)
    discount::Float64                               = 0.95
end

#*******************************************************************************
# MDP FORMULATION
#*******************************************************************************
const go_around_actions = [:continue, :go_around]
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

# Define the transition function
function POMDPs.transition(ga::GoAroundMDP, s::GAState, a::Symbol)
    if a==:continue
        s.x = s.x + s.v
        s.v = s.v + rand(Normal(0,1))
        s.t = s.t - 1
        return Deterministic(1.0)
    else
        s.x = 0
        s.v = 0
        s.t = s.t - 1
        return Deterministic(1.0)
    end
end

# Define the reward function
function POMDPs.reward(ga::GoAroundMDP, s::GAState, a::Symbol)
    rew = 0.0
    if a==:continue
        if s.t==1 && s.x==40
            rew = -1
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
    t = 48
    return GAState(x,v,t)
end

# Define conversion functions for LocalApproximationValueIteration
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, 
    s::GAState, mdp::GoAroundMDP)
    v = SVector{2,Float64}(s.x, s.v)
    return v
end

function POMDPs.convert_s(::Type{GAState}, v::AbstractVector{Float64}, 
    mdp::GoAroundMDP)
    s = GAState(v[1], v[2], v[3])
    return s
end

include("runtest.jl")

end # module
