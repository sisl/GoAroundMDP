#*******************************************************************************
# PACKAGES
#*******************************************************************************
using Random
using StatsPlots
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
using CSV
using DataFrames

#*******************************************************************************
# SETUP
#*******************************************************************************
export
    GoAroundPOMDP,
    GAState,
    GAObservation

# Define a struct for the Go Around POMDP state
mutable struct GAState
    x::Float64
    v::Float64
    passive::Bool
    t::Int64
end

# Define a struct for the Go Around POMDP observation
mutable struct GAObservation
    ox::Float64
    ot::Int64
end

# Define a struct for the Go Around POMDP
@with_kw struct GoAroundPOMDP <: POMDP{GAState, Symbol, GAObservation}
    pos_lim::Tuple{Float64, Float64}                = (0.0, 720.0)
    vel_lim::Tuple{Float64, Float64}                = (0.0, 54.0)
    discount::Float64                               = 0.95  
    p_become_aggressive                             = 0.05
    passive_vels                                    = Normal(-1,2)
    aggressive_vels                                 = Normal(1,2) 
end

#*******************************************************************************
# POMDP FORMULATION
#*******************************************************************************
POMDPs.actions(ga::GoAroundPOMDP) = [:continue, :go_around]
POMDPs.discount(ga::GoAroundPOMDP) = ga.discount
latent_states = [false, true]

# Define the actionindex function
function POMDPs.actionindex(ga::GoAroundPOMDP, a::Symbol)
    if a==:continue
        return 2
    elseif a==:go_around
        return 1
    end
    error("invalid GoAroundPOMDP action: $a")
end

function get_terminal_state()
    x = 0 
    v = 0 
    passive = true
    t = 0 # end of time
    sp = GAState(x, v, passive, t)
    return sp
end

# Define the generative model
function POMDPs.gen(ga::GoAroundPOMDP, s::GAState, a::Symbol, rng::AbstractRNG)
    if a==:continue
        if s.passive
            v = s.v + rand(rng, ga.passive_vels)
            passive = rand(rng) > ga.p_become_aggressive
        else
            v = s.v + rand(rng, ga.aggressive_vels)
            passive = rand(rng) < ga.p_become_aggressive
        end
        x = s.x - (10/150)*v
        t = s.t - 1
        ox = x
        ot = t

        sp = GAState(x, v, passive, t)
        o = (ox, ot)
        r = reward(ga, s, a, sp)
        return (sp=sp, o=o, r=r)
    else
        sp = get_terminal_state()
        r = reward(ga, s, a, sp)
        return (sp=sp, r=r)
    end
end

# Define the reward function
function POMDPs.reward(ga::GoAroundPOMDP, s::GAState, a::Symbol, sp::GAState)
    rew = 0.0
    if a==:continue
        distance = abs(s.x-360)
        if sp.t==0 && distance < 10
            rew = -1
        else
            rew = 0
        end
    elseif a==:go_around
        rew = -1*(0.8)^s.t - 0.1
    end
    return rew
end

# Define the isterminal function
function POMDPs.isterminal(ga::GoAroundPOMDP, a::Symbol, s::GAState)
    return (a==:go_around || s.t==0)
end


# Define the initialstate function
function POMDPs.initialstate(ga::GoAroundPOMDP, rng::AbstractRNG)
    x = rand(rng, 360:720)
    v = rand(rng, 0:54)
    passive = true
    t = 47
    
    sp = GAState(x,v,passive,t)
    r = reward(ga, s, a, sp)
    return (sp=sp, r=r)
end

# Define conversion functions for LocalApproximationValueIteration
function POMDPs.convert_s(::Type{GAState}, v::AbstractVector{Float64}, 
    ::UnderlyingMDP)
    s = GAState(convert(Float64,v[1]), convert(Float64,v[2]), 
                convert(Bool, v[3]), convert(Int64, v[4]))
    return s
end

function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, 
    s::GAState, ::UnderlyingMDP)
    v = SVector{4,Float64}(convert(Float64, s.x), convert(Float64, s.v), 
                           convert(Float64, s.passive), convert(Float64, s.t))
    return v
end

# Instantiate the Go Around MDP and the grid
ga = GoAroundPOMDP()
nx = 100; ny = 100
x_spacing = range(first(ga.pos_lim), stop=last(ga.pos_lim), length=nx)
y_spacing = range(first(ga.vel_lim), stop=last(ga.vel_lim), length=ny)
grid = RectangleGrid(x_spacing, y_spacing, 0:1:1, 0:1:47)

# Solve using LocalApproximationValueIteration
interp = LocalGIFunctionApproximator(grid)
approx_solver = LocalApproximationValueIterationSolver(
    interp, max_iterations=5, verbose=true, is_mdp_generative=true, 
    n_generative_samples=5)
approx_policy = solve(approx_solver, UnderlyingMDP(ga))

# Extract the policy and value function
t_arr = 48:-1:1
policy_evolution = zeros(Int8, nx, ny, 2, length(t_arr))
value_function_evolution = zeros(Float64, nx, ny, 2, length(t_arr))
for passive in latent_states
    for t in t_arr
        for (i,vel) in enumerate(y_spacing)
            for (j,pos) in enumerate(x_spacing)
                state = GAState(pos,vel,passive,t)
                policy_evolution[i,j,passive+1,49-t] = 
                    actionindex(ga, action(approx_policy, state))
                value_function_evolution[i,j,passive+1,49-t] = 
                    value(approx_policy, state)
            end
        end
    end
end

x_ticks = (x_spacing.-360)./240
y_ticks = y_spacing

policy_animation = @animate for i in 1:48
    heatmap(x_ticks, y_ticks, policy_evolution[:,:,1,i],color=:viridis,
    xlabel="Ground Vehicle Distance from Runway Centerline (miles)", 
    ylabel="Ground Vehicle Velocity (mph)",
    title="Time to Landing: $(48-i) Seconds", colorbar=false)
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), 
        color="#440154", label="Go-Around")
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), 
        color="#FDE725", label="Continue")
end
gif(policy_animation, "aggressive_policy.gif", fps = 4)

policy_animation = @animate for i in 1:48
    heatmap(x_ticks, y_ticks, policy_evolution[:,:,2,i],color=:viridis,
    xlabel="Ground Vehicle Distance from Runway Centerline (miles)", 
    ylabel="Ground Vehicle Velocity (mph)",
    title="Time to Landing: $(48-i) Seconds", colorbar=false)
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), 
        color="#440154", label="Go-Around")
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), 
        color="#FDE725", label="Continue")
end
gif(policy_animation, "passive_policy.gif", fps = 4)


value_function_animation = @animate for i in 1:48
    heatmap(x_ticks, y_ticks, value_function_evolution[:,:,1,i],color=:viridis,
    xlabel="Ground Vehicle Distance from Runway Centerline (miles)", 
    ylabel="Ground Vehicle Velocity (mph)",
    title="Time to Landing: $(48-i) Seconds")
end
gif(value_function_animation, "aggressive_value_function.gif", fps = 4)

value_function_animation = @animate for i in 1:48
    heatmap(x_ticks, y_ticks, value_function_evolution[:,:,2,i],color=:viridis,
    xlabel="Ground Vehicle Distance from Runway Centerline (miles)", 
    ylabel="Ground Vehicle Velocity (mph)",
    title="Time to Landing: $(48-i) Seconds")
end
gif(value_function_animation, "passive_value_function.gif", fps = 4)

# Array{Int64,4}(undef, 1000, 1000, 48, 2)
##
value_function = reshape(value_function_evolution, (100*100, 2*48))
CSV.write("value_function.csv",  DataFrame(value_function), writeheader=false)