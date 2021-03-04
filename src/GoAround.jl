#module GoAround

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
using HMMBase

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
    passive::Bool
    t::Int64
end

a = [0.99, 0.01]
A = [0.9 0.1; 0.1 0.9]
#            Safe means      Safe sigmas            Danger means  Danger sigmas
B = [MvNormal([360.0, 0.0], [120.0, 18.0]), MvNormal([0.0, 54.0], [120.0, 18.0])]


# Define a struct for the Go Around MDP
@with_kw mutable struct GoAroundMDP <: MDP{GAState, Symbol}
    pos_lim::Tuple{Float64, Float64}                = (0.0, 720.0)
    vel_lim::Tuple{Float64, Float64}                = (0.0, 54.0)
    discount::Float64                               = 0.95  
    p_become_aggressive                             = 0.05
    passive_vels                                    = Normal(-2,2)
    aggressive_vels                                 = Normal(2,2) 
    obs_hist                                        = Array{Int64}(undef, 0, 2) 
    hmm::HMM{Multivariate,Float64}                  = HMM(a,A,B)
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

function get_terminal_state()
    x = 0 
    v = 0 
    passive = true
    t = 0 # end of time
    sp = GAState(x, v, passive, t)
    return sp
end

# Define the generative model
function POMDPs.gen(ga::GoAroundMDP, s::GAState, a::Symbol, rng::AbstractRNG)

    if a==:continue
        o_v = s.v
        o_x = s.x

        #ga.obs_hist = vcat(ga.obs_hist, [o_x o_v])
        # Subtract 1 to take 1 or 2 to a boolean 0 or 1
        o_passive = viterbi(ga.hmm, [o_x o_v])[1] - 1
        o_t = s.t - 1
        if s.passive
            #v = s.v + rand(Normal(0,1))
            v = s.v + rand(ga.passive_vels)
            x = s.x - (10/150)*v
            passive = rand(rng) > ga.p_become_aggressive
            t = s.t - 1
            
        else
            v = s.v + rand(ga.aggressive_vels)
            x = s.x - (10/150)*v
            passive = rand(rng) < ga.p_become_aggressive
            #rand(rng) < ga.p_remain_passive
            t = s.t - 1
        end


        sp = GAState(x, v, passive, t)
        o = (o_x, o_v, o_passive, o_t)
        r = reward(ga, s, a, sp)
        return (sp=sp, o=o, r=r)
    else
        sp = get_terminal_state()
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
        rew = -1*(0.8)^s.t - 0.1
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
    passive = true
    t = 47
    
    sp = GAState(x,v,passive,t)
    r = reward(ga, s, a, sp)
    return (sp=sp, r=r)
end

# Define conversion functions for LocalApproximationValueIteration
function POMDPs.convert_s(::Type{GAState}, v::AbstractVector{Float64}, 
    ga::GoAroundMDP)
    s = GAState(convert(Float64,v[1]), convert(Float64,v[2]), convert(Bool, v[3]), convert(Int64, v[4]))
    return s
end

function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, 
    s::GAState, ga::GoAroundMDP)
    v = SVector{4,Float64}(convert(Float64, s.x), convert(Float64, s.v), convert(Float64, s.passive), convert(Float64, s.t))
    return v
end

#include("runtest.jl")

#end # module

# Instantiate the Go Around MDP and the grid
ga = GoAroundMDP()
nx = 100; ny = 100
x_spacing = range(first(ga.pos_lim), stop=last(ga.pos_lim), length=nx)
y_spacing = range(first(ga.vel_lim), stop=last(ga.vel_lim), length=ny)
grid = RectangleGrid(x_spacing, y_spacing, 0:1:1, 0:1:47)

# Solve using LocalApproximationValueIteration
interp = LocalGIFunctionApproximator(grid)
approx_solver = LocalApproximationValueIterationSolver(interp, max_iterations=1, verbose=true, is_mdp_generative=true, n_generative_samples=1)
approx_policy = solve(approx_solver, ga)

# Extract the policy and value function
t_arr = 48:-1:1
policy_evolution = zeros(Int8, nx, ny, 2, length(t_arr))
value_function_evolution = zeros(Float64, nx, ny, 2, length(t_arr))
for t in t_arr
    for (i,vel) in enumerate(y_spacing)
        for (j,pos) in enumerate(x_spacing)
            state = GAState(pos,vel,1,t)
            policy_evolution[i,j,1,49-t] = actionindex(ga, action(approx_policy, state))
            value_function_evolution[i,j,1,49-t] = value(approx_policy, state)
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
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), color="#440154", label="Go-Around")
    plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), color="#FDE725", label="Continue")
end
gif(policy_animation, "policy.gif", fps = 4)

value_function_animation = @animate for i in 1:48
    heatmap(x_ticks, y_ticks, value_function_evolution[:,:,1,i],color=:viridis,
    xlabel="Ground Vehicle Distance from Runway Centerline (miles)", 
    ylabel="Ground Vehicle Velocity (mph)",
    title="Time to Landing: $(48-i) Seconds")
end
gif(value_function_animation, "value_function.gif", fps = 4)





##******************************************************************************
# HMM TEST
#*******************************************************************************
#hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(1,1)])
a = [0.99, 0.01]
A = [0.9 0.1; 0.1 0.9]
#             Safe means     Safe sigmas            Danger means  Danger sigmas
#B = [MvNormal([360.0, 0.0], [120.0, 18.0]), MvNormal([0.0, 54.0], [120.0, 18.0])]
B = [MvNormal([360.0, 0.0], [120.0, 54.0]), MvNormal([0.0, 54.0], [120.0, 54.0])]
y = [180 2; 170 3; 170 5; 160 8; 150 10; 140 10; 130 12; 120 14; 110 14; 110 15; 100 16; 100 18; 90 19;]
y = [180 2; 170 3; 170 5; 160 8; 150 10; 140 10; 130 12; 120 14; 120 14; 120 12; 130 12; 140 10; 160 8;]
hmm = HMM(a,A,B)
y = [190 0]
state = viterbi(hmm, y)[1]
#post = posteriors(hmm,y)
#plot(post[:,1])
#plot!(post[:,2])

##
plot(TruncatedNormal(0,120,0,360), color = :red, fill=(0, .5,:red),label="Agressive",xlabel = "Position",)
plot!(TruncatedNormal(360,120,0,360), color = :blue, fill=(0, .5,:blue),label="Safe")
savefig("pos_dist.png")

#0,18,0,54
#54,18,0,54
plot(TruncatedNormal(-54,108,0,54), color = :blue, fill=(0, .5,:blue),label="Safe",xlabel = "Velocity",)
plot!(TruncatedNormal(108,108,0,54), color =:red, fill=(0, .5,:red),label="Aggressive")
savefig("vel_dist.png")

##
pos_arr = range(0, stop=360, length=100)
vel_arr = range(0, stop=54, length=100)
latent_state_arr = Array{Float64,2}(undef, 100, 100)
for (i,vel) in enumerate(vel_arr)
    for (j,pos) in enumerate(pos_arr)
       post =  posteriors(hmm,[pos vel])
       latent_state_arr[i,j] = post[1,1]
    end
end

heatmap(pos_arr, vel_arr, latent_state_arr,color=:viridis,xlabel = "Position",ylabel="Velocity")
plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), color="#440154", label="Aggressive")
plot!(Shape([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]), color="#FDE725", label="Passive")
savefig("latent_behavior.png")

# Array{Int64,4}(undef, 1000, 1000, 48, 2)