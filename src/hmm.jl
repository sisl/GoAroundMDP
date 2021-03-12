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

##
plot(Normal(-1,2), color = :blue, fill=(0, .5,:blue),label="Passive",xlabel = "Velocity",)
plot!(Normal(1,2), color = :red, fill=(0, .5,:red),label="Aggressive")
savefig("passive_v_aggressive.png")
