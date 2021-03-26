# GoAroundMDP.jl

This repo serves as an initial exploration of planning for the
landing decision in the face of a potential runway incursion.

An OWNSHIP moves deterministically towards the runway landing threshold.
Time deteministically decrements from 48 seconds to 0 seconds as the 
ownship travels the last two miles of its final approach.

A GROUND VEHICLE moves with Gaussian noise along a 1D rail towards
the runway. The vehicle has a latent state representing its aggression
level. If the ground vehicle is aggressive it begins to accelerate 
towards the runway, and if it is passive then it begins to decelerate.
An aggressive ground vehicle remains aggressive with a high probability,
and a passive ground vehicle remains passive with a high probability.

"landing_scenario.gif" provides a visualization of the scenario.

