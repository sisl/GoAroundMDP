using POMDPs
using GridInterpolations
using Reel
using Plots 
using LocalFunctionApproximation
using LocalApproximationValueIteration       

plotly()

ga = GoAroundMDP()

nx = 150; ny = 150
grid = RectangleGrid(range(first(ga.xlim), stop=last(ga.xlim), length=nx), 
                     range(first(ga.ylim), stop=last(ga.ylim), length=ny))

interp = LocalGIFunctionApproximator(grid)
approx_solver = LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations=1000, is_mdp_generative=false)
approx_policy = solve(approx_solver, ga)