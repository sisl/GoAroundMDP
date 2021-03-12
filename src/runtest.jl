using POMDPs
using GridInterpolations
using Reel
using Plots 
using LocalFunctionApproximation
using LocalApproximationValueIteration 

#plotly()


ga = GoAroundPOMDP()
#print(ga.actions)

nx = 150; ny = 150
x_spacing = range(first(ga.xlim), stop=last(ga.xlim), length=nx)
y_spacing = range(first(ga.ylim), stop=last(ga.ylim), length=ny)
grid = RectangleGrid(x_spacing, y_spacing, 0 : 1 : 48)

interp = LocalGIFunctionApproximator(grid)
approx_solver = LocalApproximationValueIterationSolver(interp, max_iterations=1, verbose=true, is_mdp_generative=true, n_generative_samples=10)
approx_policy = solve(approx_solver, ga)