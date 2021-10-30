
"""Testing pyomo with different solvers."""
# %%
import pyomo.environ as pe

# %% 
# minimize 2*x1 + 3*x2**2, xi >=0
model = pe.ConcreteModel()
model.x = pe.Var(range(2), domain=pe.NonNegativeReals)
model.OBJ = pe.Objective(expr=2 * model.x[0] + 3 * model.x[1]**2)

# %%
# Gurobi (requires gurobipy and env variables for Gurobi license to be set)
opt = pe.SolverFactory("gurobi_direct")
opt.solve(model)

# %%
# glpk (requires glpk)
opt = pe.SolverFactory('glpk')
opt.solve(model)

# %%
# ipopt (requires ipopt *binary* built with ASL interface)
opt = pe.SolverFactory('ipopt')
opt.solve(model)

# %%
