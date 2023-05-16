import gurobipy as gp

def solve_mc_gurobi(horizon):
    model = gp.Model()
    actions = model.addVars(list(range(horizon)), lb=0.0, ub=2.0, name='action')
    
    # state update equation
    def step(state, action):
        pos, vel = state
        
        # add expression cos(3 * pos)
        _3pos = model.addVar(lb=-3.6, ub=1.8)
        _cos3pos = model.addVar(lb=-1.0, ub=1.0)
        model.addConstr(_3pos == 3 * pos)
        model.addGenConstrCos(_3pos, _cos3pos)
        
        # update pos and vel
        new_vel = vel + (action - 1.0) * 0.001 - _cos3pos * 0.0025        
        model.addConstr(-0.07 <= new_vel)
        model.addConstr(new_vel <= 0.07)        
        new_pos = pos + new_vel
        model.addConstr(-1.2 <= new_pos)
        model.addConstr(new_pos <= 0.6)
        return (new_pos, new_vel)
    
    # calculate objective as final position
    cur_state = (-0.45, 0.0)
    for t in range(horizon):
        cur_state = step(cur_state, actions[t])
    obj = cur_state[0]
    
    # solve
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.Params.TimeLimit = 5 * 60
    model.optimize()
    
    # extract solution - will crash if unsolved
    if model.SolCount > 0:
        for v in model.getVars():
            if v.varName.startswith('action'):
                print('%s %g' % (v.varName, v.X))
        print('Obj: %g' % model.objVal)
    
if __name__ == "__main__":
    solve_mc_gurobi(120)
