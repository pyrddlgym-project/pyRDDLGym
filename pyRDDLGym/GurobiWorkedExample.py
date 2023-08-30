import gurobipy
from gurobipy import GRB


def inner():
    model = gurobipy.Model()
    
    a1 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='a1')
    a2 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='a2')
    
    u1 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u1')
    u2 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u2')
    u3 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u3')
    u4 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u4')
    
    z1 = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z1')
    z2 = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z2')
    z3 = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z3')
    z4 = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z4')
    
    y1 = model.addVar(0, 1, vtype=GRB.BINARY, name='y1')
    y2 = model.addVar(0, 1, vtype=GRB.BINARY, name='y2')
    
    s1 = model.addVar(0, 5, vtype=GRB.CONTINUOUS, name='s1')
    
    model.addConstr(u1 == s1 + a1 - 10)
    model.addConstr(u2 == s1 + a1 + a2 - 10)
    model.addConstr(u3 == s1 - 10)
    model.addConstr(u4 == s1 - 10)
    
    model.addConstr(-z1 <= u1)
    model.addConstr(u1 <= z1)
    model.addConstr(-z2 <= u2)
    model.addConstr(u2 <= z2)
    model.addConstr(-z3 <= u3)
    model.addConstr(u3 <= z3)
    model.addConstr(-z4 <= u4)
    model.addConstr(u4 <= z4)
    
    model.addConstr(u3 + 1000 * y1 >= z3)
    model.addConstr(-u3 + 1000 * (1 - y1) >= z3)
    model.addConstr(u4 + 1000 * y2 >= z4)
    model.addConstr(-u4 + 1000 * (1 - y2) >= z4)
    
    model.setObjective(-z1 - z2 + z3 + z4, sense=GRB.MAXIMIZE)
    model.optimize()
    
    print(s1.X)
    print(a1.X)
    print(a2.X)
    print(-z1.X - z2.X)
    print(-z3.X - z4.X)


def outer():
    model = gurobipy.Model()
    model.params.NonConvex = 2
    
    eps = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eps')
    
    u1 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u1')
    u2 = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u2')
    
    z1 = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z1')
    z2 = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z2')
    
    w = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='w')
    b = model.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
    model.addConstr(eps >= z1 + z2)
    model.addConstr(u1 == b - 10)
    model.addConstr(u2 == (2 + w) * b - 10)
    model.addConstr(-u1 <= z1)
    model.addConstr(z1 <= u1)
    model.addConstr(-u2 <= z2)
    model.addConstr(z2 <= u2)
    
    model.setObjective(eps, sense=GRB.MINIMIZE)
    model.optimize()
    
    print(b.X)
    print(w.X)
    print(eps.X)
    
    
if __name__ == '__main__':
    inner()
    outer()
