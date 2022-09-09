import argparse
from sympy import sympify, collect
from sympy.solvers import solveset
import sympy.core.relational as relational
from sympy.matrices import Matrix
import sympy
from xaddpy.utils.global_vars import REL_TYPE, REL_REVERSED
import pickle
import numpy as np
from datetime import datetime

typeConverter = {sympy.Integer: int, sympy.Float: float, sympy.core.numbers.Zero: int, sympy.core.numbers.NegativeOne: int,
                 sympy.core.numbers.One: int, sympy.core.numbers.Rational: float, sympy.core.numbers.Half: float,
                 sympy.core.numbers.Infinity: float, sympy.core.numbers.NegativeInfinity: float}


def reverse_expr(expr):
    """
    Given a Sympy inequality expression, reverse the expression.
    :param expr:
    :return:
    """

    lhs, rhs, rel = expr.lhs, expr.rhs, REL_TYPE[type(expr)]
    rel = REL_REVERSED[rel]
    return relational.Relational(lhs, rhs, rel)


def export_argmin_solution(fname, context, g_model, ):
    res_dict = {}

    # Retrieve the name space of sympy variables and save
    ns = context._name_space
    res_dict['namespace'] = ns

    # Save decision variables and their argmin solutions
    var_set = context._decisionVars
    var_to_anno = context.get_all_anno()
    res_dict['argmin'] = {}
    for v, anno in var_to_anno.items():
        node = context.get_exist_node(anno)
        res_dict['argmin'][str(v)] = str(node)
    res_dict['decision_vars'] = var_set

    # Save Gurobi model
    res_dict['gurobi_model'] = g_model

    # Output a file
    with open(fname, 'wb') as f_pickle:
        pickle.dump(res_dict, f_pickle)


def compute_rref_filter_eq_constr(eq_constr_str, variables, locals):
    """
    eq_constr_str is a list of strings corresponding to initially given equality constraints in .json file.
    This function returns two things:
        1) dictionary
        2) list
    :param eq_constr_str:   (list) List of equality constraints in str type.
    :param variables:       (list) List of sympy variables
    :param locals:          (dict) Namespace of Sympy variables
    :return:
    """
    if len(eq_constr_str) == 0:
        return {}, variables

    eq_constr_dict = {}
    eq_constr_lst = []
    eq_constr = []
    variables = Matrix(variables)

    # Each equality constraint transformed to lhs - rhs = 0
    for const in eq_constr_str:
        lhs, rhs = const.split('=')
        lhs, rhs = sympify(lhs, locals=locals), sympify(rhs, locals=locals)
        expr = lhs - rhs
        if expr.free_symbols.intersection(variables.free_symbols):
            eq_constr.append(expr)

    if len(eq_constr) == 0:
        return {}, list(variables)

    # Build the coefficient matrix (constants are also appended at the rightmost column)
    A = Matrix([[row.coeff(v) for v in variables] for row in eq_constr])
    C = Matrix([row.as_coefficients_dict()[1] for row in eq_constr])
    A_ = Matrix.hstack(A, C)

    # Compute the reduced row echelon form of the A_ matrix
    R, pivots = A_.rref()               # R is the reduced matrix; pivots is a tuple of pivot column indices
    num_lin_indep_constr = len(pivots)  # Number of pivots indicates the # of linearly independent equality constraints
    R = R[:num_lin_indep_constr, :]     # Remove rows of zeros

    # Append 1 at the end of variables column and perform matrix multiplication of R and the variable vector.
    E = R * Matrix.vstack(variables, Matrix([1]))
    for p, eq in zip(pivots, E):
        pvar = variables[p]
        rhs = solveset(eq, pvar).args[0]
        eq_constr_dict[pvar] = rhs

    return eq_constr_dict, list(variables)


def center_features(feature):
    mean = np.mean(feature, axis=0)
    feature = feature - mean
    return feature, mean


def decenter_features(feature, mean):
    return feature + mean


def get_coefficients(expr, var_lst: list):
    """
    Given a Sympy expression and a list of variables, return the coefficients of the variables in the expression.
    """
    coeffs = ()
    for var in var_lst:
        coeff = collect(expr, var, evaluate=False).get(var, None)
        if coeff is None:
            coeff = 0.0
        coeffs += (float(coeff), )
    return coeffs


def get_multiplied_expr(expr, var):
    """
    Given a Sympy expression and a sympy variable, return the expression that is multiplied to the variable.
    """
    mul_expr = collect(expr, var, evaluate=False).get(var, 0)
    return mul_expr


def is_bilinear(expr):
    """
    Given a Sympy expression, check whether it is bilinear.
    """
    terms = expr.as_ordered_terms()
    return any(map(lambda t: len(t.free_symbols) >= 2, terms))


def get_depth(context, node_id, depth, lst):
    node = context.get_exist_node(node_id)

    if node._is_leaf:
        lst.append(depth)
        return

    low = node._low
    high = node._high

    get_depth(context, low, depth + 1, lst)
    get_depth(context, high, depth + 1, lst)


def get_num_nodes(context, node_id):
    """Returns the total number of decision nodes and terminal nodes given the root node id.
    Args:
        context (xadd.XADD)
        node_id (int) The root node ID
    """
    num_nodes = [0, 0]
    counted = set()

    def _get_num_nodes(context, node_id):
        node = context.get_exist_node(node_id)
        new_node = False
        if node_id not in counted:
            new_node = True
        counted.add(node_id)

        if node._is_leaf:
            if new_node:
                num_nodes[1] += 1
            return

        if new_node:
            num_nodes[0] += 1

        # Recurse
        low = node._low
        high = node._high
        _get_num_nodes(context, low)
        _get_num_nodes(context, high)

    # Call the function
    _get_num_nodes(context, node_id)

    return num_nodes[0], num_nodes[1]   # Number of internal, terminal nodes


def get_bound(var, expr):
    """
    Return either lower bound or upper bound of 'var' from expr.
    :param var:     (sympy.Symbol) target variable
    :param expr:    (sympy.relational) An inequality over 'var'
    :return:        (sympy.Basic, bool) a sympy expression along with the boolean value indicating whether an upper
                    or lower bound. True for upper bound, False for lower bound.
    """
    comp = sympy.solve(expr, var)

    if isinstance(comp, sympy.And):
        assert len(comp.args) == 2, "No more than 3 terms should be generated as a result of solve(ineq)!"
        args1rhs = comp.args[1].canonical.rhs             # when 'var' is the only variable in the
        i = 0 if (args1rhs == sympy.oo) or (args1rhs == -sympy.oo) else 1
        comp = comp.args[i]
    else:
        comp = sympy.simplify(comp)

    comp = comp.canonical
    # check whether upper bound or lower bound over 'var'
    # if ub: 'var' <= upper bound, else: 'var' >= lower bound
    ub = isinstance(comp, relational.LessThan) or isinstance(comp, relational.StrictLessThan)
    expr = comp.rhs
    return expr, ub


def solve_lp_from_lp_xadd(context, lp, eq_constr):
    import gurobipy as gp
    from xaddpy.utils.milp_encoding import convert2GurobiExpr
    from xaddpy.utils.gurobi_util import GurobiModel

    model = GurobiModel('Validate')
    model.setObjective(1, sense=gp.GRB.MINIMIZE)        # just check feasibility
    sympy2gurobi = {}
    ns = context._name_space
    def compile_gurobi_model(node_id):
        node = context.get_exist_node(node_id)

        if node._is_leaf:
            return
        dec = node._dec
        expr = context._id_to_expr[dec]
        lhs, rel, rhs = expr.lhs, type(expr), expr.rhs
        g_lhs = convert2GurobiExpr(lhs, model, context)

        rel = REL_TYPE[rel]
        if rel == '<=' or rel == '<':
            model.addConstr(g_lhs <= 0)
        else:
            model.addConstr(g_lhs >= 0)

        low = node._low
        compile_gurobi_model(low)
        high = node._high
        compile_gurobi_model(high)

    for lhs, rhs in eq_constr.items():
        lhs = sympy.sympify(lhs, locals=ns)
        rhs = sympy.sympify(rhs, locals=ns)
        lhs = lhs - rhs
        g_lhs = convert2GurobiExpr(lhs, model, context)
        model.addConstr(g_lhs == 0)

    compile_gurobi_model(lp)
    print('Optimize model')
    model.update()
    model.optimize()
    status = model.status
    if status == gp.GRB.OPTIMAL:
        print('Feasible')
    elif status == gp.GRB.INFEASIBLE:
        print('Infeasible')
    else:
        print('status: ', status)


def get_date_time():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")