import sympy as sp
from sympy import symbols, sympify, expand, oo
from sympy.matrices import Matrix

import xaddpy.utils.util as util

import json
from typing import Callable, Optional


def build_canonical_xadd_from_file(
        context,
        fname: str,
        model_name: str,
        var_name_rule: Optional[Callable] = None,
        prob_instance: Optional[dict] = None,
):
    """
    Basic parser for encoding an LP into a case function. Sympy variables and expressions are defined.
    A json file with the following form should be passed:
    {
        "prob-type": "bilinear"
        "cvariables0": ["x1", "x2"],
        "cvariables1": ["y1", "y2"],
        "min-values": [-10, -10, -10, -10],
        "max-values": [10, 10, 10, 10],
        "bvariables": ["b"],
        "ineq-constr": ["3*w1 + 5 <= 7", "0.5 * w2 + 10 <= 3", ...],
        "eq-constr": ["w1 ([2 * w2])"],
        "xadd": "([w1 <= 0] ([0]) ([1]))",
        "is-minimize": 1,
        "min-var-set-id": 0,
        "objective": "w1 + w2"
    }
    Args:
        context (XADD)
        fname (str): file name
        model_name (str)
        var_name_rule (optional, Callable): a function with which variable names are sorted
    """
    if prob_instance is None:
        try:
            with open(fname, "r") as json_file:
                prob_instance = json.load(json_file)
        except:
            print("Failed to open file!!")
            return

    # Check the loaded file
    check_json(prob_instance)

    # Set the problem type
    context.set_problem(prob_instance['prob-type'])

    # Which set of variables to minimize? 0 or 1
    min_var_set_id = prob_instance['min-var-set-id']

    # Namespace to be used to define sympy symbols
    ns = {}

    # is_minimize?
    is_min = True if prob_instance['is-minimize'] else False

    context.link_json_file(config_json_fname=fname)

    # Create Sympy symbols for cvariables and bvariables
    if len(prob_instance['cvariables0']) == 1 and isinstance(prob_instance['cvariables0'][0], int):
        cvariables0 = symbols('x1:%s' % (prob_instance['cvariables0'][0]+1))
    else:
        cvariables0 = []
        for v in prob_instance['cvariables0']:
            cvariables0.append(sp.symbols(v))
    if len(prob_instance['cvariables1']) == 1 and isinstance(prob_instance['cvariables1'][0], int):
        cvariables1 = symbols('y1:%s' % (prob_instance['cvariables1'][0]+1))
    else:
        cvariables1 = []
        for v in prob_instance['cvariables1']:
            cvariables1.append(sp.symbols(v))
    cvariables = cvariables0 + cvariables1

    if len(prob_instance['bvariables']) == 1 and isinstance(prob_instance['bvariables'][0], int):
        bvariables = symbols(f"b1:{prob_instance['bvariables'][0] + 1}", integer=True)
    else:
        bvariables = []
        for v in prob_instance['bvariables']:
            bvariables.append(sp.symbols(v, integer=True))

    # Retrieve dimensions of problem instance
    cvar_dim = len(cvariables)                  # Continuous variable dimension
    bvar_dim = len(prob_instance['bvariables'])  # Binary variable dimension
    # bvar_dim = 2 if bvar_dim == 1 else bvar_dim    # This was here only to handle a shortest path problem with 2 edges
    var_dim = cvar_dim + bvar_dim  # Total dimensionality of variables
    assert bvar_dim + cvar_dim > 0, "No decision variables provided"
    # assert (bvar_dim == 0 and cvar_dim != 0) or (bvar_dim != 0 and cvar_dim == 0) # Previously, we accepted either continuous or binary variables, not both

    variables = list(cvariables) + list(bvariables)
    ns.update({str(v): v for v in variables})

    # retrieve lower and upper bounds over decision variables
    min_vals = prob_instance['min-values']
    max_vals = prob_instance['max-values']

    if len(min_vals) == 1 and len(min_vals) == len(max_vals):       # When a single number is used for all cvariables
        min_vals = min_vals * cvar_dim
        max_vals = max_vals * cvar_dim

    assert len(min_vals) == len(max_vals) and len(min_vals) == cvar_dim, \
        "Bound information mismatch!\n cvariables: {}\tmin-values: {}\tmax-values: {}".format(
            prob_instance['cvariables'],
            prob_instance['min-values'],
            prob_instance['max-values'])

    bound_dict = {}
    for i, (lb, ub) in enumerate(zip(min_vals, max_vals)):
        lb, ub = sp.S(lb), sp.S(ub)
        bound_dict[ns[str(cvariables[i])]] = (lb, ub)

    # Update XADD attributes
    variables = [ns[str(v)] for v in variables]
    if var_name_rule is not None:
        variables = list(sorted(variables, key=var_name_rule))      # Order variables if needed
    bvariables = [ns[str(bvar)] for bvar in bvariables]
    cvariables0 = [ns[str(cvar)] for cvar in cvariables0]
    cvariables1 = [ns[str(cvar)] for cvar in cvariables1]
    context.update_decision_vars(bvariables, cvariables0, cvariables1, min_var_set_id)
    context.update_bounds(bound_dict)
    if context._problem_type == 'predict-then-optimize':
        ns = context.create_symbolic_coeffs(ns)
    context.update_name_space(ns)

    # Read constraints and link with the created Sympy symbols
    # If an initial xadd is directly provided in str type, need also return it
    ineq_constrs = []
    eq_constr_dict = {}
    for const in prob_instance['ineq-constr']:
        ineq_constrs.append(sympify(const, locals=ns))

    if prob_instance['xadd']:
        init_xadd = parse_xadd_grammar(prob_instance['xadd'], ns)[1][0]
    else:
        init_xadd = None

        # Handle equality constraints separately.
        # Equality constraints are firstly converted to a system of linear equations. Then, the reduced row echelon form
        # of the coefficient matrix tells us linearly independent equality constraints. Only put these constraints
        # when building the initial LP XADD.
        eq_constr_dict, variables = util.compute_rref_filter_eq_constr(prob_instance['eq-constr'], variables, locals=ns)

    assert (len(ineq_constrs) + len(eq_constr_dict) == 0 and init_xadd is not None) or \
           (len(ineq_constrs) + len(eq_constr_dict) != 0 and init_xadd is None), \
           "When xadd formulation is provided, make sure no other constraints passed (vice versa)"

    # Read in objective function if provided
    obj = prob_instance['objective']
    if obj:
        obj = expand(sympify(prob_instance['objective'], locals=ns))

    # Build XADD from constraints and the objective
    if (obj and init_xadd is None):
        mp_formulation = ineq_constrs + [obj]
        math_program = context.build_initial_xadd_lp(mp_formulation, is_min=is_min)

    # Build XADD from given case statement
    elif init_xadd is not None:
        math_program = context.build_initial_xadd(init_xadd)

    elif not obj and (init_xadd is None):
        # symbolic expression of the objective: c'x, where c = param @ features
        obj = expand((context._pred_coeff_vec.T * Matrix(variables))[0])
        mp_formulation = ineq_constrs + [obj]
        math_program = context.build_initial_xadd_lp(mp_formulation, is_min=is_min)

    else:
        raise ValueError

    context._prune_equality = False
    # Substitute in equality constraints to `math_program` XADD
    # It is guaranteed that we don't need to substitute one equality constraint into another;
    # this is due to the fact that we have reduced the equality constraints into the reduced row echelon form.
    for v_i, rhs in eq_constr_dict.items():
        ## TODO: how to handle binary variables?
        # The equality constraint over v_i can be seen as its annotation
        v_i_anno = context.get_leaf_node(rhs)
        context.update_anno(v_i, v_i_anno)

        # Substitute into the problem formulation
        math_program = context.reduce_lp(context.substitute(math_program, {v_i: rhs}))

        # Keep track of the order.. later retrieve argmin(max) using this
        context.add_eliminated_var(v_i)
        variables.remove(v_i)

        # Add bound constraints of v_i if exists
        lb, ub = bound_dict[v_i]
        bound_constraints = []

        if lb != -oo:
            comp = (rhs >= lb)
            bound_constraints.append((comp, True))
        if ub != oo:
            comp = (rhs <= ub)
            bound_constraints.append((comp, True))
        for d, b in bound_constraints:
            high_val = oo if (b and not is_min) or (not b and is_min) else -oo
            low_val = -oo if (b and not is_min) or (not b and is_min) else oo
            bound_constraint = context.get_dec_node(d, low_val, high_val)
            math_program = context.apply(bound_constraint, math_program, 'min' if not is_min else 'max')
            math_program = context.reduce_lp(math_program)
    context._prune_equality = True
    math_program = context.reduce_lp(math_program)

    # Final variables set
    min_vars = cvariables0 if min_var_set_id == 0 else cvariables1
    free_vars = cvariables0 if min_var_set_id == 1 else cvariables1
    variables = dict(
        min_var_list=[ns[str(cvar)] for cvar in min_vars if ns[str(cvar)] in variables],
        free_var_list=[ns[str(cvar)] for cvar in free_vars if ns[str(cvar)] in variables],
    )
    return variables, math_program, eq_constr_dict


def check_json(json_loaded):
    assert "cvariables0" in json_loaded and "cvariables1" in json_loaded, \
        "list of cvariables not in .json file.. exiting.."
    assert "bvariables" in json_loaded, "bvariables not in .json file.. exiting.."
    assert "min-values" in json_loaded, "min-values not in .json file.. exiting.."
    assert "max-values" in json_loaded, "max-values not in .json file.. exiting.."
    assert "min-var-set-id" in json_loaded, "min-var-set not in .json file.. exiting.."
    assert "ineq-constr" in json_loaded, "ineq-constr not in .json file.. exiting.."
    assert "eq-constr" in json_loaded, "eq-constr not in .json file.. exiting.."
    assert "is-minimize" in json_loaded, "is-minimize not in .json file.. exiting.."
    assert (len(json_loaded['cvariables0']) + len(json_loaded['cvariables1'])) * len(json_loaded['bvariables']) == 0, \
        "Currently, either only cvariables or bvariables can be defined, not both."


def create_matrix(var_name, row, col, is_positive=False, feature=False, col_vec=False):

    variables = [symbols('{}{}{}'.format(var_name, i, j), positive=is_positive) for i in range(1, row + 1) for j in
                     range(1, col+1)]
    var_lst = []
    for i in range(row):
        var_lst.append([])
        for j in range(col):
            var_lst[i].append(variables[i * col + j])
    mat = Matrix(var_lst)

    if feature:
        # Append a column of ones to the left in case of 2-dimensional feature matrix;
        # or, simply append 1 to the top if 1-dimensional feature
        if col_vec:
            mat = Matrix([sympify(1), mat])
        else:
            one_column = Matrix.ones(row, cols=1)
            mat = Matrix.hstack(one_column, mat)
    else:
        # Append the bias column or a single bias parameter
        if col_vec:
            mat = Matrix([symbols('{}00'.format(var_name), is_positive=is_positive), mat])
        else:
            bias_column = Matrix([symbols('{}{}0'.format(var_name, i)) for i in range(1, row+1)])
            mat = Matrix.hstack(bias_column, mat)
    return mat


def push(obj, l, depth):
    """Helper function for parsing a string into XADD"""
    while depth:
        l = l[-1]
        depth -= 1

    l.append(obj)


def parse_xadd_grammar(s, ns):
    """Helper function for parsing a string into XADD"""
    groups = []
    depth = 0
    sympyLst = []
    s = s.strip()

    try:
        i = 0
        while i < len(s):
            if s[i] == '(':
                push([], groups, depth)
                push([], sympyLst, depth)
                depth += 1
                i += 1
            elif s[i] == ')':
                depth -= 1
                i += 1
            else:
                idx_next_open = s.find('(', i, -1)
                idx_next_close = s.find(')', i)
                if idx_next_open < 0 and idx_next_close < 0:
                    break
                elif idx_next_open < 0 and idx_next_close > 0:
                    idx_next_paren = idx_next_close
                elif idx_next_open > 0 and idx_next_close < 0:
                    raise ValueError
                elif idx_next_open > 0 and idx_next_close > 0:
                    idx_next_paren = min(idx_next_open, idx_next_close)

                s_chunk = s[i: idx_next_paren]
                i += len(s_chunk)
                s_chunk = s_chunk.strip()
                if s_chunk:
                    push(s_chunk, groups, depth)
                    push(sympify(s_chunk.strip('([])'), locals=ns), sympyLst, depth)

    except IndexError:
        raise ValueError('Parentheses mismatch')

    if depth > 0:
        raise ValueError('Parentheses mismatch')
    else:
        return groups, sympyLst