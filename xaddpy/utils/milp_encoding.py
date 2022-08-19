from xaddpy.utils.util import typeConverter
from xaddpy.utils.global_vars import EPSILON
import sympy

import numpy as np
from typing import Union


def handle_infeasibility(
        g_model,
        dec_partition: list,
):
    """
    Enforces a set of constraints such that the infeasible partition is accounted for.
    For example, suppose we have `dec_partition=[d1, d2, d3]` whose function value is \infty.
    Then, it implies that `d1 and d2 => not d3` because if d1, d2, and d3 are all true, that leads to infeasibility.
    Encoding such a logical condition can be done by defining additional binary variables.
    In the previous example, if `i1, i2, i3` are the associated binary variables, respectively, then we enforce
        (1 - i1) + (1 - i2) >= i3
        Note that if (i1, i2) = (1, 1), then i3 = 0 should hold
        otherwise, i3 can have either 0 or 1... not constrained.
    On the other hand, if `dec_partition=[d1, not d2, d3]`, then
        (1 - i1) + i2 >= i3
        e.g., (i1, i2) = (1, 0) => i3 = 0
    and so on and so forth...
    Args:
          g_model (GurobiModel)
          dec_partition (list)  The list containing decisions (conditionals) that have been encountered until reaching
            the current infeasible partition
    """
    # Decisions and their indicator variables up until the last decision
    ind_vars = []
    for dec in dec_partition:
        ind_v = g_model.getVarByName(f"ind_{dec if dec > 0 else -dec}")
        assert ind_v is not None, "Binary variables associated with parent nodes should have defined already!"
        ind_vars.append(ind_v)

    # Create the constraint
    # TODO: Constraint cache should be provided...
    try:
        constr_name = f"Feasibility_({'_'.join(map(lambda x: x.replace('-', 'n'), map(str, dec_partition[:-1])))})" \
                      f"_impl_({str(-dec_partition[-1]).replace('-', 'n')})"
    except IndexError:
        pass
    constr, _ = g_model.getConstrByName(constr_name)
    if constr is None:
        if len(dec_partition) == 1:
            g_model.addConstr(ind_vars[0] == 0 if dec_partition[0] > 0 else ind_vars[0] == 1, name=constr_name)
        else:
            lhs, rhs = 0, 0
            for i, (v, dec) in enumerate(zip(ind_vars, dec_partition)):
                if dec < 0:
                    if i == len(dec_partition) - 1:
                        rhs += 1
                    lhs += v
                else:
                    if i < len(dec_partition) - 1:
                        rhs -= 1
                    lhs -= v
            g_model.addConstr(lhs >= rhs, name=constr_name)


def build_milp(
        context,
        node_id: int,
        g_model,
        dec_partition: list,
        lb: float = float('-inf'),
        ub: float = float('inf'),
        binary: bool = False,
        **kwargs,
) -> Union[int, sympy.Basic]:
    """
    Given an XADD, encode it as MILP.
    Args:
        context (XADD)
        node_id (int)
        g_model(GurobiModel)
        dec_partition (list)
        lb (float)
        ub (float)
        binary (bool)
    """
    node = context.get_exist_node(node_id)

    if node._is_leaf:
        expr = node.expr

        # Handle infeasibility
        if expr == sympy.oo or expr == -sympy.oo:
            handle_infeasibility(g_model, dec_partition)

        return expr

    dec = node.dec
    expr = context._id_to_expr[dec]
    assert expr.rhs == 0, "RHS should always be 0"

    # Add variables associated with decision expressions (indicator) and nodes
    indicator_var_name = f"ind_{dec}"
    indicator = g_model.getVarByName(indicator_var_name, binary=True)  # To reuse ind_dec variable
    if indicator is None:
        g_model.addVar(vtype=GRB.BINARY, name=indicator_var_name)

    # Each unique node is associated with one variable (continuous since the objective is continuous)
    icvar_node_name = f'icvar_{node_id}'
    gvar_node = g_model.getVarByName(icvar_node_name, binary=False)
    if gvar_node is None:
        g_model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=icvar_node_name)

    # Recursively build the MILP model: low branch -> high branch
    dec_partition.append(-dec)
    low = build_milp(
        context,
        node._low,
        g_model,
        dec_partition=dec_partition,
        lb=lb,
        ub=ub,
        binary=binary,
        **kwargs)
    dec_partition.remove(-dec)

    dec_partition.append(dec)
    high = build_milp(
        context,
        node._high,
        g_model,
        dec_partition=dec_partition,
        lb=lb,
        ub=ub,
        binary=binary,
        **kwargs)
    dec_partition.remove(dec)

    # Check for duplicated constraints and add
    g_model.addDecNodeIndicatorConstr(dec, expr)
    g_model.addIntNodeIndicatorConstr(dec, node_id, low=low, high=high)

    return node_id


def build_milp_from_bilevel(
        context,
        node_id: int,
        g_model,
        dec_partition: list,
        num_data=None,
        root=False,
        lb=float('-inf'),
        ub=float('inf'),
        binary=False,
        **kwargs
):
    """
    Given an xadd, convert it to a gurobi MILP model directly. data are given which should be substituted on the fly.
    Args:
        context (XADD): XADD object
        node_id (int): the id of the node
        g_model (GurobiModel): the gurobi model
        dec_partition (list)
        num_data (int): the size of dataset
        root (bool)
        lb (float)
        ub (float)
        binary (bool): whether the current decision variable is binary
    """
    node = context.get_exist_node(node_id)

    # Returns the expression at leaf nodes; if it is a root node, add the constraints directly here
    if node._is_leaf:
        expr = node.expr

        # Handle infeasibility
        if expr == sympy.nan:
            # TODO: data_idx should be provided below as well?
            handle_infeasibility(g_model, dec_partition)

        # if argmax XADD is a terminal node, directly set the value of the variable
        if root:
            for i in range(num_data):
                gvar_node = g_model.getVarByName(f"icvar_{node_id}__{i}")
                assert gvar_node is not None  # This variable should be added before calling this function

                if not expr.is_number:
                    gurobi_expr = convert2GurobiExpr(
                        expr, g_model, i, binary, incl_bound=True,
                    )
                else:
                    gurobi_expr = float(expr)
                g_model.addConstr(gvar_node == gurobi_expr)
            g_model.update()
            return expr
        else:
            return expr

    dec = node.dec
    expr = context._id_to_expr[dec]
    assert expr.rhs == 0, "RHS should always be 0"

    # Add variables associated with decision expressions (indicator) and nodes
    for i in range(num_data):
        # Add indicator variables corresponding to decision nodes
        indicator_var_name = f"ind_{dec}__{i}"
        indicator = g_model.getVarByName(indicator_var_name, binary=True)  # To reuse ind_dec variable
        if indicator is None:
            g_model.addVar(vtype=GRB.BINARY, name=indicator_var_name)

        # Each unique node is associated with one variable (can assume as continuous var)
        icvar_node_name = f'icvar_{node_id}__{i}'
        gvar_node = g_model.getVarByName(icvar_node_name, binary=False)
        if gvar_node is None:
            g_model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=icvar_node_name)

    # Recursively build the MILP model: low branch -> high branch
    dec_partition.append(-dec)
    low = build_milp_from_bilevel(
        context,
        node._low,
        g_model,
        dec_partition,
        num_data,
        lb=lb,
        ub=ub,
        binary=binary,
        **kwargs
    )
    dec_partition.remove(-dec)

    dec_partition.append(dec)
    high = build_milp_from_bilevel(
        context,
        node._high,
        g_model,
        dec_partition,
        num_data,
        lb=lb,
        ub=ub,
        binary=binary,
        **kwargs
    )
    dec_partition.remove(dec)

    # Check for duplicated constraints and add logical constraints
    g_model.addDecNodeIndicatorConstr(dec, expr, size=num_data)
    g_model.addIntNodeIndicatorConstr(dec, node_id, low=low, high=high, size=num_data)
    return node_id


def build_milp_with_given_leaf_val(
        context,
        node_id: int,
        g_model,
        leaf_val: sympy.Basic,
        dec_partition: list,
        lb: float = float('-inf'),
        ub: float = float('inf'),
        binary: bool = False,
        **kwargs,
) -> Union[int, sympy.Basic]:
    """
    Given an XADD, encode it as MILP.
    Args:
        context (XADD)
        node_id (int)
        g_model(GurobiModel)
        leaf_val (sp.Basic)
        dec_partition (list)
        lb (float)
        ub (float)
        binary (bool)
    """
    node = context.get_exist_node(node_id)

    if node._is_leaf:
        expr = node.expr

        # Handle infeasibility
        if expr == sympy.oo or expr == -sympy.oo:
            handle_infeasibility(g_model, dec_partition)
            return expr
        # Make all feasible partitions have the same leaf value
        else:
            return leaf_val

    dec = node.dec
    expr = context._id_to_expr[dec]
    assert expr.rhs == 0, "RHS should always be 0"

    # Add variables associated with decision expressions (indicator) and nodes
    indicator_var_name = f"ind_{dec}"
    indicator = g_model.getVarByName(indicator_var_name, binary=True)  # To reuse ind_dec variable
    if indicator is None:
        g_model.addVar(vtype=GRB.BINARY, name=indicator_var_name)

    # Each unique node is associated with one variable (continuous since the objective is continuous)
    icvar_node_name = f'icvar_{node_id}'
    gvar_node = g_model.getVarByName(icvar_node_name, binary=False)
    if gvar_node is None:
        g_model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=icvar_node_name)

    # Recursively build the MILP model: low branch -> high branch
    dec_partition.append(-dec)
    low = build_milp_with_given_leaf_val(
        context,
        node._low,
        g_model,
        leaf_val,
        dec_partition=dec_partition,
        lb=lb,
        ub=ub,
        binary=binary,
        **kwargs)
    dec_partition.remove(-dec)

    dec_partition.append(dec)
    high = build_milp_with_given_leaf_val(
        context,
        node._high,
        g_model,
        leaf_val,
        dec_partition=dec_partition,
        lb=lb,
        ub=ub,
        binary=binary,
        **kwargs)
    dec_partition.remove(dec)

    # Check for duplicated constraints and add
    g_model.addDecNodeIndicatorConstr(dec, expr)
    g_model.addIntNodeIndicatorConstr(dec, node_id, low=low, high=high)

    return node_id


def check_redundancy_and_add_constraint(
        g_model,
        constr_cache,
        dec,
        boolean,
        *args, **kwargs
):
    dec_id = kwargs.get('dec_id', None)
    assert dec_id is not None
    num_data = kwargs.get('num_data', None)
    ind_var_name = f'ind_{dec}'

    # Indicator = 1 or 0 -> Equality constraint
    if len(args) == 2:
        lhs, rhs = args
        constr_name = f'GC_({ind_var_name})_({boolean})_({lhs})_eq_({rhs})'
        check = constr_name in constr_cache
        if not check:
            if num_data is None:
                indicator = g_model.getVarByName(ind_var_name)
                lhs = g_model.getVarByName(f'icvar_{lhs}')
                if isinstance(rhs, sympy.core.numbers.Number):
                    rhs = float(rhs)
                elif isinstance(rhs, int):
                    rhs = g_model.getVarByName(f'icvar_{rhs}')
                else:
                    rhs = convert2GurobiExpr(rhs, g_model)
                g_model.addGenConstrIndicator(indicator, boolean, lhs == rhs,
                                              name=f"GC_({ind_var_name})_({boolean})_({lhs})_equalto_({str(rhs)[:20].replace(' ', '')})")
            else:
                for i in range(num_data):
                    indicator = g_model.getVarByName(f'{ind_var_name}__{i}')
                    lhs_i = g_model.getVarByName(f'icvar_{lhs}__{i}')
                    if isinstance(rhs, sympy.core.numbers.Number):
                        rhs_i = float(rhs)
                    elif isinstance(rhs, int):
                        rhs_i = g_model.getVarByName(f'icvar_{rhs}__{i}')
                    else:
                        rhs_i = convert2GurobiExpr(rhs, g_model, data_idx=i)
                    g_model.addGenConstrIndicator(indicator, boolean, lhs_i == rhs_i,
                                                  name=f"GC_({ind_var_name})_({boolean})_({lhs})_equalto_({str(rhs)[:20].replace(' ', '')})__{i}")
            constr_cache.add(constr_name)

    # Indicator = 1 or 0 -> inequality constraint (for decision nodes)
    elif len(args) == 3:
        lhs, rel, rhs = args
        constr_name = f'GC_({ind_var_name})_({boolean})_({dec_id})'

        check = constr_name in constr_cache
        if not check:
            # To handle the case when the decision holds in equality... slightly perturb
            epsilon = kwargs.get('epsilon', EPSILON)
            rhs = rhs - epsilon if rel == '<' else rhs + epsilon

            if num_data is None:
                indicator = g_model.getVarByName(ind_var_name)
                lhs = convert2GurobiExpr(lhs, g_model)
                g_model.addGenConstrIndicator(indicator, boolean, lhs, rel, rhs, name=constr_name)
            else:
                for i in range(num_data):
                    indicator = g_model.getVarByName(f'{ind_var_name}__{i}')
                    lhs_i = convert2GurobiExpr(lhs, g_model, i)
                    g_model.addGenConstrIndicator(indicator, boolean, lhs_i, rel, rhs, name=f'{constr_name}__{i}')
            constr_cache.add(constr_name)


def convert2GurobiExpr(expr, g_model, data_idx=None, binary=False, incl_bound=True):
    """
    Given a sympy expression, convert it to a gurobi expression. An expression can be a simple linear expression,
    or linear inequality.
    """
    sympy2gurobi = g_model.sympy_to_gurobi

    # If cached, return immediately
    if data_idx is None and expr in sympy2gurobi:
        return sympy2gurobi[expr]
    elif data_idx is not None and (expr, data_idx) in sympy2gurobi:
        return sympy2gurobi[(expr, data_idx)]

    # Handle inequalities (not used anymore)
    # if isinstance(expr, relational.Rel):
    #     return convertIneq2GurobiExpr(expr, sympy2gurobi, g_model)

    # Recursively convert Sympy expression to gurobi one
    if isinstance(expr, sympy.Number) and not isinstance(expr, sympy.core.numbers.NaN):
        return typeConverter[type(expr)](expr)

    elif isinstance(expr, sympy.core.numbers.NaN):
        return float('inf')

    elif isinstance(expr, sympy.Symbol):
        if g_model is None:
            raise ValueError

        var_str = str(expr) if data_idx is None else f'{str(expr)}__{data_idx}'
        if g_model.getVars():
            var = g_model.getVarByName(var_str)
        else:
            var = None
        if var is not None:
            return var

        if binary:
            var = g_model.addVar(name=var_str, vtype=GRB.BINARY)
        elif incl_bound:
            bound = g_model._var_to_bound.get(expr, (float('-inf'), float('inf')))
            lb, ub = bound
            var = g_model.addVar(lb=lb, ub=ub, name=var_str, vtype=GRB.CONTINUOUS)
        else:
            var = g_model.addVar(lb=float('-inf'), ub=float('inf'), name=var_str, vtype=GRB.CONTINUOUS)
        return var

    res = [convert2GurobiExpr(arg_i, g_model, data_idx=data_idx, binary=binary, incl_bound=incl_bound)
           for arg_i in expr.args]

    # Operation between args0 and args1 is either Add or Mul
    if isinstance(expr, sympy.Add):
        ret = quicksum(res)
    elif isinstance(expr, sympy.Mul):
        ret = 1
        for t in res:
            ret *= t
    else:
        raise NotImplementedError("Operation not recognized!")

    # Store in cache
    if data_idx is None:
        sympy2gurobi[expr] = ret
    else:
        sympy2gurobi[(expr, data_idx)] = ret
    return ret


def set_equality_constraint(var, rhs, g_model, incl_bound=False):
    if isinstance(var, sympy.Basic):
        g_var = convert2GurobiExpr(var, g_model, incl_bound=incl_bound)
    else:
        g_var = var

    if isinstance(rhs, sympy.Basic):
        rhs = convert2GurobiExpr(rhs, g_model, incl_bound=incl_bound)

    g_model.addConstr(g_var == rhs, f"Equality_constraint_for_{g_var._VarName}")


def check_var_anno_and_add_equality_constraint(
        var_to_anno,
        g_model,
        num_data,
        write=False,
        model_str=None,
        incl_bound=False,
):
    assert (write and model_str is not None) or (not write and model_str is None)
    anno_to_var = {}
    for v_i in var_to_anno.keys():
        v_id = var_to_anno[v_i]

        # If already added node id, add equality constraint 'v_earlier == v_i' to g_model
        if v_id in anno_to_var.keys():
            v_earlier = anno_to_var[v_id]

            # Gurobi variable corresponding to v_i: repeat for all data instances
            for n in range(num_data):
                w_i = g_model.getVarByName(f'{str(v_i)}__{n}')
                assert w_i is not None

                # Gurobi variable corresponding to v_earlier
                w_earlier = g_model.getVarByName(f'{str(v_earlier)}__{n}')
                assert w_earlier is not None

                # Set the equality constraint
                set_equality_constraint(w_earlier, w_i, g_model, incl_bound)
        else:
            # map node_id -> node
            anno_to_var[v_id] = v_i

    # Remove variables that have the same annotation yet show up later are removed from var_to_anno
    var_to_anno = {anno_to_var[v_id]: v_id for v_id in anno_to_var}
    return var_to_anno


"""
Functions used for writing a MILP model to a .lp file
"""


def configure_model_after_read(g_model, model_name, param_dim=None):
    g_model.ModelName = model_name
    g_model.setParam('OutputFlag', 0)
    g_model._best_obj = float('inf')
    g_model._time_log = 0

    if param_dim is not None:
        param_array = []
        if len(param_dim) == 1:
            for i in range(1, param_dim[0] + 1):
                param_array.append(g_model.getVarByName(f'theta{i}'))
        else:
            for i in range(1, param_dim[0] + 1):
                param_array.append([])
                for j in range(param_dim[1]):
                    param_array[i - 1].append(g_model.getVarByName(f'theta{i}{j}'))
        param_array = np.array(param_array)
        g_model.update()
        g_model._param_array = param_array


def write_infeasibility_checking(
        model_str: str,
        dec_partition: list,
        binary_set: set,
        constr_cache: set,
):
    """
    Enforces a set of constraints such that the infeasible partition is accounted for.
    For example, suppose we have `dec_partition=[d1, d2, d3]` whose function value is \infty.
    Then, it implies that `d1 and d2 => not d3` because if d1, d2, and d3 are all true, that leads to infeasibility.
    Encoding such a logical condition can be done by defining additional binary variables.
    In the previous example, if `i1, i2, i3` are the associated binary variables, respectively, then we enforce
        (1 - i1) + (1 - i2) >= i3
        Note that if (i1, i2) = (1, 1), then i3 = 0 should hold
        otherwise, i3 can have either 0 or 1... not constrained.
    On the other hand, if `dec_partition=[d1, not d2, d3]`, then
        (1 - i1) + i2 >= i3
        e.g., (i1, i2) = (1, 0) => i3 = 0
    and so on and so forth...
    Args:
          model_str (str)
          dec_partition (list)  The list containing decisions (conditionals) that have been encountered until reaching
            the current infeasible partition
          binary_set (set)  The set of indicator variables defined so far (Note: indicator variables associated with
            parent nodes should always have already defined and added to this set!)
          constr_cache (set)
    """
    # Decisions and their indicator variables up until the last decision
    ind_vars = []
    for dec in dec_partition:
        ind_v = f"ind_{dec if dec > 0 else -dec}"
        assert ind_v in binary_set, "Binary variables associated with parent nodes should have defined already!"
        ind_vars.append(ind_v)

    # Create the constraint
    constr_name = f"Feasibility_({'_'.join(map(lambda x: x.replace('-', 'n'), map(str, dec_partition[:-1])))})" \
                  f"_impl_({str(-dec_partition[-1]).replace('-', 'n')})"
    assert constr_name not in constr_cache

    constr = ''
    if len(dec_partition) == 1:
        constr += f"{ind_vars[0]} = {0 if dec_partition[0] > 0 else 1}"
    else:
        rhs = 0
        for i, (v, dec) in enumerate(zip(ind_vars, dec_partition)):
            if dec < 0:
                if i == len(dec_partition) - 1:
                    rhs += 1
                constr += f" + {v}" if i != 0 else v
            else:
                if i < len(dec_partition) - 1:
                    rhs -= 1
                constr += f" - {v}" if i != 0 else f'-{v}'
        constr += f" >= {rhs}"
        constr.strip().strip('+')
    constr_cache.add(constr_name)
    model_str += f' {constr_name}: {constr}\n'
    return model_str


def write_milp(
        model_str: str,
        context,
        node_id: int,
        node_id_to_var_str: dict,
        constr_cache: set,
        dec_partition: list,
        binary_set=None,
        continuous_set=None,
        binary=False,
        data=None,
        data_idx=None,
        **kwargs
):
    node = context.get_exist_node(node_id)

    if node._is_leaf:
        expr = node.expr

        # Handle infeasibility
        if expr == sympy.oo or expr == -sympy.oo:
            model_str = write_infeasibility_checking(model_str, dec_partition, binary_set, constr_cache)
        return expr, model_str

    dec = node.dec
    expr = context._id_to_expr[dec]

    lhs, rel, rhs = expr.lhs, type(expr), expr.rhs
    assert rhs == 0, 'RHS should always be 0'

    # Move the constant term (if there is) to RHS
    if not isinstance(lhs, sympy.Symbol):
        two_terms = lhs.as_two_terms()
        if isinstance(two_terms[0], sympy.core.Number):
            const, lhs = two_terms
            rhs = rhs - const

    # convert LHS expression to string
    expr_true = (lhs, relConverter[rel], rhs)
    expr_false = (lhs, REL_REVERSED_GUROBI[relConverter[rel]], rhs)

    indicator = f'ind_{dec}' if data_idx is None else f'ind_{dec}__{data_idx}'
    binary_set.add(indicator)

    var_node = node_id_to_var_str.get(node_id, None) if data_idx is None \
        else node_id_to_var_str.get((node_id, data_idx), None)
    if var_node is None:
        var_node = f'icvar_{node_id}' if data_idx is None else f'icvar_{node_id}__{data_idx}'
        if binary:
            binary_set.add(var_node)
        else:
            continuous_set.add(var_node)

    # Recursively write the MILP model
    dec_partition.append(-dec)
    low, model_str = write_milp(model_str, context, node._low, node_id_to_var_str, constr_cache,
                                dec_partition, binary_set, continuous_set, binary, data, data_idx,
                                **kwargs)
    dec_partition.remove(-dec)
    dec_partition.append(dec)
    high, model_str = write_milp(model_str, context, node._high, node_id_to_var_str, constr_cache,
                                 dec_partition, binary_set, continuous_set, binary, data, data_idx,
                                 **kwargs)
    dec_partition.remove(dec)

    kwargs['data_idx'] = data_idx
    kwargs['dec_id'] = f'n{dec}'
    model_str = check_redundancy_and_write_indicator_constraint(model_str, constr_cache, indicator, False, *expr_false,
                                                                **kwargs)
    if not (low == sympy.oo or low == -sympy.oo):
        model_str = check_redundancy_and_write_indicator_constraint(model_str, constr_cache, indicator, False, var_node,
                                                                    low, **kwargs)
    kwargs['dec_id'] = str(dec)
    model_str = check_redundancy_and_write_indicator_constraint(model_str, constr_cache, indicator, True, *expr_true,
                                                                **kwargs)
    if not (high == sympy.oo or high == -sympy.oo):
        model_str = check_redundancy_and_write_indicator_constraint(model_str, constr_cache, indicator, True, var_node,
                                                                    high, **kwargs)

    return var_node, model_str


def write_milp_from_bilevel(
        model_str: str,
        context,
        node_id: int,
        node_id_to_var_str: dict,
        constrCache: set,
        binary_set=None,
        continuous_set=None,
        binary=False,
        data=None,
        data_idx=None,
        **kwargs
):
    node = context.get_exist_node(node_id)

    if node._is_leaf:
        expr = node.expr
        # If leaf expression not a number, append each variable string with the data index if the index is given
        if not expr.is_number:
            expr_str = convert_sympy_expr_to_lp_string(expr, data_idx)
        else:
            expr_str = str(expr)
        return expr_str, constrCache, model_str

    dec = node.dec
    expr = context._id_to_expr[dec]

    lhs, rel, rhs = expr.lhs, type(expr), expr.rhs
    assert rhs == 0, 'RHS should always be 0'

    # convert LHS expression to string with data index
    lhs_expr_str = convert_sympy_expr_to_lp_string(lhs, idx=data_idx)
    expr_true = (lhs_expr_str, relConverter[rel], 0)
    expr_false = (lhs_expr_str, REL_REVERSED_GUROBI[relConverter[rel]], 0)

    indicator = f'ind_{dec}__{data_idx}'
    binary_set.add(indicator)

    var_node = node_id_to_var_str.get((node_id, data_idx), None)
    if var_node is None:
        var_node = f'icvar_{node_id}__{data_idx}'
        if binary:
            binary_set.add(var_node)
        else:
            continuous_set.add(var_node)

    # Recursively write the MILP model
    low, constrCache, model_str = write_milp_from_bilevel(model_str, context, node._low, node_id_to_var_str,
                                                          constrCache,
                                                          binary_set, continuous_set, binary, data, data_idx, **kwargs)
    high, constrCache, model_str = write_milp_from_bilevel(model_str, context, node._high, node_id_to_var_str,
                                                           constrCache,
                                                           binary_set, continuous_set, binary, data, data_idx, **kwargs)

    kwargs['data_idx'] = data_idx
    kwargs['dec_id'] = f'n{dec}'
    model_str = check_redundancy_and_write_indicator_constraint(model_str, constrCache, indicator, False, *expr_false,
                                                                **kwargs)
    model_str = check_redundancy_and_write_indicator_constraint(model_str, constrCache, indicator, False, var_node, low,
                                                                **kwargs)
    kwargs['dec_id'] = str(dec)
    model_str = check_redundancy_and_write_indicator_constraint(model_str, constrCache, indicator, True, *expr_true,
                                                                **kwargs)
    model_str = check_redundancy_and_write_indicator_constraint(model_str, constrCache, indicator, True, var_node, high,
                                                                **kwargs)

    return var_node, constrCache, model_str


def check_redundancy_and_write_indicator_constraint(model_str, constr_cache, indicator, boolean, *args, **kwargs):
    dec_id = kwargs.get('dec_id', None)
    assert dec_id is not None

    # Equality constraint
    if len(args) == 2:
        lhs, rhs = args
        if not isinstance(lhs, sympy.Basic):
            lhs = sympy.sympify(lhs)
        if not isinstance(rhs, sympy.Basic):
            rhs = sympy.sympify(rhs)

        if not rhs.is_number:
            rhs_str = convert_sympy_expr_to_lp_string(rhs, idx=kwargs.get('data_idx', None))
        else:
            assert rhs != sympy.oo
            rhs_str = str(rhs)
        lhs_str = convert_sympy_expr_to_lp_string(lhs, idx=kwargs.get('data_idx', None))

        constr_name = f'GC_({indicator})_({boolean})_({lhs_str})_equalto_({rhs_str[:20]})'.replace(' ', '')
        if constr_name not in constr_cache:
            constr = f'{indicator} = {1 if boolean else 0} -> '
            try:
                float(rhs)
                constr += f'{lhs_str} = {rhs_str}'
            except TypeError:
                lhs, rhs = lhs - rhs, 0
                two_terms = lhs.as_two_terms()
                if two_terms[0].is_number:
                    const, lhs = two_terms
                    rhs = - const
                constr += f"{convert_sympy_expr_to_lp_string(lhs, idx=kwargs.get('data_idx', None))} = {rhs}"
            constr = constr.replace('- -', '+ ')
            constr_cache.add(constr_name)
            model_str += f' {constr_name}: {constr}\n'

    # Inequality constraint
    elif len(args) == 3:
        lhs, rel, rhs = args
        if not isinstance(lhs, sympy.Basic):
            lhs = sympy.sympify(lhs)
        if not isinstance(rhs, sympy.Basic):
            rhs = sympy.sympify(rhs)

        constr_name = f'GC_({indicator})_({boolean})_({dec_id})'
        if constr_name not in constr_cache:
            # Handle the case when the decision holds in equality.. Note: RHS is always set to zero prior to this part
            if 'epsilon' in kwargs:
                epsilon = kwargs['epsilon']
            else:
                epsilon = EPSILON

            if rel == '<':
                rhs = rhs - epsilon
            else:
                rhs = rhs + epsilon

            rhs_str = convert_sympy_expr_to_lp_string(rhs, idx=kwargs.get('data_idx', None))
            lhs_str = convert_sympy_expr_to_lp_string(lhs, idx=kwargs.get('data_idx', None))
            constr = f'{indicator} = {1 if boolean else 0} -> {lhs_str} {rel} {rhs_str}'
            constr_cache.add(constr_name)
            model_str += f' {constr_name}: {constr}\n'
    return model_str


def write_losses(model_str: str, num_data: int = 0, loss_var: str = 'loss'):
    if num_data != 0:
        # One loss variable per data point
        model_str += 'Minimize\n' + \
                     '  ' + ' + '.join([f'{loss_var}_{i}' for i in range(num_data)])
    else:
        model_str += 'Minimize\n' + f'  {loss_var}'

    # Followed by the constraint section
    model_str += '\nSubject To\n'
    return model_str


def convert_sympy_expr_to_lp_string(expr, idx: int = None):
    """
    Given a sympy expression, convert it to a string in LP file format.
    Optionally, data indices can be appended to all occasions of the variable strings in the expression.
    """
    expr_str = str(expr.evalf()).replace('*', ' ')  # Replace '*' with a whitespace

    def is_number(token):
        try:
            float(token.strip())
            return True
        except ValueError:
            return False

    def append_index_if_var(token):
        if token in {'+', '-'}:
            return token
        elif is_number(token):
            return token
        else:
            return token + f'__{idx}'

    if idx is not None:
        expr_str_lst = expr_str.split()
        expr_str = ' '.join(list(map(append_index_if_var, expr_str_lst))).replace(' + -', ' - ')
    else:
        expr_str = expr_str.replace(' + -', ' - ')
    return expr_str


def write_equality_constraints(
        model_str: str,
        eq_constr_dict: dict,
        bound_str: str,
        var_to_bound: dict,
        var_to_anno: dict = None,
        idx: int = None,
):
    for v, rhs in eq_constr_dict.items():
        bound = var_to_bound.get(v, None)
        assert bound is not None
        x = f'{str(v)}__{idx}' if idx is not None else str(v)

        # Get constant in RHS
        const = [c for c in rhs.args if not c.free_symbols]
        assert len(const) <= 1
        if len(const) == 0:
            const = '0'
        else:
            rhs = rhs - const[0]
            const = f'{-const[0]}'

        rhs_str = convert_sympy_expr_to_lp_string(expr=rhs, idx=idx)
        eqconstr_name = f'Equality_constraint_for_{x}'
        eqconstr_str = f'{eqconstr_name}: {rhs_str} - {x} = {const}'
        model_str += f' {eqconstr_str}\n'
        bound_str = write_bound(bound_str, x, bound)
        if var_to_anno is not None:
            var_to_anno.pop(v)
    return model_str, bound_str


def write_substitution_into_z(model_str: str, theta_times_x, z_vec, i: int):
    for j in range(len(z_vec)):
        theta_times_x_str = theta_times_x[j]
        constr_name = f'z_{j + 1}__{i}_eq_theta_x'
        constr = f'{theta_times_x_str} - {z_vec[j]} = 0'
        model_str += f' {constr_name}: {constr}\n'
    return model_str


def modify_bounds(bound):
    lb, ub = bound
    try:
        if float(int(lb)) == lb:
            lb = int(lb)
    except:
        pass
    try:
        if float(int(ub)) == ub:
            ub = int(ub)
    except:
        pass
    lb, ub = str(lb), str(ub)
    if ub == 'oo':
        ub = 'inf'
    if lb == '-oo':
        lb = '-inf'
    return lb, ub


def write_bound(bound_str, cont_var, bound=None):
    if bound is None:
        lb, ub = '-inf', 'inf'
    else:
        lb, ub = modify_bounds(bound)

    if lb == '-inf' and ub == 'inf':
        bound_str += f' {cont_var} free\n'
    elif lb != '0' and ub == 'inf':
        bound_str += f' {lb} <= {cont_var}\n'
    elif lb != '0' and ub != 'inf':
        bound_str += f' {lb} <= {cont_var} <= {ub}\n'
    elif lb == '0' and ub != 'inf':
        bound_str += f' {cont_var} <= {ub}\n'
    return bound_str


def return_model_info(g_model):
    g_model.update()
    vars_lst = g_model.getVars()
    num_cvar = 0
    num_bvar = 0
    for v in vars_lst:
        if v.VType == GRB.BINARY:
            num_bvar += 1
        else:
            num_cvar += 1

    num_constrs = g_model.num_constrs
    return num_cvar, num_bvar, num_constrs