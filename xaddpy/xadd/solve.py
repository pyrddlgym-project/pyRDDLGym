from xaddpy.xadd import xadd
from xaddpy.models.emspo import EMSPO
from xaddpy.utils import milp_encoding
from xaddpy.utils.gurobi_util import BaseGurobiModel, BFGurobiModel, callback_spo, callback_cbcuts, callback
import gurobipy as gp
from gurobipy import read, GRB, LinExpr, quicksum
import numpy as np
import os.path as path
import os
import time
from xaddpy.utils.logger import logger
from typing import List


def solve_predopt(
        context: xadd.XADD,
        g_model: BaseGurobiModel,
        emspo: EMSPO,
        dataset: dict,
        eq_constr_dict: dict,
        domain: str,
        var_name_rule=None,
        epsilon=0.0,
        verbose=False,
        timeout=None,
        time_interval=0,
        prob_config=None,
        args=None,
):
    """
    Solves the predict-then-optimize problem defined in `context` and `eq_constr_dict` with data instances given
    in `dataset`.
    """
    X = dataset['train'][0]
    cost = dataset['train'][1]

    # Get the argmin solutions of all decision variables
    var_to_anno = context.get_all_anno()

    # Get the sorted list of decision variables
    if var_name_rule is not None:
        variables = list(sorted(var_to_anno.keys(), key=var_name_rule))
    else:
        variables = list(var_to_anno.keys())

    # Put data into the MILP model
    logger.info('Putting data into MILP model')
    build_predopt_milp(X, cost, var_to_anno, variables, context, emspo, g_model, eq_constr_dict,
                       epsilon, domain, prob_config, args)

    logger.info('Solve the MILP model')

    # Optimize the model
    g_model.setAttr('_time_interval', time_interval)
    g_model.setParam('TimeLimit', timeout)      # Timeout after pre-defined time
    # g_model.setParam('IntFeasTol', epsilon / 10)
    if verbose:
        gurobi_log_file = f'{args.log_dir}/{args.model_name}_{args.date_time}_gurobi.log'
        logger.info(f'Gurobi solver outputs will be logged in {gurobi_log_file}')
        log_dir = path.join(path.curdir, gurobi_log_file)
        g_model.setParam('OutputFlag', 1)
        g_model.setParam('LogFile', log_dir)
        g_model.setParam('LogToConsole', 0)

    # Optimize the model with callbacks specified
    g_model.optimize(callback_spo(extracb=callback_cbcuts if g_model.scheme == 'benders' else None))

    logger.info(f'Done solving the MILP model: status = {g_model.status}')
    var_sol = None
    try:
        dec_sol, var_sol = g_model._subproblem._incumbent_sol
    except:
        pass

    return emspo.handle_solutions(verbose)


def build_predopt_milp(
        X,
        cost,
        var_to_anno: dict,
        variables: list,
        context: xadd.XADD,
        emspo: EMSPO,
        g_model: BaseGurobiModel,
        eq_constr_dict: dict,
        epsilon: float,
        domain: str,
        prob_config=None,
        args=None,
):
    assert domain != None
    obj = 0

    """
    # Create gurobi decision variables corresponding to each data instance.
    # For BFGurobiModel, it is necessary to keep variable bounds.
    # For CBGurobiModel, we discard such information as it has been taken care of during symbolic optimization 
    # (hence, all leaf values comply to the bounds) 
    """
    for v in var_to_anno:
        lb, ub = context._var_to_bound[v]
        is_binary = v in context._bool_var_set

        for i in range(len(X)):
            x_i = g_model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f'{str(v)}__{i}')
            g_model.sympy_to_gurobi[(v, i)] = x_i

    # If equality constraints exist, set those constraints first. Then, remove the variable from `var_to_anno`
    for v, rhs in eq_constr_dict.items():
        for i in range(len(X)):
            x_i = g_model.getVarByName(f'{str(v)}__{i}')
            rhs_i = milp_encoding.convert2GurobiExpr(
                rhs, g_model, data_idx=i, incl_bound=True,
            )
            milp_encoding.set_equality_constraint(x_i, rhs_i, g_model, incl_bound=True)
        var_to_anno.pop(v)

    # If some argmin solution happen to be the same as that of another variable, set them as equality constraints
    milp_encoding.check_var_anno_and_add_equality_constraint(
        var_to_anno, g_model, num_data=len(X), incl_bound=True,
    )

    # Iterate through all decision variables and their argmin XADDs to build a corresponding MILP variables/constraints
    for v, v_id in var_to_anno.items():
        lb, ub = context._var_to_bound[v]

        # Add the root node first (already generated above)
        for i in range(len(X)):
            x_i = g_model.getVarByName(f'{str(v)}__{i}')
            assert x_i is not None
            icvar_xi = g_model.addVar(lb=lb, ub=ub, name=f'icvar_{v_id}__{i}', vtype=GRB.CONTINUOUS)
            g_model.addConstr(x_i == icvar_xi, name=f'{str(v)}__{i}__eq__icvar_{v_id}__{i}')
            # node_id_to_gurobi_var[(v_id, i)] = x_i
            # g_model._var_cache[f'icvar_{v_id}__{i}'] = x_i      # another mapping to the root node gurobi variable

        # Build
        _ = milp_encoding.build_milp_from_bilevel(
            context,
            v_id,
            g_model,
            lb=lb,
            ub=ub,
            dec_partition=[],
            num_data=len(X),
            epsilon=epsilon,
        )

    emspo.subst_data_to_milp(X, cost, variables)

    dir_path = path.join(args.results_dir, "lp_files")
    os.makedirs(dir_path, exist_ok=True)
    g_model.write(f'{dir_path}/{g_model.ModelName}.lp')