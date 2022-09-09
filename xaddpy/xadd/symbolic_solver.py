from xaddpy.xadd import xadd_parse_utils
from xaddpy.xadd.xadd import XADD
from time import time
from xaddpy.utils.logger import logger

from typing import Optional


def solve_n_record_min_or_argmin(
        get_solution: bool,
        model_name: str,
        solver_type: int = 0,
        save_xadd: bool = False,
        verbose=False,
        fname_xadd: str = None,
        prob_instance: Optional[dict] = None,
        args: dict = None,
):
    context = XADD(args)
    fname_json = fname_xadd.replace('.xadd', '.json').replace('_argmin', '')

    # param_to_gurobi_var contains gurobi variables corresponding to learnable parameters
    variables, math_program, eq_constr_dict = \
        xadd_parse_utils.build_canonical_xadd_from_file(context, fname_json, model_name, prob_instance=prob_instance)

    # # iteratively min out next variables
    vars_to_optimize = variables['min_var_list']
    logger.info("Start symbolic optimization...")
    if verbose:
        logger.info("The objective: \n{}\n".format(context.get_repr(math_program)))
        logger.info(f"variables to optimize: {vars_to_optimize}; solver type: {solver_type}")

    if args.get('var_reorder', None) != None:
        var_count = context.var_count(math_program)
        reverse = False if args['var_reorder'] == 1 else True
        vars_to_optimize = sorted(vars_to_optimize, key=lambda x: var_count[x], reverse=reverse)
        logger.info(f"Reordered variables to optimize: {vars_to_optimize}; solver type: {solver_type}")

    stime = time()

    # `solver_type` == 0: min out all of one set of variables from each partition; casemin the results at the end
    # Not sure how to retrieve argmin in this case... May be impossible
    if solver_type == 0:
        result = context.min_or_max_multi_var(math_program, vars_to_optimize, is_min=True)
    # `solver_type` == 0: min out one variable from all partitions, then take casemins; repeat this for other vars
    elif solver_type == 1:
        for v in vars_to_optimize:
            logger.info(f"\tEliminate variable {str(v)}")
            result = context.min_or_max_var(math_program, v, is_min=True)
            if get_solution:
                context.reduced_arg_min_or_max(result, v)
            math_program = result

    etime = time()
    time_taken = etime - stime
    logger.info("...done!")

    # if verbose:
    #     logger.info(f"Optimal objective: {context.get_repr(result)}")

    # Get the argmin over all variables by sequential substitution
    if get_solution:
        var_to_anno = context.argmin_or_max(vars_to_optimize, resubstitution=False)
        if verbose:
            for v, anno in var_to_anno.items():
                logger.info("{}: {}".format(str(v), context.get_repr(anno)))

    # Export the XADDs
    if get_solution:
        if save_xadd:
            context.export_argmin_or_max_xadd(fname_xadd)
            context.export_min_or_max_xadd(fname_xadd.replace('_argmin.xadd', '.xadd'), result)
        return context, eq_constr_dict, time_taken
    else:
        if save_xadd:
            context.export_min_or_max_xadd(fname_xadd, result)
        context.set_objective(result)
        return context, eq_constr_dict, time_taken