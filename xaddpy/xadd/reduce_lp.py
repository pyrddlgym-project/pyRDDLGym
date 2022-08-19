import sympy.core.relational as relational
from xaddpy.utils.global_vars import REL_TYPE, REL_NEGATED
from xaddpy.utils.milp_encoding import convert2GurobiExpr
from xaddpy.utils.logger import logger

default_check_redundancy = True


class ReduceLPContext:
    def __init__(self, context, **kwargs):
        """
        :param xadd:
        """
        self.LPContext = context
        self.set_to_implications = {}
        self.set_to_nonimplications = {}
        self.local_reduce_lp = None
        self.kwargs = kwargs

    def reduce_lp(self, node_id, redun=None):
        if redun is None:
            redun = default_check_redundancy

        if self.local_reduce_lp is None:
            self.local_reduce_lp = LocalReduceLP(context=self.LPContext, reduce_lp_context=self, **self.kwargs)

        return self.local_reduce_lp.reduce_lp(node_id, redun)

    def flush_caches(self):
        self.flush_implications()
        try:
            self.local_reduce_lp.flush_caches()
        except AttributeError as e:
            logger.error(e)
            pass

    def flush_implications(self):
        """
        public void flushImplications() {
        _mlImplications.clear();
        _mlNonImplications.clear();
        _mlImplicationsChild.clear();
        _mlIntermediate.clear();
        _hmIntermediate.clear();
        _hmImplications.clear();
        _hmNonImplications.clear();
        """
        self.set_to_implications.clear()
        self.set_to_nonimplications.clear()


class LocalReduceLP:
    def __init__(self, context, reduce_lp_context, **kwargs):
        """
        :param xadd:            (XADD)
        :param reduce_lp_context: (ReduceLPContext)
        """
        # super().__init__(localRoot, xadd)
        self._context = context
        self.reduce_lp_context = reduce_lp_context
        self.lp = None
        self.verbose = kwargs.get('verbose', False)

    def flush_caches(self):
        self.lp._lhs_expr_to_gurobi.clear()
        self.lp._sympy_to_gurobi.clear()

    def reduce_lp_v2(self, node_id, test_dec, redundancy):
        """
        :param node_id:             (int)
        :param test_dec:            (set)
        :param redundancy:          (bool)
        :return:
        """
        avail_mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        if avail_mem < 10:
            logger.info('freeing up cache of reduce_lp')
            self.reduce_lp_context.flush_caches()

        node = self.context.get_exist_node(node_id)

        # A leaf node should be reduced (and cannot be restricted) by default if hashing and equality testing
        # are working in getTNode
        if node._is_leaf:
            return node_id

        # Full branch implication test
        # If `node.dec` is implied by `test_dec`, then replace `node` with `node._high`
        if self.is_test_implied(test_dec, node.dec):
            return self.reduce_lp_v2(node._high, test_dec, redundancy)
        # If the negation of `node.dec` is implied by `test_dec`, then replace `node` with `node._low`
        elif self.is_test_implied(test_dec, -1 * node.dec):
            return self.reduce_lp_v2(node._low, test_dec, redundancy)

        # Make subtree reduced before redundancy check
        test_dec.add(-1 * node.dec)
        low = self.reduce_lp_v2(node._low, test_dec, redundancy)
        try:
            test_dec.remove(-1 * node.dec)
        except KeyError as ke:
            logger.error(ke)
            pass

        test_dec.add(node.dec)
        high = self.reduce_lp_v2(node._high, test_dec, redundancy)
        test_dec.remove(node.dec)

        # After reducing subtrees, check if this node became redundant
        if redundancy:
            # 1) check if true branch is implied in the low branch if current decision is true
            test_dec.add(node.dec)
            lowReplace = self.is_result_implied(test_dec, low, high)
            test_dec.remove(node.dec)

            if lowReplace: return low

            # 2) check if false branch is implied in the true branch if current decision is false
            test_dec.add(-node.dec)
            highReplace = self.is_result_implied(test_dec, high, low)
            test_dec.remove(-node.dec)

            if highReplace: return high

        # Standard reduce: getINode will handle the case of low == high
        return self._context.get_internal_node(node.dec, low, high)

    def reduce_lp(self, node_id, redundancy):
        test_dec = set()
        node_id = self.reduce_lp_v2(node_id, test_dec, redundancy)
        return node_id

    def is_test_implied(self, test_dec, dec):
        """
        :param test_dec:    (set)
        :param dec:         (int)
        :return:
        """
        # When 'dec' is not Rel expression, simply return False
        if not isinstance(self.context._id_to_expr.get(abs(dec)), relational.Rel):
            logger.warning(f"Warning: This case is not expected! (is_test_implied) dec: {dec} "
                           f"expr_dec: {self.context._id_to_expr[abs(dec)]}")
            return False

        impliedSet = self.reduce_lp_context.set_to_implications.get(frozenset(test_dec.copy()), None)
        if impliedSet is not None and dec in impliedSet:
            # When dec can easily be checked as implied (using impliedSet)
            return True
        non_implied_set = self.reduce_lp_context.set_to_nonimplications.get(frozenset(test_dec.copy()), None)
        if non_implied_set is not None and dec in non_implied_set:
            return False

        test_dec.add(
            -dec)  # If adding the negation of `dec` to `test_dec` makes it infeasible, then `test_dec` implies `dec`
        implied = self.is_infeasible(test_dec)
        test_dec.remove(-dec)
        if implied:
            if impliedSet is None:
                impliedSet = set()
                self.reduce_lp_context.set_to_implications[frozenset(test_dec.copy())] = impliedSet
            impliedSet.add(dec)
        else:
            if non_implied_set is None:
                non_implied_set = set()
                self.reduce_lp_context.set_to_nonimplications[frozenset(test_dec.copy())] = non_implied_set
            non_implied_set.add(dec)
        return implied

    def is_infeasible(self, test_dec):
        """
        Check whether a set of decisions contained in test_dec is infeasible.
        :param test_dec:        (set)
        :return:                (bool)
        """
        infeasible = False

        # Based on decisions, solve an LP with those constraints, and determine if feasible or infeasible
        # Note: some bilinear constraints can show up, but they should be infeasible.
        # In order to check that, we use try; except.
        self.lp = LP(self.context) if not self.lp else self.lp
        lp = self.lp

        # Remove all previously set constraints and reset the objective
        lp.clear_all_constraints()
        lp.set_objective(1)
        lp.update()

        # Add constraints as given by decisions in test_dec
        for dec in test_dec:
            lp.add_decision(dec)
        lp.update()

        # Optimize the model to see if infeasible
        if len(lp.model.getQConstrs()) > 0:
            lp.non_convex_on()
            try:
                status = lp.solve()
                lp.non_convex_off()
            except Exception as e:
                logger.error(e)
                exit(1)

        else:
            lp.non_convex_off()
            try:
                status = lp.solve()
            except Exception as e:
                logger.error(e)
                exit(1)

        # except gurobipy.GurobiError as e:
        #     if self.verbose:
        #         logger.info(f"Bilinear constraint detected.... Turning on the 'NonConvex' flag and resolve!")
        #     lp.non_convex_on()
        #     status = lp.solve()

        if status == GRB.INFEASIBLE:
            infeasible = True
        if infeasible:
            lp.model.remove(lp.model.getQConstrs())
            return infeasible

        ## Test 2: test slack
        # lp2 = LP(self.context)
        # Remove bilinear constraints right away if added
        lp.model.remove(lp.model.getQConstrs())

        infeasible = lp.test_slack(test_dec)
        return infeasible

    def is_result_implied(self, test_dec, subtree, goal):
        """
        :param test_dec:    (set)
        :param subtree:     (int)
        :param goal:        (int)
        :return:
        """
        if subtree == goal:
            return True
        subtree_node = self.context.get_exist_node(subtree)
        goal_node = self.context.get_exist_node(goal)

        if not subtree_node._is_leaf:
            if not goal_node._is_leaf:
                # use variable ordering to stop pointless searches
                if subtree_node.dec >= goal_node.dec:
                    return False

            # If decisions down to the current node imply the negation of the `subtree_node.dec`:
            if self.is_test_implied(test_dec, -subtree_node.dec):
                return self.is_result_implied(test_dec, subtree_node._low, goal)  # Then, check for the low branch
            if self.is_test_implied(test_dec, subtree_node.dec):  # Or they imply `subtree_node.dec`,
                return self.is_result_implied(test_dec, subtree_node._high, goal)  # Then, check for the high branch

            # Now, recurse starting from the low branch
            test_dec.add(-subtree_node.dec)
            implied_in_low = self.is_result_implied(test_dec, subtree_node._low, goal)
            test_dec.remove(-subtree_node.dec)

            # if one branch failed, no need to test the other one
            if not implied_in_low: return False

            test_dec.add(subtree_node.dec)
            implied_in_high = self.is_result_implied(test_dec, subtree_node._high, goal)
            test_dec.remove(subtree_node.dec)

            return implied_in_high
        return False  # If XADDTNode, '==' check can make it True

    @property
    def context(self):
        return self._context


class LP:
    def __init__(self, context):
        self._context = context
        self.model = Model(name='LPReduce')
        self.model.setParam('OutputFlag', 0)
        self.model.setObjective(1, sense=GRB.MAXIMIZE)  # Any objective suffices as we only check for feasibility
        self.model.setAttr('_var_to_bound', context._var_to_bound)
        self._is_nonconvex = False
        # Variable management
        self._var_set = set()
        self.model.set_sympy_to_gurobi_dict(self.context._sympy_to_gurobi)
        self._sympy_to_gurobi = self.context._sympy_to_gurobi
        self._lhs_expr_to_gurobi = {}

    @property
    def context(self):
        return self._context

    def clear_all_constraints(self):
        self.model.remove(self.model.getConstrs())

    def add_decision(self, dec):
        # Check if the constraint has already been added to the model
        if len(self.model.getConstrs()) != 0:
            if self.model.getConstrByName('dec({})'.format(dec)):
                return
        if dec > 0:
            self.add_constraint(dec, True)
        else:
            self.add_constraint(-dec, False)

    def add_constraint(self, dec, is_true):
        """
        Given an integer id for decision expression (and whether it's true or false), add the expression to LP problem.
        1) Need to create Variable objects for each of sympy variables (if already created, retrieve from cache)
        2) Need to convert sympy dec_expr to optlang Constraint format
            e.g. c1 = Constraint(x1 + x2 + x3, ub=10)
                 for x1 + x2 + x3 <= 10
        :param dec:         (int)
        :param is_true:     (bool)
        """
        dec_expr = self.context._id_to_expr[dec]
        dec = dec if is_true else -dec
        lhs, rel, rhs = dec_expr.lhs, REL_TYPE[type(dec_expr)], dec_expr.rhs
        if not is_true:
            rel = REL_NEGATED[rel]

        # dec_expr = self.convert_expr(dec_expr)       # Convert to optlang expression
        assert rhs == 0, "RHS of a relational expression should always be 0 by construction!"
        lhs_gurobi = self.convert_expr(lhs)  # Convert lhs to gurobi expression (rhs=0)

        if rel == '>' or rel == '>=':
            self.model.addConstr(lhs_gurobi >= 0, name=f'dec({dec})')
        elif rel == '<' or rel == '<=':
            self.model.addConstr(lhs_gurobi <= 0, name=f'dec({dec})')

        # Skip expressions containing theta variables
        if dec_expr is None:
            return

        # lhs, rhs, rel = dec_expr.lhs, dec_expr.rhs, REL_TYPE[type(dec_expr)]
        # lhs, rhs, rel = dec_expr
        # if not is_true: rel = REL_NEGATED[rel]
        # if not is_true: rel = REL_REVERSED_GUROBI[rel]

        # Create Constraint
        # if rel == '<=' or rel == '<':
        #     c = Constraint(lhs-rhs, ub=0)
        # else:
        #     c = Constraint(lhs-rhs, lb=0)

        # Add Constraint
        # self.lp.addConstr(c)
        # self.model.addConstr(dec_expr, name='dec({})'.format(dec))

    def update(self):
        self.model.update()

    def solve(self):
        """
        Solve the LP defined by all added constraints, variables and objective. We only care about if it's
        infeasible or feasible. So, return status.
        :return:        (str)
        """
        try:
            self.model.optimize()
            return self.model.status
        except GurobiError as e:
            raise e

    def set_objective(self, obj):
        self.model.setObjective(obj, sense=GRB.MAXIMIZE)

    def add_variable(self, var):
        pass

    def non_convex_on(self):
        self._is_nonconvex = True
        self.model.setParam('nonConvex', 2)

    def non_convex_off(self):
        self._is_nonconvex = False
        self.model.setParam('nonConvex', 0)

    def convert_expr(self, expr):
        """
        Given a sympy expression 'expr', return the optlang expression which contains optlang.symbolics variables
        instead of sympy.Symbol variables.
        :param expr:        (sympy.Basic)
        :return:
        """
        if expr in self._lhs_expr_to_gurobi:
            return self._lhs_expr_to_gurobi[expr]
        else:
            gurobi_expr = convert2GurobiExpr(
                expr,
                g_model=self.model,
                incl_bound=True,
            )
            self._lhs_expr_to_gurobi[expr] = gurobi_expr
            return gurobi_expr

    def test_slack(self, test_dec):
        """
        In this test, check the value of slack variable, S.
        For each constraint a.T @ w + b >= 0, the slack is the greatest value S > 0 s.t. a.T @ w + b - S >= 0
        For each constraint a.T @ w + b <= 0, the slack is the greatest value S > 0 s.t. a.T @ w + b + S <= 0
        """
        infeasible = False

        # Check if test_slack is turned off by context (this happens at the time of arg substitution)
        if not self.context._prune_equality:
            return infeasible

        # Define a positive slack variable
        if len(self.model.getVars()) != 0:
            S = self.model.getVarByName('S')
            if S is None:
                S = self.model.addVar(lb=0, name='S')
        else:
            S = self.model.addVar(lb=0, name='S')

        # obj = Objective(S, direction='max')         # objective is S
        self.model.setObjective(S, sense=GRB.MAXIMIZE)

        # Remove all pre-existing constraints
        try:
            self.model.remove(self.model.getConstrs())
            # self.model._remove_constraints(self.model.constraints)
        except RuntimeError as re:
            logger.error(re)
        except AttributeError as ae:
            logger.error(ae)
            pass

        # Reset constraint for each decision in test_dec
        constraints = []
        for dec in test_dec:
            negated = True if dec < 0 else False

            dec_expr = self.context._id_to_expr[-dec] if negated else self.context._id_to_expr[dec]
            lhs, rhs, rel = dec_expr.lhs, dec_expr.rhs, REL_TYPE[type(dec_expr)]
            lhs_gurobi = self.convert_expr(lhs)

            if negated: rel = REL_NEGATED[rel]

            # Create Constraint
            if rel == '<=' or rel == '<':
                # c = Constraint(lhs - rhs + S, ub=0)
                self.model.addConstr(lhs_gurobi + S <= 0, name=f'dec({dec})')
            else:
                # c = Constraint(lhs - rhs - S, lb=0)
                self.model.addConstr(lhs_gurobi - S >= 0, name=f'dec({dec})')
            # constraints.append(c)
        # self.model.add(constraints)

        self.update()
        # Optimize the model to see if infeasible
        if len(self.model.getQConstrs()) > 0:
            self.non_convex_on()
            try:
                status = self.solve()
                self.non_convex_off()
            except Exception as e:
                logger.error(e)
                exit(1)

        else:
            self.non_convex_off()
            try:
                status = self.solve()
            except Exception as e:
                logger.error(e)
                exit(1)

        # try:
        #     self.non_convex_off()
        #     status = self.solve()
        # except gurobipy.GurobiError as e:
        #     logger.info(f"Bilinear constraint detected.... Turning on the 'NonConvex' flag and resolve!")
        #     self.non_convex_on()
        #     status = self.solve()
        # except Exception as e:
        #     logger.error(e)
        #     exit(1)

        # Due to dual reduction in presolve of Gurobi, INF_OR_UNBD status can be returned
        # Turn off the functionality and resolve
        if status == GRB.INF_OR_UNBD:
            self.model.setParam('DualReductions', 0)
            status = self.solve()
            self.model.setParam('DualReductions', 1)

        opt_obj = self.model.objVal if status == GRB.OPTIMAL else 1e100

        if status == GRB.INFEASIBLE:
            logger.warning("Infeasibility at test 2 should not have occurred!")
            infeasible = True
        elif status != GRB.UNBOUNDED and opt_obj < 1e-4:
            infeasible = True

        # Remove bilinear constraints right away if there are (since these are not returned by getConstrByName)
        self.model.remove(self.model.getQConstrs())

        return