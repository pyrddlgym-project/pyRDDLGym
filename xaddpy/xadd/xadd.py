import sympy
import sympy.core.numbers as numbers
import sympy.core.relational as relational
from sympy import oo, S

import xaddpy.utils.util
from xaddpy.xadd.node import XADDINode, XADDTNode
from xaddpy.xadd.reduce_lp import ReduceLPContext
import abc
from tqdm import tqdm
from xaddpy.utils.global_vars import REL_TYPE, OP_TYPE
from xaddpy.utils.util import compute_rref_filter_eq_constr
from xaddpy.xadd.xadd_parse_utils import parse_xadd_grammar
from xaddpy.utils.logger import logger

from typing import Union

USE_APPLY_GET_INODE_CANON = False
LARGE_INTEGER = 10000


class XADD:
    def __init__(self, args: dict = {}):
        # XADD variable maintenance
        self._cvar_to_id = {}
        self._id_to_cvar = {}
        self._var_set = set()
        self._bool_var_set = set()
        self._str_var_to_var = {}
        self._pred_coeffs = set()
        self._pred_coeff_vec = None

        self._sympy_to_gurobi = {}
        self._opt_var = None
        self._opt_var_lst = None
        self._paramSet = set()
        self._eliminated_var = []
        self._decisionVars = set()
        self._min_var_set = set()
        self._free_var_set = set()
        self._name_space = None

        # Bound maintenance (need to be passed from the output of parser function)
        self._var_to_bound = {}
        self._temp_ub_lb_cache = set()

        # Decision expression maintenance
        self._expr_to_id = {}
        self._id_to_expr = {}

        # XADD node maintenance
        self._id_to_node = {}
        self._node_to_id = {}
        self._var_to_anno = {}  # annotation dictionary for argmin / argmax

        # Flush
        self._special_nodes = set()
        self._node_to_id_new = {}
        self._id_to_node_new = {}
        self._id_to_expr_new = {}
        self._expr_to_id_new = {}

        # Reduce & Apply caches
        self._reduce_cache = {}
        self._reduce_leafop_cache = {}
        self._reduce_canon_cache = {}
        # self._reduce_annotate_cache = {}
        self._apply_cache = {}
        self._apply_caches = {}
        self._inode_to_vars = {}
        self._factor_cache = {}

        # Reduce LP
        self.RLPContext = ReduceLPContext(self, **args)

        # Node maintenance
        self._nodeCounter = 0

        # temporary nodes
        self._tempINode = XADDINode(-1, -1, -1, context=self)
        self._tempTNode = XADDTNode(sympy.S(-1), context=self)

        # Ensure that the 0th decision ID is invalid
        null = NullDec()
        self._id_to_expr[0] = null
        self._expr_to_id[null] = 0

        # Create standard nodes
        self.create_standard_nodes()

        # Solving min or max problem
        self._is_min = None

        # How to handle decisions that hold with equality
        self._prune_equality = True

        # Do or do not reorder XADD after substitution
        self._direct_substitution = False

        # Set the problem type
        self._problem_type = ''

        # Node id corresponding to the objective function
        self._obj = None
        self._additive_obj = False

        # Other attributes
        self._feature_dim: int = None
        self._param_dim: int = None
        self._param_gurobi_var = None
        self._args: dict = args

    def create_standard_nodes(self):
        """
        Create and store standard nodes and generate indices, which can be frequently used.
        """
        self.ZERO = self.get_leaf_node(sympy.S(0))
        self.ONE = self.get_leaf_node(sympy.S(1))
        self.oo = self.get_leaf_node(oo)
        self.NEG_oo = self.get_leaf_node(-oo)
        self.NAN = self.get_leaf_node(sympy.nan)

        # Create the special zero node to be used to track annotations for bilinear program (ig stands for ignore)
        zero_ig_node = XADDTNode(sympy.S(0), context=self, annotation=sympy.nan)
        self.ZERO_ig = self._nodeCounter
        self._id_to_node[self.ZERO_ig] = zero_ig_node
        self._node_to_id[zero_ig_node] = self.ZERO_ig
        self._nodeCounter += 1

    def add_eliminated_var(self, var):
        self._eliminated_var.append(var)

    def set_problem(self, prob):
        self._problem_type = prob

    def set_feature_dim(self, dim):
        self._feature_dim = dim

    def set_param_dim(self, dim):
        self._param_dim = dim

    def create_symbolic_coeffs(self, locals, coeff_symbol='c'):
        coeffs_lst = []
        dim = len(self._min_var_set)
        for i in range(dim):
            coeff_i = sympy.symbols(f'{coeff_symbol}{i + 1}')
            self._pred_coeffs.add(coeff_i)
            locals[str(coeff_i)] = coeff_i
            coeffs_lst.append(coeff_i)
        self._pred_coeff_vec = sympy.Matrix(coeffs_lst)
        return locals

    def link_json_file(self, config_json_fname):
        self._config_json_fname = config_json_fname

    def update_bounds(self, bound_dict):
        self._var_to_bound.update(bound_dict)

    def update_decision_vars(self, bvariables, cvariables0, cvariables1, min_var_set_id):
        variables = list(bvariables) + list(cvariables0) + list(cvariables1)
        self._decisionVars = set(variables)
        self._numDecVar = len(self._decisionVars)
        self._bool_var_set.update(set(bvariables))
        self._min_var_set.update(set(cvariables0) if min_var_set_id == 0 else set(cvariables1))
        self._free_var_set.update(set(cvariables1) if min_var_set_id == 0 else set(cvariables0))

    def update_name_space(self, ns):
        self._name_space = ns

    def set_objective(self, obj: Union[int, dict]):
        self._obj = obj
        if isinstance(obj, dict):
            self._additive_obj = True
        else:
            self._additive_obj = False

    def get_objective(self):
        return self._obj

    def convert_func_to_xadd(self, term):
        args = term.as_two_terms()
        xadd1 = self.convert_to_xadd(args[0])
        xadd2 = self.convert_to_xadd(args[1])
        op = OP_TYPE[type(term)]
        return self.apply(xadd1, xadd2, op)

    def convert_to_xadd(self, term):
        if isinstance(term, sympy.Symbol) or isinstance(term, numbers.Number):
            return self.get_leaf_node(term)
        else:
            return self.convert_func_to_xadd(term)

    def build_initial_xadd(self, xadd_as_list, to_canonical=False):
        """
        Given decisions and leaf values, recursively build initial XADD and return the id of the root node.
        :param xadd_as_list:        (list)
        :return:
        """
        if len(xadd_as_list) == 1:
            # return self.convert2XADD(xadd_as_list[0])
            return self.get_leaf_node(xadd_as_list[0])

        dec_expr = xadd_as_list[0]
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)

        high = self.build_initial_xadd(xadd_as_list[1])
        low = self.build_initial_xadd(xadd_as_list[2])

        # swap low and high branches if reversed
        if is_reversed:
            tmp = high;
            high = low;
            low = tmp
        if to_canonical:
            return self.get_inode_canon(dec, low, high)
        else:
            return self.get_internal_node(dec, low, high)

    def build_initial_xadd_lp(self, lp_formulation, is_min=True):
        """
        Given LP constraints, recursively build initial XADD and return the id of the root node.
        :param lp_formulation:
        :return:
        """
        if len(lp_formulation) == 1:
            # return self.convert2XADD(lp_formulation[0])
            return self.get_leaf_node(lp_formulation[0])

        dec_expr = lp_formulation[0]
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)
        low = self.oo if is_min else self.NEG_oo
        high = self.build_initial_xadd_lp(lp_formulation[1:])

        # swap low and high branches if reversed
        if is_reversed:
            tmp = high;
            high = low;
            low = tmp
        return self.get_inode_canon(dec, low, high)

    def make_canonical(self, node_id):
        self._reduce_canon_cache.clear()
        return self.make_canonical_int(node_id)

    def make_canonical_int(self, node_id):
        n = self.get_exist_node(node_id)

        # A terminal node should be reduced by default
        if n._is_leaf:
            return self.get_leaf_node_from_node(node=n)

        # Check to see if this node has already been made canonical
        ret = self._reduce_canon_cache.get(node_id, None)
        if ret is not None:
            return ret

        # Recursively ensure canonicity for subdiagrams
        low = self.make_canonical_int(n._low)
        high = self.make_canonical_int(n._high)

        # Enforce canonicity via the 'apply trick' at this level.
        dec = n.dec
        dec_expr = self._id_to_expr[dec]
        dec, is_reversed = self.get_dec_expr_index(dec_expr, True)

        # If reversed, swap low and high branches
        if is_reversed: tmp = high; high = low; low = tmp

        ret = self.get_inode_canon(dec, low, high)

        # Error check
        self.check_local_ordering_and_exit_on_error(ret)

        # Put return value in cache and return
        self._reduce_canon_cache[node_id] = ret
        return ret

    def check_local_ordering_and_exit_on_error(self, node_id):
        """
        :param node_id:         (int)
        """
        node = self.get_exist_node(node_id)
        if not node._is_leaf:
            dec_id = node.dec
            low_n = self.get_exist_node(node._low)
            if not low_n._is_leaf:
                if dec_id >= low_n.dec:
                    # compare local order
                    print("Reordering problem: {} >= {}\n{}: {}\n{}: {}".
                          format(dec_id, low_n.dec, dec_id, self._id_to_expr[dec_id], low_n.dec,
                                 self._id_to_expr[low_n.dec]))
                    raise ValueError
            high_n = self.get_exist_node(node._high)
            if not high_n._is_leaf:
                if dec_id >= high_n.dec:
                    # compare local order
                    print("Reordering problem: {} >= {}\n{}: {}\n{}: {}".
                          format(dec_id, high_n.dec, dec_id, self._id_to_expr[dec_id], high_n.dec,
                                 self._id_to_expr[high_n.dec]))
                    raise ValueError

    def contains_node_id(self, root, target):
        """
        Returns True if hte diagram in root contains a node with the target ID
        :param root:            (int)
        :param target:          (int)
        :return:
        """
        visited = set()
        return self.contains_node_id_int(root, target, visited)

    def contains_node_id_int(self, id, target, visited):
        if id == target: return True
        if id in visited: return False
        visited.add(id)
        node = self.get_exist_node(id)
        if not node._is_leaf:
            if self.contains_node_id_int(node._low, target, visited): return True
            if self.contains_node_id_int(node._high, target, visited): return True
        return False

    def get_inode_canon(self, dec, low, high):
        if dec <= 0:
            print("Warning: Canonizing Negative Decision: {} -> {}".format(dec, self._id_to_expr[abs(dec)]))
        result1 = self.get_inode_canon_apply_trick(dec, low, high)
        result2 = self.get_inode_canon_insert(dec, low, high)

        if result1 != result2 and not self.contains_node_id(result1, self.NAN):
            print("Canonical error (difference not on NAN)")
        return result2
        # return result1

    def get_inode_canon_insert(self, dec, low, high):
        false_half = self.reduce_insert_node(low, dec, self.ZERO, True)
        true_half = self.reduce_insert_node(high, dec, self.ZERO, False)
        return self.apply_int(true_half, false_half, 'sum')

    def reduce_insert_node(self, orig, dec, node_to_insert_on_dec_value, dec_value):
        insertNodeCache = {}
        return self.reduce_insert_node_int(orig, dec, node_to_insert_on_dec_value, dec_value, insertNodeCache)

    def reduce_insert_node_int(self, orig, dec, insertNode, dec_value, insertNodeCache):
        ret = insertNodeCache.get(orig, None)
        if ret is not None:
            return ret

        node = self.get_exist_node(orig)
        if node._is_leaf or ((not node._is_leaf) and (not dec >= node.dec)):
            ret = self.get_internal_node(dec, orig, insertNode) if dec_value else self.get_internal_node(dec,
                                                                                                         insertNode,
                                                                                                         orig)
        else:
            if not node.dec >= dec:
                low = self.reduce_insert_node_int(node._low, dec, insertNode, dec_value, insertNodeCache)
                high = self.reduce_insert_node_int(node._high, dec, insertNode, dec_value, insertNodeCache)
                ret = self.get_internal_node(node.dec, low, high)
            else:
                if dec_value:
                    ret = self.reduce_insert_node_int(node._low, dec, insertNode, dec_value, insertNodeCache)
                else:
                    ret = self.reduce_insert_node_int(node._high, dec, insertNode, dec_value, insertNodeCache)

        insertNodeCache[orig] = ret
        return ret

    def get_inode_canon_apply_trick(self, dec, low, high):
        ind_true = self.get_internal_node(dec, self.ZERO, self.ONE)
        ind_false = self.get_internal_node(dec, self.ONE, self.ZERO)
        true_half = self.apply_int(ind_true, high, 'prod')
        false_half = self.apply_int(ind_false, low, 'prod')
        result = self.apply_int(true_half, false_half, 'sum')
        return result

    def apply(self, id1, id2, op, annotation=None):
        """
        Recursively apply op(node1, node2). op can be min, max, sum, prod.
        :param id1:
        :param id2:
        :param op:
        :param annotation:          (tuple)
        :return:
        """
        ret = self.apply_int(id1, id2, op, annotation)
        if op == 'min' or op == 'max':
            ret = self.make_canonical(ret)
        return ret

    def get_apply_cache(self):
        assert self._opt_var is not None
        hm = self._apply_caches.get(self._opt_var, None)
        if hm is not None:
            return hm
        else:
            hm = {}
            self._apply_caches[self._opt_var] = hm
            return hm

    def apply_int(self, id1, id2, op, annotation=None):
        """
        Recursively apply op(node1, node2).
        :param id1:         (int) index of node 1
        :param id2:         (int) index of node 2
        :param op:          (str) 'max', 'min', 'sum', 'prod'
        :return:
        """
        # Check apply cache and return if found
        if annotation is None:
            _tempApplyKey = (id1, id2, op)
            ret = self._apply_cache.get(_tempApplyKey, None)
        elif self._opt_var is None:
            _tempApplyKey = (id1, id2, op, annotation[0], annotation[1])
            ret = self._apply_cache.get(_tempApplyKey, None)
        else:
            _tempApplyKey2 = (id1, id2, op, annotation[0], annotation[1])
            ret = self.get_apply_cache().get(_tempApplyKey2, None)

        if ret is not None:
            return ret

        # If not found, compute..
        n1 = self.get_exist_node(id1)
        n2 = self.get_exist_node(id2)
        ret = self.compute_leaf_node(id1, n1, id2, n2, op, annotation)

        if ret is None:
            # Determine the new decision expression
            if not n1._is_leaf:
                if not n2._is_leaf:
                    if n2.dec >= n1.dec:
                        dec = n1.dec
                    else:
                        dec = n2.dec
                else:
                    dec = n1.dec
            else:
                dec = n2.dec

            # Determine next recursion for n1
            if (not n1._is_leaf) and (n1.dec == dec):
                low1, high1 = n1._low, n1._high
            else:
                low1, high1 = id1, id1

            # Determine next recursion for n2
            if (not n2._is_leaf) and (n2.dec == dec):
                low2, high2 = n2._low, n2._high
            else:
                low2, high2 = id2, id2

            low = self.apply_int(low1, low2, op, annotation)
            high = self.apply_int(high1, high2, op, annotation)

            ret = self.get_internal_node(dec, low, high)

        # Add result to apply cache
        if annotation is None:
            self._apply_cache[(id1, id2, op)] = ret
        elif self._opt_var is None:
            self._apply_cache[(id1, id2, op, annotation[0], annotation[1])] = ret
        else:
            self.get_apply_cache()[(id1, id2, op, annotation[0], annotation[1])] = ret
        return ret

    def compute_leaf_node(self, id1, n1, id2, n2, op, annotation):
        """
        %Update%
        To support 0-1 knapsack problem with DP, summation operation now keep the track of annotation (when provided).
        In this case, annotation[0] is assumed to be None, and we concatenate annotation[1] to n1._annotation if
        both nodes are leaf nodes.
        Unlike in the case of argmin(max) over an LP, annotations in knapsack correspond to indices of items selected to
        the knapsack. Hence, an annotation is a tuple of integers.
        In the case of argmin(max), annotations represent a node id corresponding to either a lower or an upper bound
        that has led to the min (or max) value as a result of the current method.
        :param id1:         (int)
        :param n1:          (Node)
        :param id2:         (int)
        :param n2:          (Node)
        :param op:          (str) 'max', 'min', 'sum', 'prod'
        :param annotation:  (tuple)
        :return:
        """
        assert op in ('max', 'min', 'sum', 'prod')
        # NaN cannot become valid by operations
        # But, this would not occur unless we intended..
        # Hence, just deal with NaNs when two leaf nodes are compared
        if ((id1 == self.NAN) or (id2 == self.NAN)) and (n1._is_leaf and n2._is_leaf):
            return self.NAN
        elif (id1 == self.NAN) or (id2 == self.NAN):
            return None

        # 0 * x = 0
        if op == 'prod' and (
                (id1 == self.ZERO or id2 == self.ZERO) and not (id1 == self.ZERO_ig or id2 == self.ZERO_ig)
        ):
            return self.ZERO
        elif op == 'prod' and (
                (id1 == self.ZERO_ig or id2 == self.ZERO_ig)
        ):
            return self.ZERO_ig

        # Identities (1 * x = x, 0 + x = x)
        if annotation is None:
            if (op == 'sum' and id1 == self.ZERO):
                # if annotation is not None and self._problem_type == 'knapsack':
                #     return self.get_leaf_node(n2.expr, annotation=n1._annotation + (annotation[1],))
                return id2
            if (op == 'prod' and id1 == self.ONE):
                return id2
            if ((op == 'sum' or op == 'minus') and id2 == self.ZERO) or (op == 'prod' and id2 == self.ONE):
                return id1

        # Infinity identities
        # Due to annotations, XADDTNodes with oo or -oo as expression may have different node id.
        # Therefore, check for node._is_leaf first.. and maintain annotations (if exist) always.
        # Furthermore, when n1.expr == n2.expr == oo (or -oo), we need to annotate the resulting oo (or -oo) node
        # as NaN, since those indicate infeasible paths.
        if n1._is_leaf and n1.expr == oo:
            if not n2._is_leaf:
                if op == 'max' or op == 'sum' or op == 'minus':
                    return self.get_leaf_node_from_node(n1)  # To retain annotation (if exists)
                elif op == 'min':
                    return id2
            else:
                if n2.expr != oo and (op == 'max' or op == 'sum' or op == 'minus'):
                    return self.get_leaf_node_from_node(n1)  # To retain annotation (if exists)
                elif n2.expr != oo and (op == 'min'):
                    return id2 if annotation is None else self.get_leaf_node(n2.expr, annotation[1])
                elif n2.expr == oo and (op in ('max', 'min', 'sum', 'prod')):
                    ret_node = self.get_leaf_node(oo, annotation=self.NAN)
                    # self.add_special_node(ret_node)
                    return ret_node

        elif n1._is_leaf and n1.expr == -oo:
            if not n2._is_leaf:
                if op == 'sum' or op == 'minus' or op == 'min':
                    return self.get_leaf_node_from_node(n1)
                elif op == 'max':
                    return id2
            else:
                if n2.expr != -oo and (op in ('sum', 'min', 'minus')):
                    return self.get_leaf_node_from_node(n1)
                elif n2.expr != -oo and op == 'max':
                    return id2 if annotation is None else self.get_leaf_node(n2.expr, annotation[1])
                elif n2.expr == -oo and (op in ('max', 'min', 'sum')):
                    ret_node = self.get_leaf_node(-oo, annotation=self.NAN)
                    # self.add_special_node(ret_node)
                    return ret_node
                elif n2.expr == -oo and op == 'prod':
                    ret_node = self.get_leaf_node(oo, annotation=self.NAN)
                    # self.add_special_node(ret_node)
                    return ret_node

        if n2._is_leaf and n2.expr == oo:
            # n1 cannot be oo or -oo at this point..
            if op == 'sum' or op == 'max':
                return self.get_leaf_node_from_node(n2)
            elif op == 'min':
                if annotation is None:  # If annotation is given, need to annotate them at leaf nodes
                    return id1
                if n1._is_leaf:  # For non-leaf nodes,
                    return self.get_leaf_node(n1.expr, annotation[0])
            elif op == 'minus':
                return self.NEG_oo
        elif n2._is_leaf and n2.expr == -oo:
            if op == 'sum' or op == 'min':
                return self.get_leaf_node_from_node(n2)
            elif op == 'max':
                if not n1._is_leaf:
                    return id1
                else:
                    return id1 if annotation is None else self.get_leaf_node(n1.expr, annotation[0])
            elif op == 'minus':
                return self.oo

        if n1._is_leaf and n2._is_leaf:
            if id1 == self.NAN or id2 == self.NAN:
                return self.NAN
            # Operations: +, -, *
            # No need to take care of annotations for these operations unless in Knapsack problems
            if op != 'max' and op != 'min':
                n1_expr, n2_expr = n1.expr, n2.expr
                if op == 'sum':
                    result = n1_expr + n2_expr
                elif op == 'minus':
                    result = n1_expr - n2_expr
                else:
                    result = sympy.expand(n1_expr * n2_expr)

                # (deprecated) annotation is given in knapsack DP algorithm: need to simply concatenate the current item id
                if annotation is not None and self._problem_type == 'knapsack':
                    return self.get_leaf_node(result, n1._annotation + (annotation[1],))
                # annotation is given in 'sum' operation... used to handle factorization of bilinear terms during SVE
                elif annotation is not None and op == 'sum':
                    if id1 == self.ZERO_ig and id2 != self.ZERO_ig:
                        return self.get_leaf_node(result, annotation[1])
                    elif id1 != self.ZERO_ig and id2 == self.ZERO_ig:
                        return self.get_leaf_node(result, annotation[0])
                    elif id1 == self.ZERO_ig and id2 == self.ZERO_ig:
                        logger.warning("Both leaf nodes are ZERO_ig nodes. Annotation is provided but will be ignored")
                        return self.get_leaf_node(result)
                    elif id1 == self.ZERO:
                        return self.get_leaf_node(result, annotation[1])
                    elif id2 == self.ZERO:
                        return self.get_leaf_node(result, annotation[0])
                    else:
                        logger.warning("annotations are given for 'sum' op, but cannot determine which to use")
                        return self.get_leaf_node(result)
                else:
                    return self.get_leaf_node(result)

            # The canonical form of decision expression: (lhs - rhs <= 0)
            lhs = n1.expr - n2.expr
            rhs = 0
            expr = lhs <= rhs

            # handle tautological cases
            if expr == sympy.S.true:
                if op == 'min':
                    return self.get_leaf_node(n1.expr, annotation[0]) if annotation is not None else id1
                else:
                    return self.get_leaf_node(n2.expr, annotation[1]) if annotation is not None else id2
            elif expr == sympy.S.false:
                if op == 'min':
                    return self.get_leaf_node(n2.expr, annotation[1]) if annotation is not None else id2
                else:
                    return self.get_leaf_node(n1.expr, annotation[0]) if annotation is not None else id1

            if annotation is not None:
                id1 = self.get_leaf_node(n1.expr, annotation[0])
                id2 = self.get_leaf_node(n2.expr, annotation[1])
            else:
                id1 = self.get_leaf_node(n1.expr, n1._annotation)
                id2 = self.get_leaf_node(n2.expr, n2._annotation)

            canon_expr, is_reversed = self.canonical_dec_expr(expr)
            if isinstance(canon_expr, tuple):
                """
                A factored bilinear term returned: need to build an XADD accordingly
                For example,
                    canon_expr = ((z1 + z2 - z3), (w1 + w2 + w3)); is_reversed = False
                    Then, ([ w1 + w2 + w3 <= 0 ]
                              ([ z1 + z2 - z3 <= 0 ])
                                   ([ n2.expr ])
                                   ([ n1.expr ]))
                              ([ z1 + z2 - z3 <= 0 ])
                                   ([ n1.expr ])
                                   ([ n2.expr ])
                              )
                          )
                    The leaf values will be swapped if is_reversed.  
                """
                expr1, expr2 = canon_expr
                expr_index1, is_reversed1 = self.get_dec_expr_index(expr1 <= 0, create=True)
                expr_index2, is_reversed2 = self.get_dec_expr_index(expr2 <= 0, create=True)

                if bool(is_reversed) ^ bool(is_reversed1):
                    tmp = id2;
                    id2 = id1;
                    id1 = tmp

                high_node = self.get_internal_node(expr_index1, low=id1, high=id2)
                low_node = self.get_internal_node(expr_index1, low=id2, high=id1)

                if is_reversed2:
                    tmp = high_node;
                    high_node = low_node;
                    low_node = tmp
                low = high_node if op == 'max' else low_node
                high = low_node if op == 'max' else high_node
                ret = self.get_internal_node(expr_index2, low=low, high=high)
                return ret
            else:
                expr_index, _ = self.get_dec_expr_index(canon_expr, create=True, canon=True)

                # Swap low and high branches if reversed
                if is_reversed:
                    tmp = id2;
                    id2 = id1;
                    id1 = tmp
                low = id1 if op == 'max' else id2
                high = id2 if op == 'max' else id1

                return self.get_internal_node(expr_index, low=low, high=high)
        return None

    def substitute(self, node_id, subst_dict):
        """
        Symbolic substitution method.
        :param node_id:             (int)
        :param subst_dict:          (dict)
        :return:
        """
        subst_cache = {}
        return self.reduce_sub(node_id, subst_dict, subst_cache)

    def reduce_sub(self, node_id, subst_dict, subst_cache):
        """
        :param node_id:
        :param subst_dict:
        :param subst_cache:
        :return:
        """
        node = self.get_exist_node(node_id)

        # A terminal node should be reduced by default
        if node._is_leaf:
            expr = node.expr
            for sub_out, sub_in in subst_dict.items():
                # expr = expr.subs(sub_out, sub_in)
                expr = expr.xreplace({sub_out: sympy.S(sub_in)})
            expr = sympy.expand(expr)
            annotation = node._annotation
            return self.get_leaf_node(expr, annotation)

        # If it's an internal node, check the reduce cache
        ret = subst_cache.get(node_id, None)
        if ret is not None:
            return ret

        # Handle an internal node
        low = self.reduce_sub(node._low, subst_dict, subst_cache)
        high = self.reduce_sub(node._high, subst_dict, subst_cache)

        dec = node.dec
        dec_expr = self._id_to_expr[dec]
        if isinstance(dec_expr, sympy.Symbol):
            sub_in = subst_dict.get(dec_expr, None)
            if sub_in is not None:
                dec, is_reversed = self.get_dec_expr_index(sub_in, create=False)
        else:
            lhs = dec_expr.lhs
            for sub_out, sub_in in subst_dict.items():
                # Check if the expression holds in equality and the true branch is NaN
                # Assuming canonical expression.. rhs is always 0. Hence, lhs == 0 iff dec_expr == S.true.
                # In this case, set dec_expr = False, so that false branch can be chosen instead.
                # lhs = lhs.subs(sub_out, sub_in)
                lhs = lhs.xreplace({sub_out: sympy.S(sub_in)})

            if lhs == 0 and high == self.NAN:
                dec_expr = S.false
            elif lhs == 0 and low == self.NAN:
                dec_expr = S.true
            else:
                dec_expr = lhs <= 0

            # # Handle tautologies
            if dec_expr == S.true:
                subst_cache[node_id] = high
                return high
            elif dec_expr == S.false:
                subst_cache[node_id] = low
                return low
            dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)

        # Swap low and high branches if reversed
        if is_reversed:
            tmp = high;
            high = low;
            low = tmp

        # # Substitution could have affected variable ordering.
        # else:
        if not self._direct_substitution:
            ret = self.get_inode_canon(dec, low, high)
            self.check_local_ordering_and_exit_on_error(ret)
        else:
            ret = self.get_internal_node(dec, low, high)

        # Put return value in cache and return
        subst_cache[node_id] = ret
        return ret

    def substitute_bool_vars(self, node_id, subst_dict):
        """
        Symbolic substitution method for bool variables.
        :param node_id:             (int)
        :param subst_dict:          (dict)
        :return:
        """
        varSet = self.collect_vars(node_id)
        for var in subst_dict:
            if var in varSet:
                dec, _ = self.get_dec_expr_index(var, create=False)
                if subst_dict[var]:
                    node_id = self.op_out(node_id, dec, "restrict_high")
                else:
                    node_id = self.op_out(node_id, dec, "restrict_low")
        return node_id

    def op_out(self, node_id, dec_id, op):
        ret = self.reduce_op(node_id, dec_id, op)

        # Operations like sum and product may get decisions out of order
        if op == 'sum' or op == 'prod':
            return self.make_canonical(ret)
        else:
            return ret

    def reduce_op(self, node_id, dec_id, op):
        node = self.get_exist_node(node_id)

        # A terminal node should be reduced (and cannot be restricted)
        # by default if hashing and equality testing are working in getLeafNode
        if node._is_leaf:
            return node_id  # Assuming that to have a node id means canonical

        # If its an internal node, check the reduce cache
        temp_reduce_key = (node_id, dec_id, op)
        ret = self._reduce_cache.get(temp_reduce_key, None)
        if ret is not None:
            return ret

        if (op != "restrict_high") or (dec_id != node.dec):
            low = self.reduce_op(node._low, dec_id, op)
        if (op != "restrict_low") or (dec_id != node.dec):
            high = self.reduce_op(node._high, dec_id, op)
        # if (op != -1 & & var_id != -1 & & var_id == inode._var) {
        if (dec_id != -1) and (dec_id == node.dec):
            # ReduceOp
            if op == "restrict_low":
                ret = low
            elif op == "restrict_high":
                ret = high
            elif op == "sum" or op == "prod":
                ret = self.apply(low, high, op)
            else:
                raise NotImplementedError
        else:
            ret = self.get_internal_node(node.dec, low, high)

        # Put return value in cache and return
        self._reduce_cache[temp_reduce_key] = ret
        return ret

    def collect_vars(self, node_id):
        node = self.get_exist_node(node_id)
        vars = set()
        return node.collect_vars(vars)

    def reduced_arg_min_or_max(self, node_id, var):
        arg_id = self.get_arg(node_id)
        arg_id = self.reduce_lp(arg_id)
        self.update_anno(var, arg_id)
        return arg_id

    def get_arg(self, node_id, is_min=True):
        self._is_min = is_min
        ret = self.get_arg_int(node_id)
        return self.make_canonical(ret)

    def get_arg_int(self, node_id):
        """
        node_id is the id of a min/max XADD node. var is the decision variable that was max(min)imized out to result in
        node_id. Recursively build the annotation XADD for var.
        :param node_id:             (int)
        :return:
        """
        node = self.get_exist_node(node_id)

        if node._is_leaf:
            annotation = node._annotation
            if annotation is None and (node.expr == oo or node.expr == -oo):  # Annotate infeasibility
                annotation = self.NAN
            return annotation
        else:
            low = self.get_arg_int(node._low)
            high = self.get_arg_int(node._high)

            dec = node.dec
            dec_expr = self._id_to_expr[dec]
            dec, is_reversed = self.get_dec_expr_index(dec_expr, True)
            # Swap low and high branches if reversed
            if is_reversed:
                tmp = high;
                high = low;
                low = tmp
            return self.get_inode_canon(dec, low, high)

    def argmin_or_max(self, varOrder, resubstitution=True):
        """
        Once all decision variables are min(max)imized, we get argmin(max) over each variable from get_arg method.
        Then, we need to provide the variable order with which optimization is done, and we sequentially
        substitute annotations of outer variables into inner variables to get argmin(max) of all variables.
        The resulting annotation XADDs should be referenced by self._var_to_anno.
        :param varOrder:            (list) List of sympy.Symbol variables
        :return:
        """
        varOrder = self._eliminated_var + varOrder
        num_var = len(varOrder)
        self._prune_equality = False
        self.RLPContext.flush_implications()

        print("\nComputing argmin of decision variables")
        for i in tqdm(range(num_var - 2, -1, -1), desc="Variables"):
            curr_var = varOrder[i]
            curr_anno = self.get_annotation(curr_var)

            # Substitute all previous annotations sequentially to the current annotation
            # That is, for ith variable, substitute in i+1, ..., num_var-1 variables
            for j in range(num_var - 1, i, -1):
                outer_var = varOrder[j]
                outer_anno = self.get_annotation(outer_var)
                substitution = DeltaFunctionSubstitution(outer_var, curr_anno, self)
                curr_anno = self.reduce_process_xadd_leaf(outer_anno, substitution, [], [])
            self.update_anno(curr_var, self.reduce_lp(curr_anno))

        # Turn pruning equality decisions back on
        self._prune_equality = True
        self.RLPContext.flush_implications()

        for i in tqdm(range(num_var), desc='Perform reduce LP'):
            v = varOrder[i]
            v_anno = self.get_annotation(v)
            v_anno = self.reduce_lp(v_anno)
            # if resubstitution:
            #     v_anno = self.resubst_params(v_anno)
            self.update_anno(v, v_anno)
        return self.get_all_anno()

    def get_all_anno(self):
        return self._var_to_anno

    def get_annotation(self, var):
        return self._var_to_anno[var]

    def update_anno(self, var, anno):
        self._var_to_anno[var] = anno

    def get_node(self, node_id):
        """
        Retrieve a XADD node from cache.
        :param node_id:             (int)
        :return:
        """
        return self._id_to_node[node_id]

    def min_or_max_multi_var(self, node_id, var_lst, is_min=True, annotate=True):
        """
        Given an XADD root node 'node_id', minimize (or maximize) variables in 'var_lst'.
        Supports only continuous variables.
        """
        decisions, decision_values = [], []
        min_or_max = XADDLeafMultivariateMinOrMax(
            var_lst,
            is_max=False if is_min else True,
            bound_dict=self._var_to_bound,
            context=self,
            annotate=annotate,
        )
        _ = self.reduce_process_xadd_leaf(node_id, min_or_max, decisions, decision_values)
        res = min_or_max._running_result
        return res

    def min_or_max_var(self, node_id, var, is_min=True):
        """
        Given an XADD root node 'node_id', minimize (or maximize) 'var' out.
        :param node_id:      (int)
        :param var:                 (sympy.Symbol)
        :return:                    (int)
        """
        # Check if binary variable
        if var in self._bool_var_set:
            op = "min" if is_min else "max"
            self._opt_var = var
            subst_high = {var: True}
            subst_low = {var: False}
            restrict_high = self.substitute_bool_vars(node_id, subst_high)
            restrict_low = self.substitute_bool_vars(node_id, subst_low)
            running_result = self.apply(restrict_high, restrict_low, op=op, annotation=(self.ONE, self.ZERO))
            running_result = self.reduce_lp(running_result)
        # Continuous variables
        else:
            decisions, decision_values = [], []
            min_or_max = XADDLeafMinOrMax(var, is_max=False if is_min else True, bound_dict=self._var_to_bound,
                                          context=self)
            _ = self.reduce_process_xadd_leaf(node_id, min_or_max, decisions, decision_values)
            running_result = min_or_max._running_result
        return running_result

    def depth_first_reduce_xadd_leaf(self, node_id, leaf_op, decisions, decision_values):
        """
        This function applies `reduce` subroutine recursively in the depth-first manner.
        For example, when we perform multivariate symbolic minimization of a case function,
         `reduce_process_xadd_leaf` already traverses the diagram in the depth-first manner.
         At the leaf node, however, only a single variable is minimized out, then we minimize out the variable
         from another partition, whose result is compared to the earlier one.
        Instead, what we want to achieve here is this: once we get the resulting case function of a single variable
         minimization, then we perform minimization of the function w.r.t. the next variable. This will produce yet
         another case function, and we repeat until there's no remaining variables to optimize.
        """
        pass

    def reduce_process_xadd_leaf(self, node_id, leaf_op, decisions, decision_values):
        """
        :param node_id:
        :param leaf_op:
        :param decisions:
        :param decision_values:
        :return:
        """
        node = self.get_exist_node(node_id)
        if node._is_leaf:
            return leaf_op.process_xadd_leaf(decisions, decision_values, node.expr)

        # Internal node
        dec_expr = self._id_to_expr.get(node.dec)

        # Recurse the False branch
        decisions.append(dec_expr)
        decision_values.append(False)
        low = self.reduce_process_xadd_leaf(node._low, leaf_op, decisions, decision_values)

        # Recurse the True branch
        decision_values[-1] = True
        high = self.reduce_process_xadd_leaf(node._high, leaf_op, decisions, decision_values)

        decisions.pop()
        decision_values.pop()

        ret = self.get_internal_node(node.dec, low, high)
        if isinstance(leaf_op, DeltaFunctionSubstitution):
            ret = self.make_canonical(ret)

        # Put return value in cache and return
        self._reduce_leafop_cache[node_id] = ret
        return ret

    def substitute_xadd_for_var_in_expr(self, leaf_val, var, xadd):
        """
        Substitute XADD into 'var' that occurs in 'val' (a Sympy expression). This is only called for
        leaf expressions.
        :param leaf_val:    (sympy.Basic) sympy expression
        :param var:         (sympy.Symbol) variable to substitute
        :param xadd:        (int) integer that indicates the XADD to substitute into 'var'
        :return:
        """
        # Get the root node
        node = self._id_to_node[xadd]

        # Handle leaf node cases: simply substitute leaf expression into 'var' in leaf_val
        if node._is_leaf:
            xadd_leaf_expr = node.expr
            # expr = leaf_val.subs(var, xadd_leaf_expr)
            expr = leaf_val.xreplace({var: xadd_leaf_expr})
            expr = sympy.expand(expr)

            # Special treatment for oo, -oo
            try:
                two_terms = expr.as_two_terms()
                if isinstance(two_terms[0], sympy.core.Number):
                    if two_terms[0] == sympy.oo:
                        expr = sympy.oo
                    elif two_terms[0] == -sympy.oo:
                        expr = -sympy.oo
            except AttributeError as e:
                pass
            except Exception as e:
                logger.error(e)
                exit(1)
            node_id = self.get_leaf_node(expr, annotation=None)
            return node_id

        # Internal nodes: get low and high branches and do recursion
        low, high = node._low, node._high
        low = self.substitute_xadd_for_var_in_expr(leaf_val, var, low)
        high = self.substitute_xadd_for_var_in_expr(leaf_val, var, high)

        # Get the node id for a (sub)XADD and return it
        node_id = self.get_internal_node(node.dec, low=low, high=high)

        return node_id

    def get_repr(self, node_id):
        # For printing out the representation
        node = self._id_to_node[node_id]
        return node

    def get_leaf_node_from_node(self, node):
        """
        :param node:            (Node) If Node object is passed.. also take annotation into consideration
        :return:
        """
        expr, annotation = node.expr, node._annotation
        return self.get_leaf_node(expr, annotation)

    def get_leaf_node(self, expr, annotation=None):
        """
        :param expr:            (sympy.Basic) symbolic expression, not id
        :param annotation:      (int) id of annotation XADD
        :return:
        """
        self._tempTNode.set(expr, annotation)
        node_id = self._node_to_id.get(self._tempTNode, None)
        if node_id is None:
            # node not in cache, so create
            node_id = self._nodeCounter
            node = XADDTNode(expr, annotation, context=self)
            self._id_to_node[node_id] = node
            self._node_to_id[node] = node_id
            self._nodeCounter += 1

            # add in all new variables
            vars_in_expr = set()
            vars_in_expr.update(expr.free_symbols)
            diff_vars = vars_in_expr.difference(self._var_set)
            for v in diff_vars:
                # Boolean variables would have been added immediately so are already in self._bool_var_set
                if not v in self._bool_var_set:
                    num_existing_cvar = len(self._var_set)
                    self._var_set.add(v)
                    self._str_var_to_var[str(v)] = v
                    self._cvar_to_id[v] = num_existing_cvar
                    self._id_to_cvar[num_existing_cvar] = v
        return node_id

    def get_dec_node(self, dec_expr, low_val, high_val):
        """
        Get decision node with relational expression having dec, whose low and high values are also given.
        :param dec_expr:            (sympy.relational.Rel)
        :param low_val:             (float)
        :param high_val:            (float)
        :return:
        """
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)
        low = self.get_leaf_node(low_val)
        high = self.get_leaf_node(high_val)
        # Swap low and high branches if reversed
        if is_reversed:
            tmp = high;
            high = low;
            low = tmp
        return self.get_internal_node(dec, low, high)

    # def clean_up_expr(self, expr, factor=False):
    #     is_reversed = False

    # Divide lhs by the coefficient of the first term (cannot be a negative number)
    # coeff_first_term = expr.as_ordered_terms()[0]
    # if isinstance(coeff_first_term, sympy.core.Number):
    #     coeff_first_term = expr.as_ordered_terms()[1]
    #
    # if isinstance(coeff_first_term, sympy.core.Mul):
    #     arg1 = coeff_first_term.args[0]
    #     if isinstance(arg1, sympy.core.Number) and arg1 > 0:
    #         expr = expr / arg1
    #     elif isinstance(arg1, sympy.core.Number) and arg1 < 0:
    #         expr = expr / arg1
    #         is_reversed = True if not is_reversed else False  # divided by a negative number changes the direction of inequality

    # if factor:
    #     # lhs = lhs - rhs
    #     ret_expr = self._factor_cache.get(expr, None)
    #     if ret_expr is not None:
    #         expr = ret_expr
    #     else:
    #         ret_expr = sympy.factor(expr)
    #         self._factor_cache[expr] = ret_expr
    #         expr = ret_expr
    #
    #     # When lhs is factored, check if a constant is multiplied, in which case we divide both sides by the constant
    #     # and make sure the inequality holds
    #     if isinstance(expr, sympy.core.Mul):
    #         arg1 = expr.args[0]
    #         # By default, sympy will always put constants at front. So, checking arg1 suffices.
    #         if isinstance(arg1, sympy.core.Number) and arg1 > 0:
    #             expr = expr / arg1
    #         elif isinstance(arg1, sympy.core.Number) and arg1 < 0:
    #             expr = expr / arg1
    #             is_reversed = True if not is_reversed else False    # divided by a negative number changes the direction of inequality
    #
    #     return expr, is_reversed

    def canonical_dec_expr(self, expr):
        """
        Return canonical form of an expression.
        It should always take either one of the two forms: expr.lhs <= 0 or expr.lhs >= 0.
        Args:
            expr:
        Returns:
        """
        # sympy expr is already alphabetically ordered
        is_reversed = False

        # Handle tautology: simply return without doing anything
        if expr == sympy.S.true:
            return expr, is_reversed
        elif expr == sympy.S.false:
            return expr, is_reversed

        # Always make 'lhs - rhs <= 0' as canonical expression
        lhs, rhs, rel = expr.lhs, expr.rhs, REL_TYPE[type(expr)]
        lhs = lhs - rhs

        if rel == '>=' or rel == '>':
            is_reversed = True

        # Try factorization when the degree of expression is high
        poly = lhs.as_poly()
        deg = poly.total_degree()

        if deg == 1:
            # Divide lhs by the coefficient of the first term (cannot be a negative number)
            coeff_first_term = lhs.as_ordered_terms()[0]
            if isinstance(coeff_first_term, sympy.core.Number):
                coeff_first_term = lhs.as_ordered_terms()[1]

            if isinstance(coeff_first_term, sympy.core.Mul):
                arg1 = coeff_first_term.args[0]
                if isinstance(arg1, sympy.core.Number) and arg1 > 0:
                    lhs = lhs / arg1
                elif isinstance(arg1, sympy.core.Number) and arg1 < 0:
                    lhs = lhs / arg1
                    is_reversed = True if not is_reversed else False  # divided by a negative number changes the direction of inequality

            expr = relational.Relational(lhs, 0, "<=")
            return expr, is_reversed

        elif deg == 2:
            lhs_factored, is_reversed_ = self.factor_expr(lhs)
            if is_reversed_:
                is_reversed = True if not is_reversed else False
            # If no factorization occurred, just create a relational expression
            if not isinstance(lhs_factored, tuple):
                expr = relational.Relational(lhs_factored, 0, "<=")
                return expr, is_reversed
            return lhs_factored, is_reversed
        else:
            raise ValueError("An expression with degree higher than 2 is not supported (yet?)")

    def factor_expr(self, expr):
        ret = self._factor_cache.get(expr, None)
        is_reversed = False

        if ret is not None:
            expr = ret
        else:
            ret = sympy.factor(expr)
            self._factor_cache[expr] = ret
            expr = ret

        # If not factorized... just remove the constant multiplication
        if isinstance(expr, sympy.core.Add):
            first_term = expr.as_ordered_terms()[0]
            if isinstance(first_term, sympy.core.numbers.Number):
                first_term = expr.as_ordered_terms()[1]
            if isinstance(first_term, sympy.core.Mul):
                arg1 = first_term.args[0]
                if isinstance(arg1, sympy.core.numbers.Number) and arg1 > 0:
                    expr = expr / arg1
                elif isinstance(arg1, sympy.core.numbers.Number) and arg1 < 0:
                    expr = expr / arg1
                    is_reversed = True if not is_reversed else False

        # Remove the constant term (if exists) multiplied
        if isinstance(expr, sympy.core.Mul):
            arg1 = expr.args[0]
            if isinstance(arg1, sympy.core.numbers.Number):
                expr = expr / arg1
                if arg1 < 0:
                    is_reversed = True if not is_reversed else False

        if isinstance(expr, sympy.core.Mul):
            assert len(expr.args) <= 2, "Once a constant multiplication is removed, there should be maximum 2 args"
            return expr.args, is_reversed
        else:
            return expr, is_reversed

    def get_dec_expr_index(self, expr, create, canon=False):
        """
        Given a symbolic expression 'expr', return the index of the expression in XADD._id_to_expr.
        :param expr:            (sympy.Basic)
        :param create:          (bool)
        :return:                (int)
        """
        is_reversed = False
        if not canon and isinstance(expr, relational.Rel):
            expr, is_reversed = self.canonical_dec_expr(expr)

        if expr == sympy.S.true or expr == sympy.S.false:
            return expr, is_reversed

        index = self._expr_to_id.get(expr, None)

        if index is None:
            index = 0

        # If found, and not create
        if index != 0 or not create:
            return index, is_reversed
        # If nothing's found, create one and store
        else:
            vars_in_expr = expr.free_symbols.copy()
            # if vars_in_expr.intersection(self._free_var_set):
            #     index = len(self._expr_to_id) + LARGE_INTEGER  # Make expr_id start from 1 (0: NullDec)
            # else:
            #     index = len(self._expr_to_id)
            poly = expr.lhs.as_poly()
            deg = poly.total_degree()
            if deg > 1:
                index = len(self._expr_to_id) + LARGE_INTEGER ** 2  # Make expr_id start from 1 (0: NullDec)
            elif vars_in_expr.intersection(self._free_var_set):
                index = len(self._expr_to_id) + LARGE_INTEGER  # TODO: need to prevent overwriting...
            else:
                index = len(self._expr_to_id)

            self._expr_to_id[expr] = index
            self._id_to_expr[index] = expr

            if isinstance(expr, sympy.Symbol):
                self._bool_var_set.update(vars_in_expr)
            else:
                diff_vars = vars_in_expr.difference(self._var_set)
                for v in diff_vars:
                    num_existing_cvar = len(self._var_set)
                    self._var_set.add(v)
                    self._str_var_to_var[str(v)] = v
                    self._cvar_to_id[v] = num_existing_cvar
                    self._id_to_cvar[num_existing_cvar] = v
        return index, is_reversed

    def get_exist_node(self, node_id):
        node = self._id_to_node.get(node_id, None)
        if node is None:
            print("Unexpected Missing node: " + node_id)
        return node

    def get_internal_node(self, dec_id: int, low: int, high: int):
        """
        :param dec_id:      (int) id of decision expression
        :param low:         (int) id of low branch node
        :param high:        (int) id of high branch node
        :return:            (int) return id of node
        """
        if dec_id < 0:
            tmp = high;
            high = low;
            low = tmp
            dec_id = -dec_id

        # Check if low == high
        if low == high:
            return low

        # Handle tautological cases
        dec_expr = self._id_to_expr.get(dec_id, None)
        if dec_expr == sympy.S.true:
            return high
        elif dec_expr == sympy.S.false:
            return low

        # Retrieve XADDINode (create if it does not exist)
        self._tempINode.set(dec_id, low, high)
        node_id = self._node_to_id.get(self._tempINode, None)
        if node_id is None:
            node_id = self._nodeCounter
            node = XADDINode(dec_id, low, high, context=self)
            self._node_to_id[node] = node_id
            self._id_to_node[node_id] = node
            self._nodeCounter += 1
        return node_id

    """
    Verifying feasibility and redundancy of all paths in the XADD
    """

    def reduce_lp(self, node_id: int):
        """
        Consistency and redundancy checking
        :param node_id:     (int) Node id
        :return:
        """
        return self.RLPContext.reduce_lp(node_id)

    """
    Cache maintenance
    """

    def clear_special_nodes(self):
        self._special_nodes.clear()

    def add_special_node(self, n: int):
        try:
            if n is None:
                raise ValueError("add_sepcial_node: None")
        except ValueError as error:
            print(error)
            exit(1)
        self._special_nodes.add(n)

    def remove_special_node(self, n: int):
        self._special_nodes.discard(n)

    def add_annotations_to_special_nodes(self):
        for v, v_anno in self.get_all_anno().items():
            self._special_nodes.add(v_anno)

    def flush_caches(self):
        logger.info(f"[FLUSHING CACHES...  {len(self._node_to_id) + len(self._id_to_node)}, nodes -> ")

        # Can always clear these
        self._reduce_cache.clear()
        self._reduce_canon_cache.clear()
        self._reduce_leafop_cache.clear()
        self._apply_cache.clear()
        for applyCache in self._apply_caches.values():
            applyCache.clear()
        self._inode_to_vars.clear()
        # self._factor_cache.clear()
        self._temp_ub_lb_cache.clear()
        # self._reduce_annotate_cache.clear()
        self.RLPContext.flush_implications()

        # Set up temporary alternates to these HashMaps
        self._node_to_id_new.clear()
        self._id_to_node_new.clear()
        # self._id_to_expr_new.clear()
        # self._expr_to_id_new.clear()

        # Copy over 'special' nodes then set new dict
        for node_id in self._special_nodes:
            self.copy_in_new_cache_node(node_id)

        self._node_to_id = self._node_to_id_new.copy()
        self._id_to_node = self._id_to_node_new.copy()
        # self._expr_to_id = self._expr_to_id_new.copy()
        # self._id_to_expr = self._id_to_expr_new.copy()

        self.create_standard_nodes()

        logger.info(f"{len(self._node_to_id) + len(self._id_to_node)} nodes")

    def copy_in_new_cache_node(self, node_id: int):
        if node_id in self._id_to_node_new:
            return
        node = self.get_exist_node(node_id)
        if not node._is_leaf:
            # dec_id = node.dec
            # dec_expr = self._id_to_expr[dec_id]

            # Copy node and node id
            self._id_to_node_new[node_id] = node
            self._node_to_id_new[node] = node_id

            # Copy decision expr and its id
            # self._id_to_expr_new[dec_id] = dec_expr
            # self._expr_to_id_new[dec_expr] = dec_id

            # Recurse
            self.copy_in_new_cache_node(node._high)
            self.copy_in_new_cache_node(node._low)
        else:
            self._id_to_node_new[node_id] = node
            self._node_to_id_new[node] = node_id

    """
    Export and import XADDs
    """

    def export_min_or_max_xadd(self, fname: str, min_or_max_node_id: int):
        """
        Export the min xadd to a file named `fname`.
        Some additional information is also exported, for example:
        """
        with open(fname, 'w+') as f:
            # Write information regarding problem definition
            f.write(f'Problem type: {self._problem_type}\n')
            f.write(f'Config file: {self._config_json_fname}\n')
            dec_vars = ', '.join([str(v) for v in sorted(self._free_var_set, key=lambda x: str(x))])
            f.write(f'Decision variables: {dec_vars}\n')
            bounds = ':'.join([f'{bound}' for v, bound in sorted(self._var_to_bound.items(), key=lambda x: str(x[0]))
                               if v in self._free_var_set])
            f.write(f'Bounds: {bounds}\n')

            # Write the LP
            f.write(f'\nlp: ')
        self.export_xadd(min_or_max_node_id, fname, append=True)

    def export_argmin_or_max_xadd(self, fname: str):
        """
        Export the argmin xadd of `min_var` variables to a file named `fname`.
        Some additional information is also exported: such as
            name of decision variables
            bounds of those variables
            dimensionality of parameters and features
        :param fname:       (str) the file name to which argmin(max) xadd is exported
        """
        # Write information regarding problem definition
        with open(fname, 'w+') as f:
            f.write(f'Problem type: {self._problem_type}\n')
            f.write(f'Config file: {self._config_json_fname}\n')
            dec_vars = ', '.join([str(v) for v in sorted(self._min_var_set, key=lambda x: str(x))])
            f.write(f'Decision variables: {dec_vars}\n')
            bounds = ':'.join([f'{bound}' for bound in sorted(
                [b for v, b in self._var_to_bound.items() if v in self._min_var_set], key=lambda x: str(x))])
            f.write(f'Bounds: {bounds}\n')
            if self._feature_dim is not None:
                f.write('Feature dimension: {}\n'.format(self._feature_dim))

        for var in self._min_var_set:
            with open(fname, 'a+') as f:
                f.write(f'\n{str(var)}: ')
            var_arg_min_or_max = self._var_to_anno.get(var)
            self.export_xadd(var_arg_min_or_max, fname, append=True)

    def export_xadd(self, node_id: int, fname: str, append=False):
        """
        Export the XADD node to a .xadd file.
        If append is True, then open the file in the append mode.
        """
        # Firstly, turn off printing node info
        node = self._id_to_node.get(node_id, None)
        if node is None:
            raise KeyError('There is no node with the node id {}'.format(node_id))
        node.turn_off_print_node_info()

        if append:
            with open(fname, 'a+') as f:
                f.write('\n')
                f.write(str(node))
        else:
            with open(fname, 'w+') as f:
                f.write(str(node))

    def import_xadd(self, fname=None, xadd_str=None, locals=None):
        """
        Import the XADD node defined in the input file or in a string.
        """
        assert (fname is not None and xadd_str is None) or \
               (fname is None and xadd_str is not None)

        if fname is not None:
            with open(fname, 'r') as f:
                xadd_str = f.read().replace('\n', '')
        # When it is just a leaf expression: not supported yet..
        if xadd_str.rfind('(') == 0 and xadd_str.rfind('[') == 2:
            xadd_as_list = [sympy.sympify(xadd_str.strip('( [] )'), locals=locals)]
        else:
            xadd_as_list = parse_xadd_grammar(xadd_str, ns=locals if locals is not None else {})[1][0]
        xadd = self.build_initial_xadd(xadd_as_list)
        return xadd

    def import_lp_xadd(self, fname, model_name=None, prob_instance: dict = None):
        """
        Read in the XADD corresponding to an LP.
        """
        ns = {}
        xadd_str = ''
        dec_var = None

        # Check existence of the file
        import os.path as path
        if not path.exists(fname):
            raise FileNotFoundError("File {} does not exist, raising an error".format(fname))
        with open(fname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line_split = line.split(':')
                if i == 0:
                    assert line_split[0].lower() == 'problem type', "Problem type should be specified"
                    logger.info(f"Problem type: {line_split[1].strip().lower()}")
                    self.set_problem(line_split[1].strip().lower())

                elif i == 1:
                    assert line_split[0].lower() == 'config file', "Path to the configuration file should be provided"
                    json_file = fname.replace('.xadd', '.json')
                    logger.info(line.strip())
                    self.link_json_file(json_file)

                elif i == 2:
                    assert line_split[
                               0].lower() == 'decision variables', "Decision variables should be specified in the file"
                    symbols = line_split[1].strip().split(',')
                    dec_vars = sympy.symbols(symbols)
                    dec_vars = list(
                        sorted(dec_vars, key=lambda x: (float(str(x).split("_")[0][1:]), float(str(x).split("_")[1]))
                        if len(str(x).split("_")) > 1 else float(str(x)[1:])))
                    ns.update({str(v): v for v in dec_vars})
                    logger.info(line.strip())
                    self._free_var_set = set(dec_vars)

                elif i == 3:
                    assert line_split[0].lower() == 'bounds', "Bound information should be provided"
                    bound_dict = {
                        ns[str(v)]: tuple(map(float, line_split[i + 1].strip()[1:-1].replace('oo', 'inf').split(',')))
                        for i, v in enumerate(dec_vars)}
                    self._var_to_bound.update(bound_dict)
                    logger.info(line.strip())

                elif len(line_split) > 1:
                    if len(xadd_str) > 0:
                        raise ValueError
                    xadd_str = line_split[1].strip()
                    obj = sympy.symbols(line_split[0])
                    ns.update({str(obj): obj})

                elif len(line.strip()) != 0:
                    xadd_str += line
            if len(xadd_str) > 0:
                lp = self.import_xadd(xadd_str=xadd_str, locals=ns)

        # Handle equality constraints if exist by looking at the configuration json file
        if prob_instance is None:
            try:
                import json
                with open(self._config_json_fname, "r") as json_file:
                    prob_instance = json.load(json_file)
            except:
                raise FileNotFoundError("Failed to open the configuration json file!")

        eq_constr_dict, dec_vars = compute_rref_filter_eq_constr(prob_instance['eq-constr'], dec_vars,
                                                                 locals=ns)
        dec_vars = [v for v in dec_vars if v not in eq_constr_dict]

        # Name space
        ns.update({str(v): v for v in eq_constr_dict})
        ns.update({str(v): v for v in dec_vars})
        self.update_name_space(ns)

        # LP objective node
        self.set_objective(lp)

        return dec_vars, eq_constr_dict

    def import_arg_xadd(self, fname, feature_dim=None, param_dim=None, model_name=None):
        """
        Read in XADDs corresponding to argmin solutions.
        """
        ns = {}
        xadd_str = ''
        dec_var = None

        # Check existence of the file
        import os.path as path
        if not path.exists(fname):
            raise FileNotFoundError("File {} does not exist, raising an error".format(fname))

        with open(fname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line_split = line.split(':')
                if i == 0:
                    assert line_split[0].lower() == 'problem type', "Problem type should be specified (knapsack or spo)"
                    assert line_split[1].strip().lower() in ('predict-then-optimize', 'knapsack', 'bilinear')

                    self.set_problem(line_split[1].strip().lower())

                elif i == 1:
                    assert line_split[0].lower() == 'config file', "Path to the configuration file should be provided"
                    json_file = fname.replace('_argmin.xadd', '.json')
                    self.link_json_file(json_file)

                elif i == 2:
                    assert line_split[
                               0].lower() == 'decision variables', "Decision variables should be specified in the file"
                    dec_vars = sympy.symbols(line_split[1].strip().replace(',', ' '))
                    dec_vars = list(
                        sorted(dec_vars, key=lambda x: (float(str(x).split("_")[0][1:]), float(str(x).split("_")[1]))
                        if len(str(x).split("_")) > 1 else float(str(x)[1:])))
                    ns.update({str(v): v for v in dec_vars})
                    self._min_var_set = set(dec_vars)
                    dim_dec_vars = len(self._min_var_set)

                elif i == 3:
                    assert line_split[0].lower() == 'bounds', "Bound information should be provided"
                    bound_dict = {ns[str(v)]: tuple(map(int, line_split[i + 1].strip()[1:-1].split(','))) for i, v in
                                  enumerate(dec_vars)}
                    self._var_to_bound.update(bound_dict)


                elif i == 4 and line_split[0].lower() == 'feature dimension':

                    # assert line_split[0].lower() == 'feature dimension', "Feature dimension should be specified"

                    if feature_dim is None:
                        feature_dim = tuple(int(i) for i in line_split[1].strip().strip('(').strip(')').split(',') if i)

                    is_predopt = True if self._problem_type == 'predict-then-optimize' else False
                    assert (len(feature_dim) == 1 and is_predopt) or (len(feature_dim) == 2 and not is_predopt)

                    self.set_feature_dim(feature_dim)
                    if param_dim is None:
                        param_dim = (dim_dec_vars, feature_dim[0] + 1) if is_predopt else (feature_dim[1] + 1,)
                    self.set_param_dim(param_dim)
                    # ns = self.getPredictedCoeffs(locals=ns)

                    self.update_name_space(ns)

                elif len(line_split) > 1:
                    if len(xadd_str) > 0 and dec_var is not None:
                        self._var_to_anno[dec_var] = self.import_xadd(xadd_str=xadd_str, locals=ns)
                        xadd_str = ''
                    xadd_str = line_split[1].strip()
                    dec_var = ns[line_split[0]]

                elif len(line.strip()) != 0:
                    xadd_str += line
            if len(xadd_str) > 0 and dec_var is not None:
                self._var_to_anno[dec_var] = self.import_xadd(xadd_str=xadd_str, locals=ns)
        # Handle equality constraints if exist by looking at the configuration json file
        try:
            import json
            with open(self._config_json_fname, "r") as json_file:
                prob_instance = json.load(json_file)
        except:
            raise FileNotFoundError("Failed to open the configuration json file!")

        eq_constr_dict, dec_vars = compute_rref_filter_eq_constr(prob_instance['eq-constr'], dec_vars, locals=ns)
        dec_vars = [v for v in dec_vars if v not in eq_constr_dict]

        return dec_vars, eq_constr_dict

    def var_count(self, node_id, count=None):
        if count == None:
            count = dict()

        node = self.get_exist_node(node_id)
        expr = node.expr if node._is_leaf else self._id_to_expr.get(node.dec, None)

        if not node._is_leaf:
            self.var_count(node._low, count)
            self.var_count(node._high, count)

        for s in expr.free_symbols:
            if str(s).startswith('x'):
                count[s] = count.get(s, 0) + 1

        return count


class NullDec:
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, NullDec)


class XADDLeafOperation(metaclass=abc.ABCMeta):
    def __init__(self, context):
        self._context = context

    @abc.abstractmethod
    def process_xadd_leaf(self, decisions, decision_values, leaf_val):
        pass


class DeltaFunctionSubstitution(XADDLeafOperation):
    def __init__(self, sub_var, xadd_sub_at_leaves, context):
        """
        From the case statement of 'xadd_sub_at_leaves', all occurrences of sub_var will be replaced.
        """
        super().__init__(context)
        self._leafSubs = {}
        self._xadd_sub_at_leaves = xadd_sub_at_leaves
        self._subVar = sub_var

    def process_xadd_leaf(self, decisions, decision_values, leaf_val):
        self._leafSubs = {}
        # If boolean variable, handle differently
        if self._subVar in self._context._bool_var_set:
            self._leafSubs[self._subVar] = True if leaf_val == 1 else False
            return self._context.substitute_bool_vars(self._xadd_sub_at_leaves, self._leafSubs)
        elif leaf_val == sympy.nan:
            return self._context.NAN
        else:
            self._leafSubs[self._subVar] = leaf_val
            ret = self._context.substitute(self._xadd_sub_at_leaves, self._leafSubs)
            ret = self._context.reduce_lp(ret)
            return ret


class XADDLeafMultivariateMinOrMax(XADDLeafOperation):
    def __init__(self, var_lst, is_max, bound_dict, context, annotate):
        super().__init__(context)
        self._var_lst = var_lst
        self._context._opt_var_lst = var_lst
        self._is_max = is_max
        self.bound_dict = bound_dict
        self._running_result = -1
        self._annotate = annotate

    @property
    def _var(self):
        return self._var_lst[0]

    @property
    def _lower_bound(self):
        return self.bound_dict[self._var][0]

    @property
    def _upper_bound(self):
        return self.bound_dict[self._var][1]

    def process_xadd_leaf(self, decisions, decision_values, leaf_val):
        """
        :param decisions:
        :param decision_values:
        :param leaf_val:        (sympy.Basic) leaf expression
        :return:
        """
        # Check if below computation is unnecessary
        # min(oo, oo) = oo; max(oo, oo) = oo; min(-oo, -oo) = -oo; max(-oo, -oo) = -oo;
        # But, argmax and argmin are ambiguous in these cases, and so we simply annotate them with NaN
        if leaf_val == oo or leaf_val == -oo:
            min_max_eval = self._context.get_leaf_node(leaf_val, annotation=self._context.NAN)

            # Compare with the running result
            if self._running_result == -1:
                self._running_result = min_max_eval
            return self._context.get_leaf_node(leaf_val)

        # Bound management
        lower_bound = []
        upper_bound = []
        lower_bound.append(sympy.S(self._lower_bound))
        upper_bound.append(sympy.S(self._upper_bound))

        # Independent decisions (incorporated later): [(dec_expr, bool)]
        target_var_indep_decisions = []

        # Get lower and upper bounds over the variable
        for dec_expr, is_true in zip(decisions, decision_values):
            # Check boolean decisions or if self._var in dec_expr
            if (dec_expr in self._context._bool_var_set) or (self._var not in dec_expr.atoms()):
                target_var_indep_decisions.append((dec_expr, is_true))
                continue

            lhs, rhs, gt = dec_expr.lhs, dec_expr.rhs, isinstance(dec_expr, relational.GreaterThan)
            gt = (gt and is_true) or (not gt and not is_true)
            expr = lhs >= rhs if gt else lhs <= rhs

            # Get bounds over 'var'
            bound_expr, upper = xaddpy.utils.util.get_bound(self._var, expr)
            if upper:
                upper_bound.append(bound_expr)
            else:
                lower_bound.append(bound_expr)

        # lower bound over 'var' is the maximum among lower bounds
        xadd_lower_bound = -1
        for e in lower_bound:
            xadd_lower_bound = self._context.get_leaf_node(e) if xadd_lower_bound == -1 \
                else self._context.apply(xadd_lower_bound, self._context.get_leaf_node(e), op='max')

        xadd_upper_bound = -1
        for e in upper_bound:
            xadd_upper_bound = self._context.get_leaf_node(e) if xadd_upper_bound == -1 \
                else self._context.apply(xadd_upper_bound, self._context.get_leaf_node(e), op='min')

        # Reduce lower and upper bound xadds for potential computational gains
        xadd_lower_bound = self._context.reduce_lp(xadd_lower_bound)
        xadd_upper_bound = self._context.reduce_lp(xadd_upper_bound)

        # Ensure lower bounds are smaller than upper bounds
        for e1 in lower_bound:
            for e2 in upper_bound:
                comp = (e2 - e1 >= 0)  # ub - lb
                if comp == sympy.S.true or \
                        e2 == oo or e1 == -oo:
                    continue
                target_var_indep_decisions.append((comp, True))
                assert isinstance(comp, relational.GreaterThan)

        # Substitute lower and upper bounds into leaf node
        eval_lower = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_lower_bound)
        eval_upper = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_upper_bound)

        # Take casemin / casemax of eval_lower and eval_upper
        """
        If `leaf_val` is bilinear, then we know that a leaf value of `eval_upper - eval_lower` will factorize as 
            (ub_vj - lb_vj) * (d_vj + \sum_i x_i Q_ij) and that (ub_vj - lb_vj) >= 0
        Therefore, we simply need to add the following conditional:
            ( [d_vj + \sum_i x_i Q_ij <= 0]
                ( [eval_upper] )
                ( [eval_lower] ))
        This can be done via the following trick:
            Let A = ( [d_vj + \sum_i x_i Q_ij <= 0], and B = ( [d_vj + \sum_i x_i Q_ij <= 0]
                        ( [1] )                                  ( [0] )
                        ( [0] ))                                 ( [1] ))
            Then, consider ``C = A \oprod `eval_upper`` and ``D = B \oprod `eval_lower``.
            The desired result can be obtained by 
                C \oplus D
            Then, we should canonicalize the resulting node. 
        """
        is_bilinear = xaddpy.utils.util.is_bilinear(leaf_val)
        expr = 0
        if is_bilinear:
            # Get the expression multiplied to `self._var`
            expr = xaddpy.utils.util.get_multiplied_expr(leaf_val, self._var)
        if is_bilinear and expr != 0:
            dec_expr = expr <= 0
            if dec_expr == sympy.S.true:
                min_max_eval = eval_upper
            elif dec_expr == sympy.S.false:
                min_max_eval = eval_lower
            else:
                dec, is_reversed = self._context.get_dec_expr_index(dec_expr, create=True)
                ind_true = self._context.get_internal_node(dec, self._context.ZERO_ig, self._context.ONE)
                ind_false = self._context.get_internal_node(dec, self._context.ONE, self._context.ZERO_ig)
                upper_half = self._context.apply(ind_true if not is_reversed else ind_false, eval_upper, 'prod')
                lower_half = self._context.apply(ind_false if not is_reversed else ind_true, eval_lower, 'prod')
                min_max_eval = self._context.apply(upper_half, lower_half, 'sum',
                                                   annotation=(
                                                   xadd_upper_bound, xadd_lower_bound) if self._annotate else None)
                min_max_eval = self._context.make_canonical(min_max_eval)
        else:
            # Note: always 1st argument should be upper bound, while 2nd argument is lower bound
            min_max_eval = self._context.apply(eval_upper, eval_lower, 'max' if self._is_max else 'min',
                                               annotation=(
                                               xadd_upper_bound, xadd_lower_bound) if self._annotate else None)

        # Reduce LP
        min_max_eval = self._context.reduce_lp(min_max_eval)
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = False

        # Incorporate independent decisions
        for d, b in target_var_indep_decisions:
            high_val = oo if (b and self._is_max) or (not b and not self._is_max) \
                else -oo
            low_val = -oo if (b and self._is_max) or (not b and not self._is_max) \
                else oo
            indep_constraint = self._context.get_dec_node(d, low_val, high_val)
            # Note 'min' and 'max' are swapped below: ensuring non-valid paths result in infinite penalty
            min_max_eval = self._context.apply(indep_constraint, min_max_eval, 'min' if self._is_max else 'max')
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Reduce
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = True
            min_max_eval = self._context.reduce_lp(min_max_eval)

        """
        Min(max)imize out remaining variables
        """
        if len(self._var_lst) > 1:
            min_or_max = XADDLeafMultivariateMinOrMax(
                self._var_lst[1:],
                is_max=self._is_max,
                bound_dict=self.bound_dict,
                context=self._context,
                annotate=self._annotate,
            )
            decisions, decision_values = [], []
            _ = self._context.reduce_process_xadd_leaf(min_max_eval, min_or_max, decisions, decision_values)
            min_max_eval = min_or_max._running_result

        if self._running_result == -1:
            self._running_result = min_max_eval
        else:
            self._running_result = self._context.apply(self._running_result, min_max_eval,
                                                       'max' if self._is_max else 'min')
            self._running_result = self._context.reduce_lp(self._running_result)
        #
        # # Compare with the running result
        # if self._running_result == -1:
        #     self._running_result = min_max_eval
        # else:
        #     self._running_result = self._context.apply(self._running_result, min_max_eval, 'max' if self._is_max else 'min')
        #
        # # Reduce running result
        # self._running_result = self._context.reduce_lp(self._running_result)

        return self._context.get_leaf_node(leaf_val)


class XADDLeafMinOrMax(XADDLeafOperation):
    def __init__(self, var, is_max, bound_dict, context):
        super().__init__(context)
        self._var = var
        self._context._opt_var = var
        self._is_max = is_max
        self._running_result = -1
        if var in bound_dict:
            self._lower_bound = bound_dict[var][0]
            self._upper_bound = bound_dict[var][1]
        else:
            print("No domain bounds over {} are provided... using -oo and oo as lower and upper bounds.".format(var))
            self._lower_bound = -oo
            self._upper_bound = oo

    def process_xadd_leaf(self, decisions, decision_values, leaf_val):
        """
        :param decisions:
        :param decision_values:
        :param leaf_val:        (sympy.Basic) leaf expression
        :return:
        """
        # Check if below computation is unnecessary
        # min(oo, oo) = oo; max(oo, oo) = oo; min(-oo, -oo) = -oo; max(-oo, -oo) = -oo;
        # But, argmax and argmin are ambiguous in these cases, and so we simply annotate them with NaN
        if leaf_val == oo or leaf_val == -oo:
            min_max_eval = self._context.get_leaf_node(leaf_val, annotation=self._context.NAN)

            # Compare with the running result
            if self._running_result == -1:
                self._running_result = min_max_eval
            return self._context.get_leaf_node(leaf_val)

        # Bound management
        lower_bound = []
        upper_bound = []
        lower_bound.append(sympy.S(self._lower_bound))
        upper_bound.append(sympy.S(self._upper_bound))

        # Independent decisions (incorporated later): [(dec_expr, bool)]
        target_var_indep_decisions = []

        # Get lower and upper bounds over the variable
        for dec_expr, is_true in zip(decisions, decision_values):
            # Check boolean decisions or if self._var in dec_expr
            if (dec_expr in self._context._bool_var_set) or (self._var not in dec_expr.atoms()):
                target_var_indep_decisions.append((dec_expr, is_true))
                continue

            lhs, rhs, gt = dec_expr.lhs, dec_expr.rhs, isinstance(dec_expr, relational.GreaterThan)
            gt = (gt and is_true) or (not gt and not is_true)
            expr = lhs >= rhs if gt else lhs <= rhs

            # Get bounds over 'var'
            bound_expr, upper = xaddpy.utils.util.get_bound(self._var, expr)
            if upper:
                upper_bound.append(bound_expr)
            else:
                lower_bound.append(bound_expr)

        # lower bound over 'var' is the maximum among lower bounds
        xadd_lower_bound = -1
        for e in lower_bound:
            xadd_lower_bound = self._context.get_leaf_node(e) if xadd_lower_bound == -1 \
                else self._context.apply(xadd_lower_bound, self._context.get_leaf_node(e), op='max')

        xadd_upper_bound = -1
        for e in upper_bound:
            xadd_upper_bound = self._context.get_leaf_node(e) if xadd_upper_bound == -1 \
                else self._context.apply(xadd_upper_bound, self._context.get_leaf_node(e), op='min')

        # Reduce lower and upper bound xadds for potential computational gains
        xadd_lower_bound = self._context.reduce_lp(xadd_lower_bound)
        xadd_upper_bound = self._context.reduce_lp(xadd_upper_bound)

        # Ensure lower bounds are smaller than upper bounds
        for e1 in lower_bound:
            for e2 in upper_bound:
                comp = (e2 - e1 >= 0)  # ub - lb
                if comp == sympy.S.true or \
                        e2 == oo or e1 == -oo:
                    continue
                target_var_indep_decisions.append((comp, True))
                assert isinstance(comp, relational.GreaterThan)
                # comp_lhs, is_reversed = self._context.clean_up_expr(comp.lhs, factor=True)
                # self._context._temp_ub_lb_cache.add(comp_lhs if not is_reversed else -comp_lhs)

        # Substitute lower and upper bounds into leaf node
        eval_lower = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_lower_bound)
        eval_upper = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_upper_bound)

        # Take casemin / casemax of eval_lower and eval_upper
        """
        If `leaf_val` is bilinear, then we know that a leaf value of `eval_upper - eval_lower` will factorize as 
            (ub_vj - lb_vj) * (d_vj + \sum_i x_i Q_ij) and that (ub_vj - lb_vj) >= 0
        Therefore, we simply need to add the following conditional:
            ( [d_vj + \sum_i x_i Q_ij <= 0]
                ( [eval_upper] )   anno: xadd_upper_bound
                ( [eval_lower] )   anno: xadd_lower_bound
            )
        This can be done via the following trick:
            Let A = ( [d_vj + \sum_i x_i Q_ij <= 0], and B = ( [d_vj + \sum_i x_i Q_ij <= 0]
                        ( [1] )                                  ( [0] )
                        ( [0] ))                                 ( [1] ))
            Then, consider ``C = A \oprod `eval_upper`` and ``D = B \oprod `eval_lower``.
            The desired result can be obtained by 
                C \oplus D
            Then, we should canonicalize the resulting node. 
        """
        is_bilinear = xaddpy.utils.util.is_bilinear(leaf_val)
        expr = 0
        if is_bilinear:
            # Get the expression multiplied to `self._var`
            expr = xaddpy.utils.util.get_multiplied_expr(leaf_val, self._var)
        if is_bilinear and expr != 0:
            dec_expr = expr <= 0
            if dec_expr == sympy.S.true:
                min_max_eval = eval_upper
            elif dec_expr == sympy.S.false:
                min_max_eval = eval_lower
            else:
                dec, is_reversed = self._context.get_dec_expr_index(dec_expr, create=True)
                ind_true = self._context.get_internal_node(dec, self._context.ZERO_ig, self._context.ONE)
                ind_false = self._context.get_internal_node(dec, self._context.ONE, self._context.ZERO_ig)
                upper_half = self._context.apply(ind_true if not is_reversed else ind_false, eval_upper, 'prod')
                lower_half = self._context.apply(ind_false if not is_reversed else ind_true, eval_lower, 'prod')
                min_max_eval = self._context.apply(upper_half, lower_half, 'sum',
                                                   annotation=(xadd_upper_bound, xadd_lower_bound))
                min_max_eval = self._context.make_canonical(min_max_eval)
        else:
            # Note: always 1st argument should be upper bound, while 2nd argument is lower bound
            min_max_eval = self._context.apply(eval_upper, eval_lower, 'max' if self._is_max else 'min',
                                               annotation=(xadd_upper_bound, xadd_lower_bound))
        # self._context._temp_ub_lb_cache.clear()

        # Reduce LP
        min_max_eval = self._context.reduce_lp(min_max_eval)
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = False

        # Incorporate independent decisions
        for d, b in target_var_indep_decisions:
            high_val = oo if (b and self._is_max) or (not b and not self._is_max) \
                else -oo
            low_val = -oo if (b and self._is_max) or (not b and not self._is_max) \
                else oo
            indep_constraint = self._context.get_dec_node(d, low_val, high_val)
            # Note 'min' and 'max' are swapped below: ensuring non-valid paths result in infinite penalty
            min_max_eval = self._context.apply(indep_constraint, min_max_eval, 'min' if self._is_max else 'max')
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Reduce
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = True
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Compare with the running result
        if self._running_result == -1:
            self._running_result = min_max_eval
        else:
            self._running_result = self._context.apply(self._running_result, min_max_eval,
                                                       'max' if self._is_max else 'min')

        # Reduce running result
        self._running_result = self._context.reduce_lp(self._running_result)

        return self._context.get_leaf_node(leaf_val)