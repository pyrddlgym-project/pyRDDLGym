import sympy.core.relational as relational
import sympy
from sympy import oo
import abc


class Node(metaclass=abc.ABCMeta):
    def __init__(self, context):
        self._context = context
        self._print_node_info = True

    def __str__(self):
        return

    def collect_vars(self, var_set=None):
        """
        :param var_set:        (set)
        :return:
        """
        pass

    def collect_nodes(self, nodes=None):
        """
        :param nodes:       (set)
        :return:
        """
        pass


class XADDTNode(Node):
    def __init__(self, expr, annotation=None, context=None):
        """
        A leaf XADD node implementation. Annotation can be tracked. Need to provide integer ids for
        leaf expression, node id, and annotation (if not None).
        :param expr:            (sympy.Basic) XADDTNode receives symbolic expression not integer id
        :param annotation:        (int)
        """
        # Link the node with XADD
        assert context is not None, "XADD should be passed when instantiating nodes!"
        super().__init__(context)

        # Set the expression associated with the leaf node
        self.expr = expr

        # Set the annotation
        self._annotation = annotation

        # Flag the node as leaf
        self._is_leaf = True

    def turn_off_print_node_info(self):
        self._print_node_info = False

    def turn_on_print_node_info(self):
        self._print_node_info = True

    def set(self, expr, annotation):
        """
        Set expression and annotation.
        :param expr:        (sympy.Basic) Symbolic expression
        :param annotation:    (int) id of annotation
        """
        self.expr = expr
        self._annotation = annotation

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr):
        assert isinstance(expr, sympy.Basic) or expr == oo or expr == -oo, "expr should be a Sympy object for XADDTNode!"
        self._expr = expr

    @property
    def _annotation(self):
        return self.__annotation

    @_annotation.setter
    def _annotation(self, annotation):
        self.__annotation = annotation

    def collect_vars(self, var_set=None):
        """
        Return a set containing all sympy Symbols.
        """
        if var_set is not None:
            var_set.update(self.expr.free_symbols)
        else:
            var_set = self.expr.free_symbols
        return var_set

    def collect_nodes(self, nodes=None):
        if nodes is not None:
            nodes.add(self)
        else:
            nodes = set()
            nodes.add(self)

    def __hash__(self):
        if self._annotation is None:
            return hash(self.expr)
        else:
            return hash((self.expr, self._annotation))

    def __eq__(self, other):
        if self._annotation is None:
            return other._annotation is None and self.expr == other.expr
        else:
            return self.expr == other.expr and self._annotation == other._annotation

    def __str__(self, level=0):
        # curr_node_expr = self.expr
        str_expr = "( [{}] )".format(self.expr)
        str_node_id = " node_id: {}".format(self._context._node_to_id.get(self))
        str_anno = " anno: {}".format(self._annotation) if self._annotation is not None else ""
        if self._print_node_info:
            return str_expr + str_node_id + str_anno
        else:
            return str_expr
        #
        # ret = "\t" * level + str(self.expr) + "\n"
        # if not self._is_leaf:
        #     low = cache.id_to_node[self.low]
        #     high = cache.id_to_node[self.high]
        #     ret += high.__str__(level + 1)
        #     ret += low.__str__(level+1)
        # else:
        #
        #     res_str = "{}".format(str(self.expr))
        # return ret

    def __repr__(self, level=0):
        # curr_node_expr = self.expr
        str_expr = "( [{}] )".format(self.expr)
        str_node_id = " node_id: {}".format(self._context._node_to_id.get(self))
        str_anno = " anno: {}".format(self._annotation) if self._annotation is not None else ""
        if self._print_node_info:
            return str_expr + str_node_id + str_anno
        else:
            return str_expr


class XADDINode(Node):
    def __init__(self, dec, low=None, high=None, degree=2, context=None):
        """
        Basic decision node of a tree case function.
        The value is a Sympy inequality expression, and the low and the high branches correspond to
        False and True, respectively. Each node will have a unique identifier (integer) stored in a dictionary
        as an attribute in a XADD object.
        :param dec:     (int) Decision expression in a canonical form (rhs is a number, lhs contains variables)
        :param low:     (int) False branch
        :param high:    (int) True branch
        :param degree:  (int) Tree degree (default: 2, when linked list: 1)
        """
        # Link the node with XADD
        assert context is not None, "XADD should be passed when instantiating nodes!"
        super().__init__(context)

        self.dec = dec

        # Flag for checking if a leaf Node
        self._is_leaf = False

        # degree == 1 is used for linked list object (only at the beginning)
        if degree == 1:
            self.next_node = None
        elif degree == 2:
            self._low = low
            self._high = high
        else:
            raise ValueError("Degree > 2 is not defined")

        self.degree = degree

    def turn_off_print_node_info(self):
        self._print_node_info = False
        high = self._context._id_to_node.get(self._high, None)
        if high is not None:
            high.turn_off_print_node_info()

        low = self._context._id_to_node.get(self._low, None)
        if low is not None:
            low.turn_off_print_node_info()

    def turn_on_print_node_info(self):
        self._print_node_info = True
        high = self._context._id_to_node.get(self._high, None)
        if high is not None:
            high.turn_on_print_node_info()

        low = self._context._id_to_node.get(self._low, None)
        if low is not None:
            low.turn_on_print_node_info()

    def set(self, dec_id, low, high):
        self.dec = dec_id
        self._low = low
        self._high = high

    @property
    def dec(self):
        return self._dec

    @dec.setter
    def dec(self, dec):
        assert isinstance(dec, int)
        self._dec = dec
        # # ensure canonical form when setting
        # assert isinstance(expr, sympy.Basic) or expr == oo or expr == -oo, "expr should be a Sympy object!"
        #
        # if isinstance(expr, relational.Rel):
        #     # Basic canonical form of inequality from Sympy
        #     expr = expr.canonical
        #
        #     # check whether the relation is 'greater than (>=)'
        #     self._gt = True if isinstance(expr, relational.GreaterThan) else False
        # else:
        #     # When a terminal node: a linear expression
        #     expr = sympy.simplify(expr)
        # self._expr = expr
        # self._atoms = expr.atoms()

    def get_next(self):
        """
        Return next node of a node in a linked list. Raise error when called from a binary tree.
        """
        assert self.degree == 1, "Cannot call get_next method from a binary tree!"
        if not self._is_leaf:
            return self.next_node
        else:
            raise ValueError("Cannot call get_next method from a leaf node!")

    def set_next(self, node):
        """
        Set next node of a node in a linked list.
        :param node:        (Node)
        """
        self.next_node = node       # set the next node

    def get_low_child(self):
        return self._context._id_to_node[self._low]

    def get_high_child(self):
        return self._context._id_to_node[self._high]

    def get_expr(self):
        return self._context._id_to_expr[self.expr]

    def get_bound(self, var):
        """
        Return either lower bound or upper bound of 'var' from self.expr.
        :param var:     (sympy.Symbol) target variable
        :return:        (sympy.Basic, bool) a sympy expression along with the boolean value indicating whether an upper
                        or lower bound. True for upper bound, False for lower bound.
        """
        # note expr is in canonical form: expression (<=, >=, <, >) number; expression does not have negative sign;
        # variables in expression are ordered.
        comp = sympy.solve(self.expr, var)              # Solve for 'var'
        # Check if only a number is multiplied to 'var'. In this case, the result is of form 'var <=, >= expression'
        # plus 'var > - infty' or 'var < infty'. We do not need the finiteness condition, so remove it.
        if isinstance(comp, sympy.And):
            assert len(comp.args) == 2, "No more than 3 terms should be generated as a result of solve(ineq)!"
            args1rhs = comp.args[1].canonical.rhs
            i = 0 if (args1rhs == oo) or (args1rhs == -oo) else 1
            comp = comp.args[i]
        else:
            comp = sympy.simplify(comp)
        # otherwise, some variables are multiplied to 'var'. Need to condition on this term being either positive or
        # negative to determine the result
        # else:
        #     print("Expression: {}".format(comp))
        #     raise ValueError("This case is not expected when getting bounds!")
            # res = 1
            # res = [res * term for term in expr.lhs.atoms() if term != var]

        # canonical form
        comp = comp.canonical
        # check whether upper bound or lower bound over 'var'
        ub = isinstance(comp, relational.LessThan)        # if ub: 'var' <= upper bound
        expr = comp.rhs
        return expr, ub

    def collect_nodes(self, nodes=None):
        if self in nodes:
            return
        nodes.add(self)
        self._context.get_exist_node(self._low).collect_nodes(nodes)
        self._context.get_exist_node(self._high).collect_nodes(nodes)

    def collect_vars(self, var_set=None):
        # Check cache
        vars2 = self._context._inode_to_vars.get(self, None)
        if vars2 is not None:
            var_set.update(vars2)
            return var_set
        low = self._context.get_exist_node(self._low)
        high = self._context.get_exist_node(self._high)
        expr = self._context._id_to_expr[self.dec]
        var_set.update(expr.free_symbols)
        low.collect_vars(var_set)
        high.collect_vars(var_set)

        self._context._inode_to_vars[self] = var_set.copy()
        return var_set

    def __hash__(self):
        """
        Note that the terminal node and internal node are used differently in comparing keys in dictionary.
        """
        if self.degree == 2:
            return hash((self.dec, self._low, self._high))
        elif self.degree == 1:
            return hash((self.dec, self.next_node))

    def __eq__(self, other):
        if isinstance(other, XADDINode):
            if self.degree == 2:
                return (self.dec == other.dec) and (self._low == other._low) and (self._high == other._high)
            elif self.degree == 1:
                return (self.dec, self.next_node) == (other.dec, other.next_node)
        else:
            return False

    def __str__(self, level=0):
        ret = ""
        ret += "( [{}]".format(self._context._id_to_expr[self.dec])

        # print node id
        if self._print_node_info:
            ret += " (dec, id): {}, {}".format(self.dec, self._context._node_to_id.get(self))

        # Node level cache
        high = self._context._id_to_node.get(self._high, None)
        if high is not None:
            ret += "\n" + "\t"*(level+1) + " {} ".format(high.__str__(level+1))
        else:
            ret += "h:[None] "

        low = self._context._id_to_node.get(self._low, None)
        if low is not None:
            ret += "\n" + "\t" * (level + 1) + " {} ".format(low.__str__(level + 1))
        else:
            ret += "l:[None] "
        ret += ") "
        return ret

    def __repr__(self, level=0):
        ret = ""
        ret += "( [{}]".format(self._context._id_to_expr[self.dec])

        # print node id
        if self._print_node_info:
            ret += " (dec, id): {}, {}".format(self.dec, self._context._node_to_id.get(self))

        # Node level cache
        high = self._context._id_to_node.get(self._high, None)
        if high is not None:
            ret += "\n" + "\t"*(level+1) + " {} ".format(high.__str__(level+1))
        else:
            ret += "h:[None] "

        low = self._context._id_to_node.get(self._low, None)
        if low is not None:
            ret += "\n" + "\t" * (level + 1) + " {} ".format(low.__str__(level + 1))
        else:
            ret += "h:[None] "
        ret += ") "
        return ret