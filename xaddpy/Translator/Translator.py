import random

from xaddpy.xadd.xadd import XADD
import sympy as sp
import sys


class XADDTranslator():
    def __init__(self, ast):
        self.context = XADD()
        self.pvars = {}
        self.pvars_order = []
        self.ast = ast
        self.cpf_dic = {}
        # generate pvar dic
        # states
        for state in ast.state_fluent_variables:
            self.pvars[state[1][0]] = sp.S(state[1][0])
            self.pvars_order.append(state[1][0])
        # actions
        for action in ast.action_fluent_variables:
            self.pvars[action[1][0]] = sp.S(action[1][0])
            self.pvars_order.append(action[1][0])
        # TODO: derived and interim

    def Translate(self):
        self.cpf_dic = {}

        cpfs = self.ast.domain.cpfs[-1]
        for cpf in cpfs:
            pvar = cpf.pvar[1][0]
            if pvar not in self.cpf_dic:
                xadd_as_list = self.__build_expr_tree(cpf.expr)
                node = self.context.build_initial_xadd(xadd_as_list)
                self.cpf_dic[cpf.pvar[1][0]] = node

        # for key, value in self.cpf_dic.items():
        #     print(self.context.get_exist_node(value))

        assignment = {}
        for key, value in self.pvars.items():
            assignment[value] = random.choice([0, 1])
        print(assignment)

        key = list(self.cpf_dic.keys())[2]
        print(self.context.get_exist_node(self.cpf_dic[key]))
        a = self.context.substitute(self.cpf_dic[key], assignment)
        a2 = self.context.get_exist_node(a)
        print('leaf:', a2)
        print('value:', a2.expr)
        print('type:', type(a2))
        a3 = float(2.0)
        print(a3)
        return


    def __build_expr_tree(self, expr):
        tree_as_list = self.__scan_expr_tree(expr)
        return tree_as_list

    def __scan_expr_tree(self, expr):
        # leaf options
        if expr.etype[0] == 'constant':
            num = sp.S(expr.args)
            return [num]
        if expr.etype[0] == 'randomvar':
            return self.__scan_expr_tree(expr.args[0])
        if expr.etype[0] == 'pvar':
            # return self.__format_pvar_args(expr.args)
            return self.pvars[expr.etype[1]]
        if expr.etype[0] == 'boolean':
            op = expr.etype[1]
            if op == '~':
                right_expr = self.__scan_expr_tree(expr.args[0])
                cond = self.__get_boolean_condition(None, op, right_expr)
            else:
                left_expr = self.__scan_expr_tree(expr.args[0])
                right_expr = self.__scan_expr_tree(expr.args[1])
                cond = self.__get_boolean_condition(left_expr, op, right_expr)
            return cond
        if expr.etype[0] == 'control':
            # if
            if_xadd = self.__parse_if_condition(expr.args[0])
            # then
            then_xadd = self.__scan_expr_tree(expr.args[1])
            if not isinstance(then_xadd,list):
                then_xadd = [then_xadd]
            # else
            else_xadd = self.__scan_expr_tree(expr.args[2])
            if not isinstance(else_xadd, list):
                else_xadd = [else_xadd]
            branch = [if_xadd, then_xadd, else_xadd]
            return branch
        if expr.etype[0] == 'arithmetic':
            return sp.S(0)
        else:
            return sp.S(0)
        return 0

    def __get_boolean_condition(self, arg1, op, arg2):
        dec_expr = None
        if op == '~':
            dec_expr = arg2 <= 0.5
        elif op == '^':
            print(arg1, arg2)
            dec_expr = arg1 + arg2 >= 1.5
            print(dec_expr)
        elif op == '<=>':
            true_true = [arg2 >= 0.5, [sp.S(1)], [sp.S(0)]]
            false_ture = [arg2 >= 0.5, [sp.S(1)], [sp.S(0)]]
            dec_expr = [arg1 >= 0.5, true_true, false_ture]
        return dec_expr

    def __parse_if_condition(self, expr):
        if expr.etype[0] == 'pvar':
            cond = self.pvars[expr.etype[1]] >= 0.5
            # return cond
        elif expr.etype[0] == 'boolean':
            op = expr.etype[1]
            if op == '~':
                right_expr = self.__scan_expr_tree(expr.args[0])
                cond = self.__get_boolean_condition(None, op, right_expr)
            else:
                left_expr = self.__scan_expr_tree(expr.args[0])
                right_expr = self.__scan_expr_tree(expr.args[1])
                cond = self.__get_boolean_condition(left_expr, op, right_expr)
        return cond