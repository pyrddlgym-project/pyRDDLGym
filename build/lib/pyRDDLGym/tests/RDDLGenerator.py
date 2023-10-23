class RDDLGenerator(object):
    def __init__(self, RDDL_AST):
        self.buffer = ""
        self.AST = RDDL_AST

    def GenerateRDDL(self):
        d = self.AST.domain
        self.buffer += self.__print_block_header(d.name)
        self.buffer += '\n'

        # requirements
        if d.requirements:
            self.buffer += self.__print_requirements(d.requirements)
        self.buffer += '\n\n'

        # types
        if d.types:
            self.buffer += self.__print_types([t[0] for t in d.types])
        self.buffer += '\n\n'

        # pvars
        self.buffer += self.__print_pvariables(d)
        self.buffer += '\n\n'

        # cpfs
        self.buffer += self.__print_cpfs(d)
        self.buffer += '\n\n'

        # reward
        self.buffer += self.__print_reward(d.reward)
        self.buffer += '\n\n'

        # constraints
        self.buffer += self.__print_constraints(d)

        self.buffer += '};\n'

        return self.buffer

    def __tabs(self, level):
        t = ''
        for i in range(level):
            t += '\t'
        return t

    def __print_block_header(self, name):
        return 'domain {} '.format(name) + '{'

    def __print_requirements(self, reqs):
        s = self.__tabs(1) + 'requirements = {\n'
        for req in reqs:
            s += self.__tabs(2) + req + ',\n'
        s = s[:-2]
        s += '\n' + self.__tabs(1) + '};'
        return s

    def __print_types(self, types):
        s  = self.__tabs(1) + 'types {\n'
        for type in types:
            s += self.__tabs(2) + type + ': object;\n'
        s += self.__tabs(1) + '};'
        return s

    def __print_pvariables(self, d):
        s = self.__tabs(1) + 'pvariables {\n'
        s += self.__print_typed_pvariables('non-fluent', d.non_fluents)
        s += '\n'
        s += self.__print_typed_pvariables('state-fluent', d.state_fluents)
        s += '\n'
        s += self.__print_typed_pvariables('derived-fluent', d.derived_fluents)
        s += '\n'
        s += self.__print_typed_pvariables('interm-fluent', d.intermediate_fluents)
        s += '\n'
        s += self.__print_typed_pvariables('action-fluent', d.action_fluents)
        s += self.__tabs(1) + '};'
        return s

    def __print_typed_pvariables(self, pvar_type, pvariables):
        s = ""
        for name, pvar in pvariables.items():
            s += self.__tabs(2)
            params = ''
            if pvar.arity > 0:
                params = '(' + ', '.join(pvar.param_types) + ')'
            if pvar.is_derived_fluent() or pvar.is_intermediate_fluent():
                if pvar.level:
                    s += pvar.name + params + ' : {' + pvar_type + ', ' + pvar.range + ', ' + 'level=' + str(pvar.level) + '}\n'
                else:
                    s += pvar.name + params + ' : {' + pvar_type + ', ' + pvar.range + '}\n'
            else:
                s += pvar.name + params + ' : {' + pvar_type + ', ' + pvar.range + ', ' + 'default=' + str(pvar.default) + '}\n'
        return s

    def __print_cpfs(self, d):
        s = self.__tabs(1) + "cpfs {\n"
        s += self.__print_typed_cpfs(d.state_cpfs)
        s += '\n'
        s += self.__print_typed_cpfs(d.intermediate_cpfs)
        s += '\n'
        s += self.__print_typed_cpfs(d.derived_cpfs)
        s += self.__tabs(1) + '};'
        return s

    def __print_typed_cpfs(self, cpfs):
        temp = ''
        for cpf in cpfs:
            name = cpf.pvar[1][0]
            args = '' if cpf.pvar[1][1] is None else '(' + ', '.join(cpf.pvar[1][1]) + ')'
            temp += self.__tabs(2) + name + args +'=\n'
            temp += self.__tabs(4) + self.__scan_expr_tree(cpf.expr)
            temp += '\n'
        return temp

    def __scan_expr_tree(self, expr):#, temp):
        temp = ''
        if expr.etype[0] == 'constant':
            return str(expr.args)
        if expr.etype[0] == 'pvar':
            return self.__format_pvar_args(expr.args)
        if expr.etype[0] == 'relational':
            arg_list = []
            for child in expr.args:
                arg_list.append(self.__scan_expr_tree(child))
            op = expr.etype[1]
            return temp + '(' + arg_list[0] + op + arg_list[1] + ')'
        if expr.etype[0] == 'boolean':
            arg_list = []
            for child in expr.args:
                arg_list.append(self.__scan_expr_tree(child))
            op = expr.etype[1]
            if len(arg_list) == 1:
                return temp + op + arg_list[0]
            else:
                return temp + arg_list[0] + op + arg_list[1]
        if expr.etype[0] == 'control':
            # if
            if_txt = self.__scan_expr_tree(expr.args[0])
            # then
            then_txt = self.__scan_expr_tree(expr.args[1])
            # else
            else_txt = self.__scan_expr_tree(expr.args[2])
            return temp + 'if (' + if_txt + ')\n\tthen ' + then_txt + '\n\telse ' + else_txt
        if expr.etype[0] == 'arithmetic':
            arg_list = []
            for child in expr.args:
                arg_list.append(self.__scan_expr_tree(child))
            op = expr.etype[1]
            if op == '+' or op == '-':
                return temp + op.join(arg_list)
            else:
                return temp + arg_list[0] + op + '(' + arg_list[1] + ')'
        if expr.etype[0] == 'randomvar':
            arg_list = []
            for child in expr.args:
                arg_list.append(self.__scan_expr_tree(child))
            return temp + expr.etype[1] + '(' + ', '.join(arg_list) + ')'
        if expr.etype[0] == 'func':
            temp = expr.etype[1] + '[' + temp
            arg_list = []
            for child in expr.args:
                arg_list.append(self.__scan_expr_tree(child))
            temp = temp + ', '.join(arg_list) + ']'
            return temp
        if expr.etype[0] == 'aggregation':
            inner_expr = self.__scan_expr_tree(expr.args[-1])
            vars = []
            for arg in expr.args[:-1]:
                vars.append(' : '.join(arg[1]))
            temp += '(' + expr.etype[1] + '_{' + ' , '.join(vars) + '} [' + inner_expr + '])'
            return temp
        else:
            for child in expr.args:
                temp += self.__scan_expr_tree(child)
            return temp

    def __format_pvar_args(self, pvar):
        name = pvar[0]
        args = '' if pvar[1] is None else '('+', '.join(pvar[1])+')'
        return name+args

    def __print_constraints(self, d):
        s = ''
        if d.preconds:
            s += self.__print_typed_constraints('action-preconditions', d.preconds)
            s += '\n\n'
        if d.invariants:
            s += self.__print_typed_constraints('state-invariants', d.invariants)
            s += '\n\n'
        if d.constraints:
            s += self.__print_typed_constraints('state-action-constraints', d.constraints)
            s += '\n\n'
        return s

    def __print_typed_constraints(self, constraint_type, constraints):
        const_str = self.__tabs(1) + constraint_type + ' {\n'
        for c in constraints:
            const_str += self.__tabs(2) + self.__scan_expr_tree(c) + ';\n'
        const_str += self.__tabs(1) + '};'
        return const_str

    def __print_reward(self, reward):
        s = self.__tabs(1) + 'reward = '
        s += self.__scan_expr_tree(reward)
        return s