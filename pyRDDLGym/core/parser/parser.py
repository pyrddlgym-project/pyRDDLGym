# This file is based on thiago pbueno's pyrddl.
# https://github.com/thiagopbueno/pyrddl
# it was adapted and extended for pyRDDLGym

import logging

import os
import tempfile

from ply import lex, yacc

from pyRDDLGym.Core.Parser.rddl import RDDL
from pyRDDLGym.Core.Parser.domain import Domain
from pyRDDLGym.Core.Parser.nonfluents import NonFluents
from pyRDDLGym.Core.Parser.instance import Instance
from pyRDDLGym.Core.Parser.pvariable import PVariable
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Parser.cpf import CPF
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLParseError

alpha = r'[A-Za-z]'
digit = r'[0-9]'
idenfifier = r'(' + alpha + r')((' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r'))?(\')?'
integer = digit + r'+'
double = digit + r'*\.' + digit + r'+'
variable = r'\?(' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r')'
enum_value = r'\@(' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r')'


class RDDLlex(object):

    def __init__(self):
        self.reserved = {
            'domain': 'DOMAIN',
            'instance': 'INSTANCE',
            'horizon': 'HORIZON',
            'discount': 'DISCOUNT',
            'objects': 'OBJECTS',
            'init-state': 'INIT_STATE',
            'requirements': 'REQUIREMENTS',
            'state-action-constraints': 'STATE_ACTION_CONSTRAINTS',
            'action-preconditions': 'ACTION_PRECONDITIONS',
            'termination': 'TERMINATION',
            'state-invariants': 'STATE_INVARIANTS',
            'types': 'TYPES',
            'object': 'OBJECT',
            'bool': 'BOOL',
            'int': 'INT',
            'real': 'REAL',
            'neg-inf': 'NEG_INF',
            'pos-inf': 'POS_INF',
            'pvariables': 'PVARIABLES',
            'non-fluent': 'NON_FLUENT',
            'non-fluents': 'NON_FLUENTS',
            'state-fluent': 'STATE',
            'interm-fluent': 'INTERMEDIATE',
            'derived-fluent': 'DERIVED_FLUENT',
            'observ-fluent': 'OBSERVATION',
            'action-fluent': 'ACTION',
            'level': 'LEVEL',
            'default': 'DEFAULT',
            'max-nondef-actions': 'MAX_NONDEF_ACTIONS',
            'terminate-when': 'TERMINATE_WHEN',
            'terminal': 'TERMINAL',
            'cpfs': 'CPFS',
            'cdfs': 'CDFS',
            'reward': 'REWARD',
            'forall': 'FORALL',
            'exists': 'EXISTS',
            'argmax': 'ARGMAX',  # CHANGED BY MIKE ON JAN 15
            'argmin': 'ARGMIN',  # CHANGED BY MIKE ON JAN 15
            # 'sum': 'SUM',
            'true': 'TRUE',
            'false': 'FALSE',
            'if': 'IF',
            'then': 'THEN',
            'else': 'ELSE',
            'switch': 'SWITCH',
            'case': 'CASE',
            'otherwise': 'OTHERWISE',
            'KronDelta': 'KRON_DELTA',
            'DiracDelta': 'DIRAC_DELTA',
            'Uniform': 'UNIFORM',
            'Bernoulli': 'BERNOULLI',
            'Discrete': 'DISCRETE',
            'UnnormDiscrete': 'UNNORMDISCRETE',
            'Normal': 'NORMAL',
            'Poisson': 'POISSON',
            'Exponential': 'EXPONENTIAL',
            'Weibull': 'WEIBULL',
            'Gamma': 'GAMMA',
            'Binomial': 'BINOMIAL',
            'NegativeBinomial': 'NEGATIVEBINOMIAL',
            'Beta': 'BETA',
            'Geometric': 'GEOMETRIC',
            'Pareto': 'PARETO',
            'Student': 'STUDENT',
            'Gumbel': 'GUMBEL',
            'Laplace': 'LAPLACE',
            'Cauchy': 'CAUCHY',
            'Gompertz': 'GOMPERTZ',
            'ChiSquare': 'CHISQUARE',
            'Kumaraswamy': 'KUMARASWAMY',
            'MultivariateNormal': 'MULTIVARIATENORMAL',
            'MultivariateStudent': 'MULTIVARIATESTUDENT',
            'Dirichlet': 'DIRICHLET',
            'Multinomial': 'MULTINOMIAL',
            'det': 'DET',
            'inverse': 'INVERSE',
            'pinverse': 'PSEUDOINVERSE',
            'cholesky': 'CHOLESKY',
            'row': 'ROW',
            'col': 'COLUMN'
        }

        self.tokens = [
            'IDENT',
            'VAR',
            'ENUM_VAL',
            'INTEGER',
            'DOUBLE',
            'AND',
            'OR',
            'NOT',
            'PLUS',
            'TIMES',
            'LPAREN',
            'RPAREN',
            'LCURLY',
            'RCURLY',
            'DOT',
            'COMMA',
            'UNDERSCORE',
            'LBRACK',
            'RBRACK',
            'IMPLY',
            'EQUIV',
            'NEQ',
            'LESSEQ',
            'LESS',
            'GREATEREQ',
            'GREATER',
            'ASSIGN_EQUAL',
            'COMP_EQUAL',
            'DIV',
            'MINUS',
            'COLON',
            'SEMI',
            'DOLLAR_SIGN',
            'QUESTION',
            'AMPERSAND'
        ]
        self.tokens += list(self.reserved.values())

    t_ignore = ' \t'

    t_AND = r'\^'
    t_OR = r'\|'
    t_NOT = r'~'
    t_PLUS = r'\+'
    t_TIMES = r'\*'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LCURLY = r'\{'
    t_RCURLY = r'\}'
    t_DOT = r'\.'
    t_COMMA = r'\,'
    t_UNDERSCORE = r'\_'
    t_LBRACK = r'\['
    t_RBRACK = r'\]'
    t_IMPLY = r'=>'
    t_EQUIV = r'<=>'
    t_NEQ = r'~='
    t_LESSEQ = r'<='
    t_LESS = r'<'
    t_GREATEREQ = r'>='
    t_GREATER = r'>'
    t_ASSIGN_EQUAL = r'='
    t_COMP_EQUAL = r'=='
    t_DIV = r'/'
    t_MINUS = r'-'
    t_COLON = r':'
    t_SEMI = r';'
    t_DOLLAR_SIGN = r'\$'
    t_QUESTION = r'\?'
    t_AMPERSAND = r'\&'

    def t_newline(self, t):
        r'\n+'
        self._lexer.lineno += len(t.value)

    def t_COMMENT(self, t):
        r'//[^\r\n]*'
        pass

    @lex.TOKEN(idenfifier)
    def t_IDENT(self, t):
        t.type = self.reserved.get(t.value, 'IDENT')
        return t

    @lex.TOKEN(variable)
    def t_VAR(self, t):
        return t

    @lex.TOKEN(enum_value)
    def t_ENUM_VAL(self, t):
        return t

    @lex.TOKEN(double)
    def t_DOUBLE(self, t):
        t.value = float(t.value)
        return t

    @lex.TOKEN(integer)
    def t_INTEGER(self, t):
        t.value = int(t.value)
        return t

    def t_error(self, t):
        print("Illegal character: {} at line {}".format(t.value[0], self._lexer.lineno))
        t.lexer.skip(1)

    def build(self, **kwargs):
        self._lexer = lex.lex(object=self, **kwargs)

    def input(self, data):
        if self._lexer is None:
            self.build()
        self._lexer.input(data)

    def token(self):
        return self._lexer.token()

    def __call__(self):
        while True:
            tok = self.token()
            if not tok:
                break
            yield tok


class RDDLParser(object):

    def __init__(self, lexer=None, verbose=False):
        if lexer is None:
            self.lexer = RDDLlex()
            self.lexer.build()

        self._verbose = verbose

        self.tokens = self.lexer.tokens

        self.precedence = (
            ('left', 'IF'),
            ('left', 'ASSIGN_EQUAL'),
            ('left', 'EXISTS'),
            ('left', 'FORALL'),
            ('left', 'AGG_OPER', 'ARGMAX', 'ARGMIN'),  # CHANGED BY MIKE ON JAN 15
            ('left', 'EQUIV'),
            ('left', 'IMPLY'),
            ('left', 'OR'),
            ('left', 'AND', 'AMPERSAND'),
            ('left', 'NOT'),
            ('left', 'COMP_EQUAL', 'NEQ', 'LESS', 'LESSEQ', 'GREATER', 'GREATEREQ'),
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES', 'DIV'),
            ('right', 'UMINUS')
        )
        self.parsing_logfile = None
        self.debugging = False

    def p_rddl(self, p):
        '''rddl : rddl_block'''
        p[0] = RDDL(p[1])

    def p_rddl_block(self, p):
        '''rddl_block : rddl_block domain_block
                      | rddl_block instance_block
                      | rddl_block nonfluent_block
                      | empty'''
        if p[1] is None:
            p[0] = dict()
        else:
            name, block = p[2]            
            # <-- START OF CHANGES TO SUPPORT 2018 INSTANCE BLOCKS
            if name == 'instance':
                block, inst = block
                if inst is not None:
                    p[1]['non_fluents'] = inst
            # END OF CHANGES TO SUPPORT 2018 INSTANCE BLOCKS -->
            p[1][name] = block
            p[0] = p[1]

    def p_domain_block(self, p):
        '''domain_block : DOMAIN IDENT LCURLY req_section domain_list RCURLY'''
        d = Domain(p[2], p[4], p[5])
        p[0] = ('domain', d)

    def p_req_section(self, p):
        '''req_section : REQUIREMENTS ASSIGN_EQUAL LCURLY string_list RCURLY SEMI
                       | REQUIREMENTS LCURLY string_list RCURLY SEMI
                       | empty'''
        if len(p) == 7:
            p[0] = p[4]
        elif len(p) == 6:
            p[0] = p[3]
        self._print_verbose('requirements')

    def p_domain_list(self, p):
        '''domain_list : domain_list type_section
                       | domain_list pvar_section
                       | domain_list cpf_section
                       | domain_list reward_section
                       | domain_list termination_section
                       | domain_list action_precond_section
                       | domain_list state_action_constraint_section
                       | domain_list state_invariant_section
                       | empty'''
        if p[1] is None:
            p[0] = dict()
        else:
            name, section = p[2]
            p[1][name] = section
            p[0] = p[1]

    def p_type_section(self, p):
        '''type_section : TYPES LCURLY type_list RCURLY SEMI'''
        p[0] = ('types', p[3])
        self._print_verbose('types')

    def p_type_list(self, p):
        '''type_list : type_list type_def
                     | empty'''
        if p[1] is None:
            p[0] = []
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_type_def(self, p):
        '''type_def : IDENT COLON OBJECT SEMI
                    | IDENT COLON LCURLY enum_list RCURLY SEMI'''
        if len(p) == 5:
            p[0] = (p[1], p[3])
        elif len(p) == 7:
            p[0] = (p[1], p[4])

    def p_enum_list(self, p):
        '''enum_list : enum_list COMMA ENUM_VAL
                     | ENUM_VAL
                     | empty'''
        if p[1] is None:
            p[0] = []
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_pvar_section(self, p):
        '''pvar_section : PVARIABLES LCURLY pvar_list RCURLY SEMI'''
        p[0] = ('pvariables', p[3])
        self._print_verbose('pvariables')

    def p_pvar_list(self, p):
        '''pvar_list : pvar_list pvar_def
                     | empty'''
        if p[1] is None:
            p[0] = []
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_pvar_def(self, p):
        '''pvar_def : nonfluent_def
                    | statefluent_def
                    | actionfluent_def
                    | intermfluent_def
                    | derivedfluent_def
                    | observfluent_def'''
        p[0] = p[1]

    def p_nonfluent_def(self, p):
        #    0                1    2          3        4         5    6          7   8           9           10         11
        '''nonfluent_def : IDENT param_list LCURLY NON_FLUENT COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI'''
        # print("here")
        if len(p) == 13:
            p[0] = PVariable(name=p[1], fluent_type='non-fluent', range_type=p[6], param_types=p[2], default=p[10])
        else:
            p[0] = PVariable(name=p[1], fluent_type='non-fluent', range_type=p[6], default=p[10])

    def p_statefluent_def(self, p):
        '''statefluent_def : IDENT param_list LCURLY STATE COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI'''
        if len(p) == 13:
            p[0] = PVariable(name=p[1], fluent_type='state-fluent', range_type=p[6], param_types=p[2], default=p[10])
        else:
            p[0] = PVariable(name=p[1], fluent_type='state-fluent', range_type=p[6], default=p[10])

    def p_actionfluent_def(self, p):
        '''actionfluent_def : IDENT param_list LCURLY ACTION COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI'''
        if len(p) == 13:
            p[0] = PVariable(name=p[1], fluent_type='action-fluent', range_type=p[6], param_types=p[2], default=p[10])
        else:
            p[0] = PVariable(name=p[1], fluent_type='action-fluent', range_type=p[6], default=p[10])

    def p_intermfluent_def(self, p):
        '''intermfluent_def : IDENT param_list LCURLY INTERMEDIATE COMMA type_spec COMMA LEVEL ASSIGN_EQUAL range_const RCURLY SEMI
                            | IDENT param_list LCURLY INTERMEDIATE COMMA type_spec RCURLY SEMI'''
        if len(p) == 13:
            p[0] = PVariable(name=p[1], fluent_type='interm-fluent', range_type=p[6], param_types=p[2], level=p[10])
        else:
            p[0] = PVariable(name=p[1], fluent_type='interm-fluent', range_type=p[6], param_types=p[2], level=1)

    def p_derivedfluent_def(self, p):
        '''derivedfluent_def : IDENT param_list LCURLY DERIVED_FLUENT COMMA type_spec COMMA LEVEL ASSIGN_EQUAL range_const RCURLY SEMI
                             | IDENT param_list LCURLY DERIVED_FLUENT COMMA type_spec RCURLY SEMI'''
        if len(p) == 13:
            p[0] = PVariable(name=p[1], fluent_type='derived-fluent', range_type=p[6], param_types=p[2], level=p[10])
        else:
            p[0] = PVariable(name=p[1], fluent_type='derived-fluent', range_type=p[6], param_types=p[2], level=1)

    def p_observfluent_def(self, p):
        '''observfluent_def : IDENT param_list LCURLY OBSERVATION COMMA type_spec RCURLY SEMI'''
        p[0] = PVariable(name=p[1], fluent_type='observ-fluent', range_type=p[6], param_types=p[2])

    def p_cpf_section(self, p):
        '''cpf_section : cpf_header LCURLY cpf_list RCURLY SEMI'''
        '''cpf_section : cpf_header LCURLY cpf_list RCURLY SEMI'''
        p[0] = ('cpfs', (p[1], p[3]))
        self._print_verbose('cpfs')

    def p_cpf_header(self, p):
        '''cpf_header : CPFS
                      | CDFS'''
        p[0] = p[1]

    def p_cpf_list(self, p):
        '''cpf_list : cpf_list cpf_def
                    | empty'''
        if p[1] is None:
            p[0] = []
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_cpf_def(self, p):
        '''cpf_def : pvar_expr ASSIGN_EQUAL expr SEMI
                   | pvar_expr ASSIGN_EQUAL randomvector_expr SEMI'''
        p[0] = CPF(pvar=p[1], expr=p[3])

    def p_reward_section(self, p):
        '''reward_section : REWARD ASSIGN_EQUAL expr SEMI'''
        p[0] = ('reward', p[3])
        self._print_verbose('reward')

    def p_termination_section(self, p):
        '''termination_section  : TERMINATION LCURLY termination_list RCURLY SEMI
                                |  TERMINATION LCURLY RCURLY SEMI'''
        if len(p) == 6:
            p[0] = ('terminals', p[3])
            
        elif len(p) == 5:
            p[0] = ('terminals', [])
        self._print_verbose('termination')

    def p_termination_list(self, p):
        '''termination_list : termination_list termination_cond_def
                            | termination_cond_def'''
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_termination_cond_def(self, p):
        '''termination_cond_def : expr SEMI'''
        p[0] = p[1]

    def p_action_precond_section(self, p):
        '''action_precond_section : ACTION_PRECONDITIONS LCURLY action_precond_list RCURLY SEMI
                                  | ACTION_PRECONDITIONS LCURLY RCURLY SEMI'''
        if len(p) == 6:
            p[0] = ('preconds', p[3])
        elif len(p) == 5:
            p[0] = ('preconds', [])
        self._print_verbose('action-preconditions')

    def p_action_precond_list(self, p):
        '''action_precond_list : action_precond_list action_precond_def
                               | action_precond_def'''
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_action_precond_def(self, p):
        '''action_precond_def : expr SEMI'''
        p[0] = p[1]

    def p_state_action_constraint_section(self, p):
        '''state_action_constraint_section : STATE_ACTION_CONSTRAINTS LCURLY state_cons_list RCURLY SEMI
                                           | STATE_ACTION_CONSTRAINTS LCURLY RCURLY SEMI'''
        if len(p) == 6:
            p[0] = ('constraints', p[3])
        elif len(p) == 5:
            p[0] = ('constraints', [])
        self._print_verbose('state-action-constraints')

    def p_state_cons_list(self, p):
        '''state_cons_list : state_cons_list state_cons_def
                           | state_cons_def'''
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_state_cons_def(self, p):
        '''state_cons_def : expr SEMI'''
        p[0] = p[1]

    def p_state_invariant_section(self, p):
        '''state_invariant_section : STATE_INVARIANTS LCURLY state_invariant_list RCURLY SEMI
                                   | STATE_INVARIANTS LCURLY RCURLY SEMI'''
        if len(p) == 6:
            p[0] = ('invariants', p[3])
        elif len(p) == 5:
            p[0] = ('invariants', [])
        self._print_verbose('invariants')

    def p_state_invariant_list(self, p):
        '''state_invariant_list : state_invariant_list state_invariant_def
                                | state_invariant_def'''
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_state_invariant_def(self, p):
        '''state_invariant_def : expr SEMI'''
        p[0] = p[1]

    def p_term_list(self, p):
        '''term_list : term_list COMMA term
                     | term
                     | empty'''
        if p[1] is None:
            p[0] = []
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_term(self, p):
        # CHANGED BY MIKE ON JAN 15
        '''term : VAR
                | ENUM_VAL
                | pvar_expr
                | argmaxmin_expr'''        
        if isinstance(p[1], tuple):
            p[0] = Expression(p[1])
        else:
            p[0] = p[1]

    def p_expr(self, p):
        # CHANGED BY MIKE ON JAN 15
        '''expr : pvar_expr
                | group_expr
                | function_expr
                | relational_expr
                | boolean_expr
                | quantifier_expr
                | numerical_expr
                | aggregation_expr
                | argmaxmin_expr
                | matrix_expr
                | control_expr
                | randomvar_expr
                | randomvar_from_pvar_expr'''
        p[0] = Expression(p[1])
    
    def p_pvar_expr(self, p):
        # CHANGED BY MIKE ON JAN 15
        '''pvar_expr : IDENT LPAREN term_list RPAREN
                     | IDENT
                     | ENUM_VAL
                     | VAR'''
        if len(p) == 2:
            p[0] = ('pvar_expr', (p[1], None))
        elif len(p) == 5:
            p[0] = ('pvar_expr', (p[1], p[3]))

    def p_group_expr(self, p):
        '''group_expr : LBRACK expr RBRACK
                      | LPAREN expr RPAREN'''
        p[0] = p[2]

    def p_function_expr(self, p):
        '''function_expr : IDENT LBRACK expr_list RBRACK'''
        p[0] = ('func', (p[1], p[3]))

    def p_relational_expr(self, p):
        '''relational_expr : expr COMP_EQUAL expr
                           | expr NEQ expr
                           | expr GREATER expr
                           | expr GREATEREQ expr
                           | expr LESS expr
                           | expr LESSEQ expr'''
        p[0] = (p[2], (p[1], p[3]))

    def p_boolean_expr(self, p):
        '''boolean_expr : expr AND expr
                        | expr AMPERSAND expr
                        | expr OR expr
                        | expr IMPLY expr
                        | expr EQUIV expr
                        | NOT expr %prec UMINUS
                        | bool_type'''
        if len(p) == 4:
            p[0] = (p[2], (p[1], p[3]))
        elif len(p) == 3:
            p[0] = (p[1], (p[2],))
        elif len(p) == 2:
            p[0] = ('boolean', p[1])

    def p_quantifier_expr(self, p):
        '''quantifier_expr : FORALL UNDERSCORE LCURLY typed_var_list RCURLY expr %prec FORALL
                           | EXISTS UNDERSCORE LCURLY typed_var_list RCURLY expr %prec EXISTS'''
        p[0] = (p[1], (*p[4], p[6]))

    def p_numerical_expr(self, p):
        '''numerical_expr : expr PLUS expr
                          | expr MINUS expr
                          | expr TIMES expr
                          | expr DIV expr
                          | MINUS expr %prec UMINUS
                          | PLUS expr %prec UMINUS
                          | INTEGER
                          | DOUBLE'''
        if len(p) == 4:
            p[0] = (p[2], (p[1], p[3]))
        elif len(p) == 3:
            p[0] = (p[1], (p[2],))
        elif len(p) == 2:
            p[0] = ('number', p[1])

    def p_aggregation_expr(self, p):
        '''aggregation_expr : IDENT UNDERSCORE LCURLY typed_var_list RCURLY expr %prec AGG_OPER'''
        p[0] = (p[1], (*p[4], p[6]))
    
    # CHANGED BY MIKE ON JAN 15
    def p_argmaxmin_expr(self, p):
        '''argmaxmin_expr : ARGMAX UNDERSCORE LCURLY typed_var_list RCURLY expr %prec ARGMAX
                          | ARGMIN UNDERSCORE LCURLY typed_var_list RCURLY expr %prec ARGMIN'''
        p[0] = (p[1], (*p[4], p[6]))

    def p_control_expr(self, p):
        '''control_expr : IF LPAREN expr RPAREN THEN expr ELSE expr %prec IF
                        | SWITCH LPAREN expr RPAREN LCURLY case_list RCURLY'''
                        # | SWITCH LPAREN term RPAREN LCURLY case_list RCURLY
        # if-then-else
        if len(p) == 9:
            p[0] = (p[1], (p[3], p[6], p[8]))
        # switch
        elif len(p) == 8:
            p[0] = (p[1], (p[3], *p[6]))

    def p_randomvar_expr(self, p):
        '''randomvar_expr : BERNOULLI LPAREN expr RPAREN
                          | DIRAC_DELTA LPAREN expr RPAREN
                          | KRON_DELTA LPAREN expr RPAREN
                          | UNIFORM LPAREN expr COMMA expr RPAREN
                          | NORMAL LPAREN expr COMMA expr RPAREN
                          | EXPONENTIAL LPAREN expr RPAREN
                          | DISCRETE LPAREN IDENT COMMA lconst_case_list RPAREN
                          | UNNORMDISCRETE LPAREN IDENT COMMA lconst_case_list RPAREN
                          | DIRICHLET LPAREN IDENT COMMA expr RPAREN
                          | POISSON LPAREN expr RPAREN
                          | WEIBULL LPAREN expr COMMA expr RPAREN
                          | GAMMA   LPAREN expr COMMA expr RPAREN
                          | BINOMIAL   LPAREN expr COMMA expr RPAREN
                          | NEGATIVEBINOMIAL   LPAREN expr COMMA expr RPAREN
                          | BETA   LPAREN expr COMMA expr RPAREN
                          | GEOMETRIC LPAREN expr RPAREN
                          | PARETO   LPAREN expr COMMA expr RPAREN
                          | STUDENT LPAREN expr RPAREN
                          | GUMBEL   LPAREN expr COMMA expr RPAREN
                          | LAPLACE LPAREN expr COMMA expr RPAREN
                          | CAUCHY LPAREN expr COMMA expr RPAREN
                          | GOMPERTZ LPAREN expr COMMA expr RPAREN
                          | CHISQUARE LPAREN expr RPAREN
                          | KUMARASWAMY LPAREN expr COMMA expr RPAREN'''
        if len(p) == 7:
            if isinstance(p[5], list):
                p[0] = ('randomvar', (p[1], (('enum_type', p[3]), *p[5])))
            else:
                p[0] = ('randomvar', (p[1], (p[3], p[5])))
        elif len(p) == 5:
            p[0] = ('randomvar', (p[1], (p[3],)))
    
    # CHANGED BY MIKE ON JAN 16
    def p_randomvar_from_pvar_expr(self, p):
        '''randomvar_from_pvar_expr : DISCRETE UNDERSCORE LCURLY typed_var_list RCURLY LPAREN expr RPAREN
                                    | UNNORMDISCRETE UNDERSCORE LCURLY typed_var_list RCURLY LPAREN expr RPAREN'''
        p[0] = ('randomvar', (p[1] + '(p)', (*p[4], (p[7],))))
    
    # CHANGED BY MIKE ON JAN 17
    def p_randomvector_expr(self, p):
        '''randomvector_expr : MULTIVARIATENORMAL LBRACK randomvector_term_list RBRACK LPAREN randomvector_pvar_expr COMMA randomvector_pvar_expr RPAREN
                             | MULTIVARIATESTUDENT LBRACK randomvector_term_list RBRACK LPAREN randomvector_pvar_expr COMMA randomvector_pvar_expr COMMA randomvector_pvar_expr RPAREN
                             | DIRICHLET LBRACK randomvector_term_list RBRACK LPAREN randomvector_pvar_expr RPAREN
                             | MULTINOMIAL LBRACK randomvector_term_list RBRACK LPAREN randomvector_pvar_expr COMMA randomvector_pvar_expr RPAREN'''
        if len(p) == 12:
            p[0] = Expression(('randomvector', (p[1], (p[3], (p[6], p[8], p[10])))))
        elif len(p) == 10:
            p[0] = Expression(('randomvector', (p[1], (p[3], (p[6], p[8])))))
        else:
            p[0] = Expression(('randomvector', (p[1], (p[3], (p[6],)))))
        
    # CHANGED BY MIKE ON JAN 17
    def p_randomvector_pvar_expr(self, p):
        '''randomvector_pvar_expr : IDENT LPAREN randomvector_term_list RPAREN
                                  | IDENT'''
        if len(p) == 2:
            p[0] = Expression(('pvar_expr', (p[1], None)))
        elif len(p) == 5:
            p[0] = Expression(('pvar_expr', (p[1], p[3])))
        
    # CHANGED BY MIKE ON JAN 17
    def p_randomvector_term_list(self, p):
        '''randomvector_term_list : randomvector_term_list COMMA randomvector_term
                                  | randomvector_term
                                  | empty'''
        if p[1] is None:
            p[0] = []
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    # CHANGED BY MIKE ON JAN 17
    def p_randomvector_term(self, p):
        '''randomvector_term : VAR
                             | ENUM_VAL
                             | UNDERSCORE'''        
        p[0] = p[1]
    
    # CHANGED BY MIKE ON JAN 22
    def p_matrix_expr(self, p):
        '''matrix_expr : DET UNDERSCORE LCURLY typed_var COMMA typed_var RCURLY expr %prec AGG_OPER
                       | INVERSE LBRACK ROW ASSIGN_EQUAL VAR COMMA COLUMN ASSIGN_EQUAL VAR RBRACK LBRACK expr RBRACK
                       | PSEUDOINVERSE LBRACK ROW ASSIGN_EQUAL VAR COMMA COLUMN ASSIGN_EQUAL VAR RBRACK LBRACK expr RBRACK
                       | CHOLESKY LBRACK ROW ASSIGN_EQUAL VAR COMMA COLUMN ASSIGN_EQUAL VAR RBRACK LBRACK expr RBRACK'''
        if len(p) == 9:
            p[0] = (p[1], (p[4], p[6], p[8]))
        elif len(p) == 14:
            p[0] = (p[1], ([p[5], p[9]], p[12]))
        
    def p_typed_var_list(self, p):
        '''typed_var_list : typed_var_list COMMA typed_var
                          | typed_var'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_typed_var(self, p):
        '''typed_var : VAR COLON IDENT'''
        p[0] = ('typed_var', (p[1], p[3]))

    def p_expr_list(self, p):
        '''expr_list : expr_list COMMA expr
                     | expr'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_case_list(self, p):
        '''case_list : case_list COMMA case_def
                     | case_def'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_case_def(self, p):
        '''case_def : CASE term COLON expr
                    | DEFAULT COLON expr'''
        if len(p) == 5:
            p[0] = ('case', (p[2], p[4]))
        elif len(p) == 4:
            p[0] = ('default', p[3])

    def p_lconst_case_list(self, p):
        '''lconst_case_list : lconst COLON expr
                            | lconst COLON OTHERWISE
                            | lconst_case_list COMMA lconst COLON expr'''
        if len(p) == 4:
            p[0] = [('lconst', (p[1], p[3]))]
        elif len(p) == 6:
            p[1].append(('lconst', (p[3], p[5])))
            p[0] = p[1]

    def p_lconst(self, p):
        '''lconst : IDENT
                  | ENUM_VAL'''
        p[0] = p[1]

    # new definitions
    def p_param_list(self, p):
        '''param_list : COLON
                      | LPAREN param_list2 RPAREN COLON'''
        if len(p) == 2:
            p[0] = None
        else:
            p[0] = p[2]

    def p_param_list2(self, p):
        '''param_list2 : type_spec
                       | param_list2 COMMA type_spec'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        else:
            p[0] = [p[1]]

    # def p_param_list(self, p):
    #     '''param_list : LPAREN string_list RPAREN
    #                   | empty'''
    #     if (len(p) == 3):
    #         p[0] = p[1]
    #     else:
    #         p[0] = []

    def p_type_spec(self, p):
        '''type_spec : IDENT
                     | INT
                     | REAL
                     | BOOL'''
        p[0] = p[1]

    def p_range_const(self, p):
        '''range_const : bool_type
                       | double_type
                       | int_type
                       | ENUM_VAL
                       | IDENT'''
        p[0] = p[1]

    def p_bool_type(self, p):
        '''bool_type : TRUE
                     | FALSE'''
        p[0] = True if p[1] == 'true' else False

    def p_double_type(self, p):
        '''double_type : DOUBLE
                       | MINUS DOUBLE
                       | POS_INF
                       | NEG_INF'''
        p[0] = p[1] if len(p) == 2 else -p[2]

    def p_int_type(self, p):
        '''int_type : INTEGER
                    | MINUS INTEGER'''
        p[0] = p[1] if len(p) == 2 else -p[2]

    def p_pos_int_type_or_pos_inf(self, p):
        '''pos_int_type_or_pos_inf : INTEGER
                                   | POS_INF'''
        p[0] = p[1]
    
    def fake_nonfluents_block(self, inst):
        nonfluents = None
        if 'init_non_fluent' in inst:
            sections = {'domain': inst['domain'],
                        'objects': inst['objects'],
                        'init_non_fluent': inst['init_non_fluent']}
            inst_name = 'nf__' + inst['domain']
            nonfluents = NonFluents(inst_name, sections)
            if 'non_fluents' in inst:
                print('warning: parser will override instance non-fluents block ' + 
                      inst['non_fluents'] + ' with non-fluents {...}; block')
            inst['non_fluents'] = inst_name
            del inst['objects']
            del inst['init_non_fluent']
        return nonfluents
    
    def p_instance_block(self, p):
        '''instance_block : INSTANCE IDENT LCURLY instance_list RCURLY'''
        
        # <-- START OF CHANGES TO SUPPORT 2018 INSTANCE BLOCKS
        # inst = Instance(p[2], p[4])
        # p[0] = ('instance', inst)
        inst = p[4]
        nonfluents = self.fake_nonfluents_block(inst)
        inst = Instance(p[2], inst)
        p[0] = ('instance', (inst, nonfluents))
        # END OF CHANGES TO SUPPORT 2018 INSTANCE BLOCKS -->
        
    def p_instance_list(self, p):
        # <-- START OF CHANGES TO SUPPORT 2018 INSTANCE BLOCKS
        '''instance_list : instance_list domain_section
                         | instance_list nonfluents_section
                         | instance_list init_non_fluent_section
                         | instance_list objects_section
                         | instance_list init_state_section
                         | instance_list max_nondef_actions_section
                         | instance_list horizon_spec_section
                         | instance_list discount_section
                         | empty'''
        # END OF CHANGES TO SUPPORT 2018 INSTANCE BLOCKS -->
        if p[1] is None:
            p[0] = dict()
        else:
            name, section = p[2]
            p[1][name] = section
            p[0] = p[1]

    def p_domain_section(self, p):
        '''domain_section : DOMAIN ASSIGN_EQUAL IDENT SEMI'''
        p[0] = ('domain', p[3])

    def p_nonfluents_section(self, p):
        '''nonfluents_section : NON_FLUENTS ASSIGN_EQUAL IDENT SEMI'''
        p[0] = ('non_fluents', p[3])
        self._print_verbose('non-fluents')

    def p_objects_section(self, p):
        '''objects_section : OBJECTS LCURLY objects_list RCURLY SEMI'''
        p[0] = ('objects', p[3])
        self._print_verbose('objects')

    def p_init_state_section(self, p):
        '''init_state_section : INIT_STATE LCURLY pvar_inst_list RCURLY SEMI'''
        p[0] = ('init_state', p[3])
        self._print_verbose('init-state')

    def p_max_nondef_actions_section(self, p):
        '''max_nondef_actions_section : MAX_NONDEF_ACTIONS ASSIGN_EQUAL pos_int_type_or_pos_inf SEMI'''
        p[0] = ('max_nondef_actions', p[3])
        self._print_verbose('max-non-def-actions')

    def p_horizon_spec_section(self, p):
        '''horizon_spec_section : HORIZON ASSIGN_EQUAL pos_int_type_or_pos_inf SEMI
                                | HORIZON ASSIGN_EQUAL TERMINATE_WHEN LPAREN expr RPAREN'''
        if len(p) == 5:
            p[0] = ('horizon', p[3])
        elif len(p) == 7:
            p[0] = ('horizon', p[5])
        self._print_verbose('horizon')

    def p_discount_section(self, p):
        '''discount_section : DISCOUNT ASSIGN_EQUAL DOUBLE SEMI'''
        p[0] = ('discount', p[3])
        self._print_verbose('discount')

    def p_nonfluent_block(self, p):
        '''nonfluent_block : NON_FLUENTS IDENT LCURLY nonfluent_list RCURLY'''
        nf = NonFluents(p[2], p[4])
        p[0] = ('non_fluents', nf)

    def p_nonfluent_list(self, p):
        '''nonfluent_list : nonfluent_list domain_section
                          | nonfluent_list objects_section
                          | nonfluent_list init_non_fluent_section
                          | empty'''
        if p[1] is None:
            p[0] = dict()
        else:
            name, section = p[2]
            p[1][name] = section
            p[0] = p[1]

    def p_init_non_fluent_section(self, p):
        '''init_non_fluent_section : NON_FLUENTS LCURLY pvar_inst_list RCURLY SEMI'''
        p[0] = ('init_non_fluent', p[3])
        self._print_verbose('init-non-fluent')

    def p_objects_list(self, p):
        '''objects_list : objects_list objects_def
                        | objects_def
                        | empty'''
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_objects_def(self, p):
        '''objects_def : IDENT COLON LCURLY object_const_list RCURLY SEMI'''
        p[0] = (p[1], p[4])

    def p_object_const_list(self, p):
        '''object_const_list : object_const_list COMMA IDENT
                             | IDENT'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_pvar_inst_list(self, p):
        '''pvar_inst_list : pvar_inst_list pvar_inst_def
                          | pvar_inst_def'''
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_pvar_inst_def(self, p):
        '''pvar_inst_def : IDENT LPAREN lconst_list RPAREN SEMI
                         | IDENT SEMI
                         | NOT IDENT LPAREN lconst_list RPAREN SEMI
                         | NOT IDENT SEMI
                         | IDENT LPAREN lconst_list RPAREN ASSIGN_EQUAL range_const SEMI
                         | IDENT ASSIGN_EQUAL range_const SEMI'''
        if len(p) == 6:
            p[0] = ((p[1], p[3]), True)
        elif len(p) == 3:
            p[0] = ((p[1], None), True)
        elif len(p) == 7:
            p[0] = ((p[2], p[4]), False)
        elif len(p) == 4:
            p[0] = ((p[2], None), False)
        elif len(p) == 8:
            p[0] = ((p[1], p[3]), p[6])
        elif len(p) == 5:
            p[0] = ((p[1], None), p[3])

    def p_lconst_list(self, p):
        '''lconst_list : lconst_list COMMA lconst
                       | lconst'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_string_list(self, p):
        '''string_list : string_list COMMA IDENT
                       | IDENT
                       | empty'''
        if p[1] is None:
            p[0] = []
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        line_err = p.lineno - 1
        lines = self._input.splitlines()
        line1 = max(line_err - 5, 0)
        line2 = min(line_err + 5, len(lines) - 1)
        
        exception_str = 'Syntax error on line {}:\n...'.format(line_err)
        for l in range(line1, line2):
            if l == line_err:
                exception_str += '\n >> ' + '\033[4m' + lines[l] + '\033[0m'
            else:
                exception_str += '\n   ' + lines[l]
        exception_str += '\n...'
        
        if self.debugging:
            exception_str += 'See log file {} for details.'.format(self.parsing_logfile)
        
        raise RDDLParseError(exception_str)

    def build(self, **kwargs):
        self._parser = yacc.yacc(module=self, **kwargs)

    def parse(self, input):
        self._input = input
        if self.debugging:
            self.parsing_logfile = os.path.join(tempfile.gettempdir(), 'rddl_parse.log')
            log = logging.getLogger(__name__)
            log.addHandler(logging.FileHandler(self.parsing_logfile))
            return self._parser.parse(input=input, lexer=self.lexer, debug=log)
        return self._parser.parse(input=input, lexer=self.lexer)

    def _print_verbose(self, p_name):
        if self._verbose:
            print('>> Parsed `{}` ...'.format(p_name))
