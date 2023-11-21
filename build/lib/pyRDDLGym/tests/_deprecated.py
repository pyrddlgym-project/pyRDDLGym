from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser import RDDLReader as RDDLReader
from pyRDDLGym.Core import Grounder as RDDLGrounder
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

DOMAIN = 'power_unit_commitment.rddl'

# DOMAIN = 'ThiagosReservoir.rddl'
# DOMAIN = 'Thiagos_Mars_Rover.rddl'
# DOMAIN = 'dbn_prop.rddl'
# DOMAIN = 'RamMod_Thiagos_HVAC.rddl'
DOMAIN = 'RamMod_smaller_Thiagos_HVAC.rddl'
# DOMAIN = 'RamMod_Thiagos_HVAC_grounded.rddl'

# DOMAIN = 'Thiagos_HVAC.rddl'
# DOMAIN = 'Thiagos_HVAC_grounded.rddl'
# DOMAIN = 'wildfire_mdp.rddl'
DOMAIN = 'recsim_ecosystem_welfare.rddl'

def main():

    MyReader = RDDLReader.RDDLReader('RDDL/' + DOMAIN)
    # MyReader = RDDLReader.RDDLReader('RDDL/power_unit_commitment_domain.rddl',
    #                                  'RDDL/power_unit_commitment_instance.rddl')
    domain = MyReader.rddltxt

    MyLexer = parser.RDDLlex()
    MyLexer.build()
    MyLexer.input(domain)
    # [token for token in MyLexer._lexer]

    # build parser - built in lexer, non verbose
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()

    # parse RDDL file
    rddl_ast = MyRDDLParser.parse(domain)

    # MyXADDTranslator = XADDTranslator(rddl_ast)
    # MyXADDTranslator.Translate()

    grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
    model = grounder.Ground()
    # print(model.states)
    # print(model.init_state)
    # for cpf_key, primed_cpf in model.interm.items():
    #     expr = model.cpfs[primed_cpf]
    #     print(primed_cpf, expr)
    # print(model.cpfs)
    # pprint(vars(model))
    # #
    # good_policy = True
    sampler = RDDLSimulator(model)
    loops = 1
    #
    print('\nstarting simulation')
    for h in range(loops):
        state = sampler.reset_state()
        print(state)
        action = {
        'recommend(c1, i1)': True,
        'recommend(c2, i2)': True,
        'recommend(c3, i3)': True,
        'recommend(c4, i4)': True,
        'recommend(c5, i5)': True,
        }
        state = sampler.sample_next_state(action)
        print(state)
    #     total_reward = 0.
    #     for _ in range(2000):
    #         sampler.check_state_invariants()
    #         sampler.check_action_preconditions()
    #         actions = {'AIR_r1': 0., 'AIR_r2': 0., 'AIR_r3': 0.}
    #         if good_policy:
    #             if state['TEMP_r1'] < 20.5:
    #                 actions['AIR_r1'] = 5.
    #             if state['TEMP_r2'] < 20.5:
    #                 actions['AIR_r2'] = 5.
    #             if state['TEMP_r3'] < 20.5:
    #                 actions['AIR_r3'] = 5.
    #             if state['TEMP_r1'] > 23.:
    #                 actions['AIR_r1'] = 0.
    #             if state['TEMP_r2'] > 23.:
    #                 actions['AIR_r2'] = 0.
    #             if state['TEMP_r3'] > 23.:
    #                 actions['AIR_r3'] = 0.
    #         state = sampler.sample_next_state(actions)
    #         reward = sampler.sample_reward()
    #         sampler.update_state()
    #         if h == 0:
    #             print('state = {}'.format(state))
    #             print('reward = {}'.format(reward))
    #             print('derived = {}'.format(model.derived))
    #             print('interm = {}'.format(model.interm))
    #             print('')
    #         total_reward += reward
    #     print('trial {}, total reward {}'.format(h, total_reward))

    # grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
    # grounder.Ground()
    # pprint(vars(grounder))

    # grounder.InitGround()
    # generator = RDDLGenerator.RDDLGenerator(rddl_ast)
    # rddl = generator.GenerateRDDL()


    print("reached end of test.py")

if __name__ == "__main__":
    main()




# from Parser import parser as parser
# from Parser import RDDLReader as RDDLReader
# import Grounder.RDDLGrounder as RDDLGrounder
# from Visualizer.MarsRoverDisplay import MarsRoverDisplay
# from Visualizer.ReservoirDisplay import ReservoirDisplay
# from Visualizer.HVACDisplay import HVACDisplay

# DOMAIN = 'power_unit_commitment.rddl'

# # DOMAIN = 'ThiagosReservoir.rddl'
# DOMAIN = 'ThiagosReservoir_grounded.rddl'
# # DOMAIN = 'Thiagos_Mars_Rover.rddl'
# # DOMAIN = 'Thiagos_Mars_Rover_grounded.rddl'
# # DOMAIN = 'Thiagos_HVAC.rddl'
# # DOMAIN = 'dbn_prop.rddl'
# # DOMAIN = 'Thiagos_HVAC_grounded.rddl'
# # DOMAIN = 'wildfire_mdp.rddl'
# DOMAIN = 'recsim_ecosystem_welfare.rddl'

# def main():

#     MyReader = RDDLReader.RDDLReader('RDDL/' + DOMAIN)
#     # MyReader = RDDLReader.RDDLReader('RDDL/power_unit_commitment_domain.rddl',
#     #                                  'RDDL/power_unit_commitment_instance.rddl')
#     domain = MyReader.rddltxt

#     MyLexer = parser.RDDLlex()
#     MyLexer.build()
#     MyLexer.input(domain)
#     # [token for token in MyLexer._lexer]

#     # build parser - built in lexer, non verbose
#     MyRDDLParser = parser.RDDLParser(None, False)
#     MyRDDLParser.build()

#     # parse RDDL file
#     rddl_ast = MyRDDLParser.parse(domain)



#     # MyXADDTranslator = XADDTranslator(rddl_ast)
#     # MyXADDTranslator.Translate()

#     from pprint import pprint
#     grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
#     model = grounder.Ground()
#     # print(model.states)
#     # print(model.init_state)
#     # for cpf_key, primed_cpf in model.interm.items():
#     #     expr = model.cpfs[primed_cpf]
#     #     print(primed_cpf, expr)
#     # print(model.cpfs)
#     # pprint(vars(model))
#     # #
#     # good_policy = True
#     sampler = RDDLSimulator(model)
#     loops = 1
#     #
#     print('\nstarting simulation')
#     for h in range(loops):
#         state = sampler.reset_state()
#         print(state)
#         action = {
#         'recommend(c1, i1)': True,
#         'recommend(c2, i2)': True,
#         'recommend(c3, i3)': True,
#         'recommend(c4, i4)': True,
#         'recommend(c5, i5)': True,
#         }
#         state = sampler.sample_next_state(action)
#         print(state)
#     #     total_reward = 0.
#     #     for _ in range(2000):
#     #         sampler.check_state_invariants()
#     #         sampler.check_action_preconditions()
#     #         actions = {'AIR_r1': 0., 'AIR_r2': 0., 'AIR_r3': 0.}
#     #         if good_policy:
#     #             if state['TEMP_r1'] < 20.5:
#     #                 actions['AIR_r1'] = 5.
#     #             if state['TEMP_r2'] < 20.5:
#     #                 actions['AIR_r2'] = 5.
#     #             if state['TEMP_r3'] < 20.5:
#     #                 actions['AIR_r3'] = 5.
#     #             if state['TEMP_r1'] > 23.:
#     #                 actions['AIR_r1'] = 0.
#     #             if state['TEMP_r2'] > 23.:
#     #                 actions['AIR_r2'] = 0.
#     #             if state['TEMP_r3'] > 23.:
#     #                 actions['AIR_r3'] = 0.
#     #         state = sampler.sample_next_state(actions)
#     #         reward = sampler.sample_reward()
#     #         sampler.update_state()
#     #         if h == 0:
#     #             print('state = {}'.format(state))
#     #             print('reward = {}'.format(reward))
#     #             print('derived = {}'.format(model.derived))
#     #             print('interm = {}'.format(model.interm))
#     #             print('')
#     #         total_reward += reward
#     #     print('trial {}, total reward {}'.format(h, total_reward))

#     grounder = RDDLGrounder.RDDLGroundedGrounder(rddl_ast)
#     model = grounder.Ground()
#     # marsVisual = MarsRoverDisplay(model, grid_size=[50,50], resolution=[500,500])
#     # marsVisual.display_img(duration=0.5)
#     # marsVisual.save_img('./pict2.png')

#     # reservoirVisual = ReservoirDisplay(model, grid_size=[50,50], resolution=[500,500])

#     HVACVisual = HVACDisplay(model, grid_size=[50,50], resolution=[500,500])


#     # print(model._nonfluents)
#     # print(model._states)
#     # print(model._objects)



#     # generator = RDDLGenerator.RDDLGenerator(rddl_ast)
#     # rddl = generator.GenerateRDDL()
#     # print(rddl)

#     print("reached end of test.py")

# if __name__ == "__main__":
#     main()



# GROUNDER OLD
# class RDDLGroundedGrounder(Grounder):
#
#   def __init__(self, RDDL_AST):
#     super(RDDLGroundedGrounder, self).__init__()
#     self._AST = RDDL_AST
#     # self._Model = None
#     self._actions = {}
#     self._nonfluents = {}
#     self._states = {}
#     self._nextstates = {}
#     self._prevstates = {}
#     self._init_state = {}
#     self._cpfs = {}
#     self._cpforder = {0: []}
#     self._derived = {}
#     self._interm = {}
#     self._objects = {}
#
#     self._reward = None
#     self._preconditions = []
#     self._invariants = []
#
#     self._actionsranges = {}
#     self._statesranges = {}
#
#   def Ground(self):
#     # there are no objects or types in grounded domains
#
#     # initialize the Model object
#     model = RDDLModel()
#
#     self._getObjects()
#
#     # ground pvariables and appropriate cpfs if applicable
#     # update pvariables
#     self._groundPvariables()
#
#     # update non_fluents values in case the default values were overridden in the instance
#     self._groundNonfluents()
#
#     # ground init_state
#     self._groundInitState()
#
#     # ground reward
#     self._groundReward()
#
#     # ground constraints
#     self._groundPreConstraints()
#
#     # update model object
#     model.states = self._states
#     model.actions = self._actions
#     model.nonfluents = self._nonfluents
#     model.next_state = self._nextstates
#     model.prev_state = self._prevstates
#     model.init_state = self._init_state
#     model.cpfs = self._cpfs
#     model.cpforder = self._cpforder
#     model.reward = self._reward
#     model.preconditions = self._preconditions
#     model.invariants = self._invariants
#     model.derived = self._derived
#     model.interm = self._interm
#
#     # new properties
#     model.max_allowed_actions = self._groundMaxActions()
#     model.horizon = self._groundHorizon()
#     model.discount = self._groundDiscount()
#     model.actionsranges = self._actionsranges
#     model.statesranges = self._statesranges
#     model.objects = self._objects
#     # new properties
#
#     return model
#
#   def _getObjects(self):
#     self._objects = {}
#     try:
#       for type in self._AST.non_fluents.objects:
#         self._objects[type[0]] = type[1]
#     except:
#       return
#
#   def _groundHorizon(self):
#     return self._AST.instance.horizon
#
#   def _groundMaxActions(self):
#     numactions = self._AST.instance.max_nondef_actions
#     if numactions == 'pos-inf':
#       return len(self._actions)
#     else:
#       return int(numactions)
#
#   def _groundDiscount(self):
#     return self._AST.instance.discount
#
#   def _groundPvariables(self):
#     for pvariable in self._AST.domain.pvariables:
#       name = pvariable.name
#       if pvariable.fluent_type == 'non-fluent':
#         self._nonfluents[name] = pvariable.default
#       elif pvariable.fluent_type == 'action-fluent':
#         self._actions[name] = pvariable.default
#         self._actionsranges[name] = pvariable.range
#       elif pvariable.fluent_type == 'state-fluent':
#         cpf = None
#         next_state = name + '\''
#         for cpfs in self._AST.domain.cpfs[1]:
#           if cpfs.pvar[1][0] == next_state:
#             cpf = cpfs
#         if cpf is not None:
#           self._states[name] = pvariable.default
#           self._statesranges[name] = pvariable.range
#           self._nextstates[name] = next_state
#           self._prevstates[next_state] = name
#           self._cpfs[next_state] = cpf.expr
#           self._cpforder[0].append(name)
#       elif pvariable.fluent_type == 'derived-fluent':
#         cpf = None
#         for cpfs in self._AST.domain.derived_cpfs:
#           if cpfs.pvar[1][0] == name:
#             cpf = cpfs
#         if cpf is not None:
#           self._derived[name] = pvariable.default
#           self._cpfs[name] = cpf.expr  #sim expects expression here.
#           level = pvariable.level
#           if level is None:
#             level = 1
#           if level in self._cpforder:
#             self._cpforder[level].append(name)
#           else:
#             self._cpforder[level] = [name]
#       elif pvariable.fluent_type == 'interm-fluent':
#         cpf = None
#         for cpfs in self._AST.domain.intermediate_cpfs:
#           if cpfs.pvar[1][0] == name:
#             cpf = cpfs
#         if cpf is not None:
#           self._interm[name] = pvariable.default
#           self._cpfs[name] = cpf.expr
#           level = pvariable.level
#           if level is None:
#             level = 1
#           if level in self._cpforder:
#             self._cpforder[level].append(name)
#           else:
#             self._cpforder[level] = [name]
#
#   def _groundNonfluents(self):
#     if hasattr(self._AST.non_fluents, 'init_non_fluent'):
#       for init_vals in self._AST.non_fluents.init_non_fluent:
#         key = init_vals[0][0]
#         val = init_vals[1]
#         self._nonfluents[key] = val
#
#   def _groundInitState(self):
#     self._init_state = self._states.copy()
#     if hasattr(self._AST.instance, 'init_state'):
#       for init_vals in self._AST.instance.init_state:
#         key = init_vals[0][0]
#         val = init_vals[1]
#         self._init_state[key] = val
#
#   def _groundReward(self):
#     self._reward = self._AST.domain.reward
#
#   def _groundPreConstraints(self):
#     if hasattr(self._AST.domain, 'preconds'):
#       for precond in self._AST.domain.preconds:
#         self._preconditions.append(precond)
#
#     if hasattr(self._AST.domain, 'invariants'):
#       for inv in self._AST.domain.invariants:
#         self._invariants.append(inv)


# import os
# import sys
#
# from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
# from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
# from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
# from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
# from pyRDDLGym.Examples.ExampleManager import ExampleManager
#
#
# def main(domain, instance):
#
#     # set up the environment
#     info = ExampleManager.GetEnvInfo(domain)    
#     env = RDDLEnv.build(info, instance, enforce_action_constraints=True)
#
#     # load the config file with planner settings
#     abs_path = os.path.dirname(os.path.abspath(__file__))
#     config_path = os.path.join(abs_path, 'JaxPlanConfigs', f'{domain}_slp.cfg') 
#     planner_args, _, train_args = load_config(config_path)
#
#     # create the planning algorithm and controller
#     planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
#     controller = JaxOfflineController(planner, params={}, **train_args)
#
#     # expand budget in config
#     train_args['train_seconds'] = 60
#
#     # train for 10 seconds, evaluate, then repeat
#     eval_period = 10
#     time_last_eval = 0
#     for callback in planner.optimize_generator(**train_args):
#         if callback['elapsed_time'] - time_last_eval > eval_period:
#             controller.params = callback['best_params']
#             controller.evaluate(env, verbose=False, render=True)
#             time_last_eval = callback['elapsed_time']
#
#     env.close()
#
# if __name__ == "__main__":
#     args = sys.argv[1:]
#     if len(args) < 2:
#         print('python JaxExample2.py <domain> <instance>')
#         exit(0)
#     domain, instance = args[:2]
#     main(domain, instance)
#

