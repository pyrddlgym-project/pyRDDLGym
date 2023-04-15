import jax
import optax
from math import inf
from sys import argv
import matplotlib.pyplot as plt
from datetime import datetime

from jax.config import config as jconfig
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', 'cpu')

from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner, JaxRDDLBackpropPlannerMeta, JaxRDDLBackpropPlannerMetaWithRegularization
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic

# specify the model
manager = RDDLRepoManager()

EnvInfo = manager.get_problem('Navigation_MDP_ippc2011')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(1))
model = myEnv.model


# initialize the planner
b_train = 16
b_test = 128
nepochs = 20000
step = 100
clip_grad = None #1e1
fuzzy_logic_weight = 50
policy_hyperparams = {
    'move-north': 50.0,
    'move-south': 50.0,
    'move-east': 50.0,
    'move-west': 50.0
}


#initialize the plan
straight_line_plan = JaxStraightLinePlan(
)

planner = JaxRDDLBackpropPlannerMetaWithRegularization(
    model,
    reg_param=1,
    plan=straight_line_plan,
    batch_size_train=b_train,
    batch_size_test=b_test,
    optimizer=optax.rmsprop,
    optimizer_kwargs={'learning_rate': 0.1},
    clip_grad=clip_grad,
    logic=FuzzyLogic(
        weight=fuzzy_logic_weight,
        eps=1e-8),
    )


# train for nepochs epochs using gradient ascent
# print progress every step epochs
train_return, test_return = [], []

key = jax.random.PRNGKey(3452)
gen = planner.optimize(
    epochs=nepochs,
    key=key,
    step=step,
    policy_hyperparams=policy_hyperparams)
for callback in gen:
    print('step={} train_return={:.6f} test_return={:.6f}'.format(
          str(callback['iteration']).rjust(4),
          callback['train_return'],
          callback['test_return']))
    train_return.append(callback['train_return'])
    test_return.append(callback['test_return'])

    if jax.numpy.isnan(train_return[-1]) or jax.numpy.isnan(test_return[-1]):
        raise RuntimeError


plt.plot(range(len(train_return)), train_return, label='train')
plt.plot(range(len(test_return)), test_return, label='test')
plt.legend()
plt.savefig('pic.png')
