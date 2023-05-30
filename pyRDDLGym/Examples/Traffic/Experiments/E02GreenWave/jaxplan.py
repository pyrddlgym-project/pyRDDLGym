import jax
import optax
from math import inf
from sys import argv
import matplotlib.pyplot as plt
from datetime import datetime

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic

from jax.config import config as jconfig
jconfig.update('jax_debug_nans', True)
#jconfig.update('jax_platform_name', 'cpu')


# specify the model
EnvInfo = ExampleManager.GetEnvInfo('traffic2phase')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance='instance01.rddl')
model = myEnv.model


# initialize the planner
b_train = 1 # Deterministic transitions -> No need for larger batches?
b_test = 1
clip_grad = 1e-2
wrap_sigmoid = True

weight = 10
step = 100
nepochs = 5000
t_plan = myEnv.horizon
lr = 1e-3
momentum = 0.1

#initialize the plan
straight_line_plan = JaxStraightLinePlan(
    wrap_sigmoid=wrap_sigmoid
)

planner = JaxRDDLBackpropPlanner(
    model,
    plan=straight_line_plan,
    rollout_horizon=t_plan,
    batch_size_train=b_train,
    batch_size_test=b_test,
    optimizer=optax.rmsprop,
    optimizer_kwargs={'learning_rate': lr, 'momentum': momentum},
    clip_grad=clip_grad,
    logic=FuzzyLogic(
        weight=weight,
        eps=1e-8),
    action_bounds={'advance': (0.0, 1.0)})


# train for nepochs epochs using gradient ascent
# print progress every step epochs
train_return, test_return = [], []
grads, grads_min, grads_max = [], [], []

best_test_return = -inf
best_params = None
best_actions = None

key = jax.random.PRNGKey(3452)
gen = planner.optimize(
    epochs=nepochs,
    key=key,
    step=step,
    policy_hyperparams={'advance': weight})
for callback in gen:
    print('step={} train_return={:.6f} test_return={:.6f}'.format(
          str(callback['iteration']).rjust(4),
          callback['train_return'],
          callback['test_return']))
    train_return.append(callback['train_return'])
    test_return.append(callback['test_return'])

    if jax.numpy.isnan(train_return[-1]) or jax.numpy.isnan(test_return[-1]):
        raise RuntimeError

    if test_return[-1] > best_test_return:
        best_params, best_test_return = callback['params'], test_return[-1]
        best_actions = callback['action']['advance'][:,:,0]

    grads.append(callback['updates']['advance'])
    grads_min.append(jax.numpy.min(grads[-1]))
    grads_max.append(jax.numpy.max(grads[-1]))
    #print(grads[-1])
    print(jax.numpy.sum(callback['action']['advance'][:,:,0], axis=1))

