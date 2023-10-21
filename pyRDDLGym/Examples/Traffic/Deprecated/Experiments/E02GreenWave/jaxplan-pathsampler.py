import jax
import optax
import jax.numpy as jnp
from jax.nn import initializers as jin
#import jax.random
from math import inf
import pickle
import json
from sys import argv
import matplotlib.pyplot as plt
from datetime import datetime

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic

from utils import warmup
from utils import offset_plan

from jax.config import config as jconfig
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', 'gpu')

from sys import argv
n = float(int(argv[1]))
seed = int(argv[2])

key = jax.random.PRNGKey(seed)

# specify the model
EnvInfo = ExampleManager.GetEnvInfo('traffic2phase')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance='instances/instance01.rddl')
model = myEnv.model
N = myEnv.numConcurrentActions

init_state_subs, t_warmup = warmup(myEnv, EnvInfo)

# initialize the planner
b_train = 32 # Deterministic transitions -> No need for larger batches?
b_test = 32
clip_grad = 1e-2
wrap_sigmoid = True

weight = 15
step = 100
nepochs = 3000
t_plan = myEnv.horizon
rollout_horizon = t_plan - t_warmup
lr = 1e-2
momentum = 0.1

#initialize the plan
initializer_actions = jnp.array([offset_plan(i)[:rollout_horizon] for i in range(N)], dtype=float)
initializer_actions = 2*initializer_actions - 1

key, subkey = jax.random.split(key)
fixed_random_policy = jax.random.normal(subkey, shape=initializer_actions.shape)
initializer_actions = (n/10)*initializer_actions + ((10-n)/10)*fixed_random_policy

initializer_actions = jnp.swapaxes(initializer_actions, 0, 1)
initializer_shape = (rollout_horizon, N)

straight_line_plan = JaxStraightLinePlan(
    initializer=jin.constant(initializer_actions, initializer_shape),
    wrap_sigmoid=wrap_sigmoid
)

planner = JaxRDDLBackpropPlanner(
    model,
    plan=straight_line_plan,
    rollout_horizon=rollout_horizon,
    batch_size_train=b_train,
    batch_size_test=b_test,
    optimizer=optax.rmsprop,
    optimizer_kwargs={'learning_rate': lr, 'momentum': momentum},
    use64bit=True,
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
best_params, best_actions = None, None

gen = planner.optimize(
    epochs=nepochs,
    key=key,
    step=step,
    subs=init_state_subs,
    policy_hyperparams={'advance': weight})
for callback in gen:
    #print('step={} train_return={:.6f} test_return={:.6f}'.format(
    #      str(callback['iteration']).rjust(4),
    #      callback['train_return'],
    #      callback['test_return']))
    train_return.append(callback['train_return'])
    test_return.append(callback['test_return'])

    if jax.numpy.isnan(train_return[-1]) or jax.numpy.isnan(test_return[-1]):
        raise RuntimeError

    if test_return[-1] > best_test_return:
        best_params, best_test_return = callback['params'], test_return[-1]
        best_actions = callback['action']['advance']


now = datetime.now().strftime('%Y%m%d_%H%M%S')

with open('actions/log.txt', 'a') as file:
    print(now, 'N=', n, 'Initial return=', test_return[0], 'Best return=', best_test_return, 'Improvement score=', (test_return[0]-best_test_return)/test_return[0], file=file)
with open(f'actions/actions_dump_{now}_n{n}.json', 'w') as file:
    json.dump({'a': best_actions.tolist()}, file)

fig, ax = plt.subplots()
X = [step*n for n in range(len(train_return))]
ax.plot(X, train_return, label='Train')
ax.plot(X, test_return, label='Test')
ax.legend()
ax.grid(visible=True)
plt.savefig(f'img/progress_w{weight}_lr{lr}_m{momentum}_{now}_n{n}.png')
