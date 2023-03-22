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

try:
    weight = float(argv[1])
    step = int(argv[2])
    nepochs = int(argv[3])
    optype = argv[4]
    lr = float(argv[5])
    momentum = float(argv[6])
except IndexError as e:
    print('cmd.args: w, step, nepochs, optype, lr, momentum')
    raise e

# specify the model
EnvInfo = ExampleManager.GetEnvInfo('traffic')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance='webster_exp_equal_split.rddl')
model = myEnv.model

# save state (subs) after warmup
webster_change_ts = {32, 36, 69, 73, 106, 110, 143, 147}
webster_CT = 148

t_warmup = 2 * webster_CT
t_plan = myEnv.horizon - t_warmup

for wstep in range(t_warmup):
    action = {'advance___i0': (wstep%webster_CT in webster_change_ts)}
    next_state, reward, info, done = myEnv.step(action)
    state = next_state
    if done:
        raise RuntimeError('[jaxplan.py] Environment finished during warmup')

init_state_subs = myEnv.sampler.subs


# initialize the planner
""" good settings for horizon=600 (with warmup 300)
nepochs = 15000
step = 100
b_train = 8
b_test = 8
lr = 1e-2
clip_grad = 1e-2
weight = 90 """

b_train = 2
b_test = 2
clip_grad = 1e-3
wrap_sigmoid = True

if optype == 'rmsprop':
    optimizer = optax.rmsprop(lr, momentum=momentum)
    optitle = f'RMSProp(lr={lr}, momentum={momentum})'
elif optype == 'adam':
    optimizer = optax.adam(lr)
    optitle = f'Adam(lr={lr})'
else:
    raise ValueError

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
    optimizer=optimizer,
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
    subs=init_state_subs,
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


"""
fig, ax = plt.subplots(1, 2)
X = jax.numpy.arange(0, nepochs+step-1, step)
ax[0].plot(X, grads_min, label='min')
ax[0].plot(X, grads_max, label='max')
ax[0].set_xlabel('steps')
ax[0].set_ylabel('update bounds')
ax[0].set_title('Update magnitude')
ax[0].legend()
ax[0].grid(visible=True)

#myEnv.set_visualizer(EnvInfo.get_visualizer())

#warmup
state = myEnv.reset()
for step in range(t_warmup):
    action = {'advance___i0': (step%webster_CT in webster_change_ts)}
    next_state, reward, info, done = myEnv.step(action)
    state = next_state

#eval of best plan
plan = best_actions[0]
total_reward = 0
for step in range(t_warmup, myEnv.horizon):
    #myEnv.render()
    #key, subkey = jax.random.split(key, num=2)
    #action = planner.get_action(subkey, best_params, step, subs=myEnv.sampler.subs)
    #plan.append(action['advance___i0'])
    action = plan[step-t_warmup]
    next_state, reward, info, done = myEnv.step({'advance___i0': action})
    state = next_state
    total_reward += reward
    if done:
        break

print(plan)
print('Cmlt.reward:', total_reward)
print('Number of CHANGEs', sum(plan))



ax[1].plot(X, train_return, label='train')
ax[1].plot(X, test_return, label='test')
ax[1].set_xlabel('steps')
ax[1].set_ylabel('TTT')
ax[1].legend()
ax[1].grid(visible=True)
ax[1].set_title('Webster result')
plt.suptitle(f'optimizer={optitle}. w={weight}. wrap_sigmoid={wrap_sigmoid}')

plt.tight_layout()
time = datetime.now()
timestamp = time.strftime("%H:%M:%S")
plt.savefig(f'img/webster_{timestamp}.png')
"""
