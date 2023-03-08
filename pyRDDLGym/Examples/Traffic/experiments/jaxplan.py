import jax
import optax
from math import inf
from sys import argv

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic

try:
    weight = float(argv[1])
except KeyError as e:
    print('cmd.args: w')
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

for step in range(t_warmup):
    action = {'advance___i0': (step%webster_CT in webster_change_ts)}
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

nepochs = 200000
step = 1000
b_train = 8
b_test = 8
lr = 1e-2
clip_grad = 3e-2
#weight = argv[1]


while (lr > 1e-4):
    print('LEARNING RATE = ', lr)
    #initialize the plan
    straight_line_plan = JaxStraightLinePlan()

    planner = JaxRDDLBackpropPlanner(
        model,
        plan=straight_line_plan,
        rollout_horizon=t_plan,
        batch_size_train=b_train,
        batch_size_test=b_test,
        optimizer=optax.rmsprop(lr), #, momentum=0.9),
        #optimizer=optax.adam(lr),
        clip_grad=clip_grad,
        logic=FuzzyLogic(weight=weight),
        action_bounds={'advance': (0.0, 1.0)})


    # train for nepochs epochs using gradient ascent
    # print progress every step epochs
    train_return, test_return = [], []
    grads = []

    best_test_return = -inf
    best_params = None
    best_actions = None

    key = jax.random.PRNGKey(3452)
    gen = planner.optimize(
        epochs=nepochs,
        key=key,
        step=step,
        subs=init_state_subs)
    for callback in gen:
        print('step={} train_return={:.6f} test_return={:.6f}'.format(
              str(callback['iteration']).rjust(4),
              callback['train_return'],
              callback['test_return']))
        train_return.append(callback['train_return'])
        test_return.append(callback['test_return'])

        if jax.numpy.isnan(train_return[-1]) or jax.numpy.isnan(test_return[-1]):
            lr = lr/2
            # break optimization loop after reducing lr
            gen.close()
            break

        if test_return[-1] > best_test_return:
            best_params, best_test_return = callback['params'], test_return[-1]
            best_actions = callback['action']['advance'][:,:,0]
        grads.append(callback['grad']['advance'])
        #print(grads[-1])
        print(jax.numpy.sum(callback['action']['advance'][:,:,0], axis=1))
    else:
        # break outer while loop
        break


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



import matplotlib.pyplot as plt
Xs = [step*x for x in range(len(train_return))]
plt.plot(Xs, train_return, label='train')
plt.plot(Xs, test_return, label='test')
plt.title('JAXPlan reward curves on Single Intersection/Webster scenario\n'
          f'Straight line plan, {b_train},{b_test} batch-train,test-sizes, RMSProp {lr}, weight {weight}')
plt.xlabel('steps')
plt.ylabel('TTT')
plt.tight_layout()
plt.savefig(f'webster_planningcrvs___T{t_plan}___lr{lr}___weight{weight}___btr{b_train}_btst{b_test}___clipgrad{clip_grad}.png')
