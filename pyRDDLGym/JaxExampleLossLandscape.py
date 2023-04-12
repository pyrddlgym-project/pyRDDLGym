import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from pyRDDLGym.JaxExample import slp_train
from pyRDDLGym.Planner import JaxConfigManager


def sample_points(center, dir1, dir2, x1min, x1max, x2min, x2max, n):
    x1 = np.linspace(x1min, x1max, n)
    x2 = np.linspace(x2min, x2max, n)
    grid = np.meshgrid(x1, x2)
    dirs = np.column_stack((dir1, dir2))
    points = np.einsum('ij,jkl->ikl', dirs, np.array(grid))
    points = np.reshape(points, newshape=(points.shape[0], -1)).T
    points += center
    return points, (x1, x2)


def sample_directions(dim, loc=0.0, scale=1.0):
    dir1 = np.random.normal(loc=loc, scale=scale, size=(dim,))
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = np.random.normal(loc=loc, scale=scale, size=(dim,))
    dir2 = dir2 / np.linalg.norm(dir2)
    return dir1, dir2        


def loss_surface(problem, timeout, w=None, wa=None, solve=False):
    
    # solve the planning problem to get center for the plot
    if solve:
        _, planner, train_args, _ = JaxConfigManager.get(f'{problem}.cfg')
        sol_params = slp_train(planner, timeout, **train_args)
    
    # create the planning problem but non-aggregated return
    myEnv, planner, train_args, _ = JaxConfigManager.get(
        f'{problem}.cfg', utility=(lambda x: x))
    rddl = myEnv.model
    key = train_args['key']
    subs = planner.test_compiled.init_values
    train_subs, _ = planner._batched_init_subs(subs)
    hyperparams = train_args.get('policy_hyperparams', {})
    model_params = planner.compiled.model_params
    
    if w is not None:
        model_params = {name: w for name in model_params}
    if wa is not None:
        hyperparams = {name: wa for name in hyperparams}
    
    # how to slice action vector to dict 
    shapes, starts, sizes = {}, {}, {}
    istart = 0
    for name in rddl.actions:
        shapes[name] = (planner.horizon,) + \
            np.shape(planner.compiled.init_values[name])
        starts[name] = istart
        sizes[name] = np.prod(shapes[name], dtype=int)
        istart += sizes[name]    
    count_entries = sum(sizes.values())
    print(f'total actions = {count_entries}')
    
    # convert solution to action vector
    center = None
    if solve:
        center = np.zeros((count_entries))
        for name, value in sol_params.items():
            start = starts[name]
            center[start:start + sizes[name]] = np.ravel(value)
    
    # loss surface
    def unravel_params(params):
        policy_params = {}
        for name in rddl.actions:
            start = starts[name]
            policy_params[name] = jnp.reshape(
                params[start:start + sizes[name]], newshape=shapes[name])
        return policy_params
    
    def loss_func(params):
        policy_params = unravel_params(params)
        loss_values, _ = planner.train_loss(
            key, policy_params, hyperparams, train_subs, model_params)
        mean_loss = jnp.mean(loss_values)
        return mean_loss
    
    def loss_func_batched(params):
        loss_values = jax.vmap(loss_func)(params)
        loss_values = jnp.ravel(loss_values)
        return loss_values
    
    loss_func_jit = jax.jit(loss_func_batched)
    
    return loss_func_jit, count_entries, center


def plot_surface(x, y, z, name):
    z = np.reshape(z, newshape=(x.size, y.size))
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Loss surface')
    fig.write_html(f'{name}.html')
    fig.show()


def run_experiment(problem, probname,
                   xmax=20.0, n=500, iters=100, ws=[2, 20, 200], timeout=60):
    x1min, x1max = -xmax, xmax
    x2min, x2max = -xmax, xmax
    samples = n ** 2
    bs = samples // iters
    
    center, dir1, dir2, points, x, y = None, None, None, None, None, None
    for i, w in enumerate(ws):
        
        # return the loss surface function
        # if the first iteration, solve the problem and use result as plot center
        loss_func, entries, solution = loss_surface(
            problem, timeout, w=float(w), wa=float(w), solve=i == 0)
        
        # cache the first solution, direction vectors and compute samples
        # of plans at which to evaluate loss surface
        if i == 0:
            center = solution
            dir1, dir2 = sample_directions(entries)
            points, (x, y) = sample_points(
                center, dir1, dir2, x1min, x1max, x2min, x2max, n)
        
        # perform the batched evaluation of the loss surface
        zbatches = []
        for i in range(iters):
            print(f'batch {i}')
            batch = jnp.asarray(points[i * bs:i * bs + bs,:])
            zbatch = np.asarray(loss_func(batch))
            zbatches.append(zbatch)
        z = np.concatenate(zbatches)
        
        # render the loss surface
        plot_surface(x, y, z, f'surface_{w}_{probname}')


if __name__ == '__main__':
    problem, probname = 'Wildfire', 'Wildfire'
    run_experiment(problem, probname)
