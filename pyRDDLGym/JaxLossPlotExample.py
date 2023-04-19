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


def _shape_info(planner):
    shapes, starts, sizes = {}, {}, {}
    istart = 0
    for name in planner.compiled.rddl.actions:
        shapes[name] = (planner.horizon,) + \
            np.shape(planner.compiled.init_values[name])
        starts[name] = istart
        sizes[name] = np.prod(shapes[name], dtype=int)
        istart += sizes[name]    
    count_entries = sum(sizes.values())
    return shapes, starts, sizes, count_entries

    
def loss_surface(problem, w=None, wa=None, train=True):
    
    # create the planning problem but non-aggregated return
    _, planner, train_args, _ = JaxConfigManager.get(
        f'{problem}.cfg')
    
    model_params = planner.compiled.model_params
    if w is not None:
        model_params = {name: w for name in model_params}
        
    hyperparams = train_args.get('policy_hyperparams', {})
    if wa is not None:
        hyperparams = {name: wa for name in hyperparams}
    
    # plan vector
    shapes, starts, sizes, _ = _shape_info(planner)
    
    def unravel_params(params):
        policy_params = {}
        for name in planner.compiled.rddl.actions:
            start = starts[name]
            policy_params[name] = jnp.reshape(
                params[start:start + sizes[name]], newshape=shapes[name])
        return policy_params
    
    # loss surface
    key = train_args['key']
    train_subs, test_subs = planner._batched_init_subs(
        planner.compiled.init_values if train else planner.test_compiled.init_values)
    subs = train_subs if train else test_subs
    loss_fn = planner.train_loss if train else planner.test_loss
    
    def loss_func(params):
        params = unravel_params(params)
        loss, _ = loss_fn(key, params, hyperparams, subs, model_params)
        return loss
        
    def loss_func_batched(params):
        return jnp.ravel(jax.vmap(loss_func)(params))
    
    return jax.jit(loss_func_batched)


def plot_surface(x, y, z, name):
    z = np.reshape(z, newshape=(x.size, y.size))
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Loss surface', autosize=False, width=1000, height=1000)
    fig.write_html(f'{name}.html')
    fig.show()


def run_experiment(problem, probname,
                   xmax=25.0, n=500, iters=100,
                   ws=[(100.0, 5.0), (100.0, 5.0), (10000.0, 100.0)]):
    
    # solve with default parameters
    _, planner, train_args, _ = JaxConfigManager.get(f'{problem}.cfg')
    sol_params = slp_train(planner, 60, **train_args)
    
    # reshape to solution vector
    _, starts, sizes, count_entries = _shape_info(planner)
    center = np.zeros((count_entries,))
    for name, value in sol_params.items():
        start = starts[name]
        center[start:start + sizes[name]] = np.ravel(value)
    
    # sample evaluation points and directions
    samples = n ** 2
    bs = samples // iters
    d1, d2 = sample_directions(count_entries)
    points, (x, y) = sample_points(center, d1, d2, -xmax, xmax, -xmax, xmax, n)
    
    # evaluate loss surfaces
    for i, (w, wa) in enumerate(ws):
        loss_func = loss_surface(problem, w=w, wa=wa, train=i > 0)
        zbatches = []
        for j in range(iters):
            print(f'batch {j}')
            batch = jnp.asarray(points[j * bs:j * bs + bs,:])
            zbatch = np.asarray(loss_func(batch))
            zbatches.append(zbatch)
        z = np.concatenate(zbatches)
        plot_surface(x, y, z, f'surface_{i}_{probname}')


if __name__ == '__main__':
    problem, probname = 'Wildfire', 'Wildfire'
    run_experiment(problem, probname)
