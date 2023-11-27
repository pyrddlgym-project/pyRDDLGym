"""A tool for generating instances of the SumOfHalfSpaces domain"""

import numpy as np


def apply_objective_fn(x, W, B):
    B = B[:,np.newaxis,:]
    res = np.inner(W, x-B)
    res = np.diagonal(res)
    res = np.sign(res)
    return np.sum(res, axis=1)


def render_one_dim_objective(W, B, save_to=None):
    """Plots a 2-d graph of the objective function

       f(x) = sum_i sign(w_i * (x - b_i))

    W and B have shapes (n_summands, 1)
    """
    import matplotlib.pyplot as plt
    xs = np.arange(-10., 10., step=0.1)
    ys = apply_objective_fn(xs[np.newaxis,...,np.newaxis], W, B)
    plt.plot(xs, ys)
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
        plt.clf()
    else:
        plt.show()

def render_two_dim_objective(W, B, save_to=None):
    """Plots a contour plot of the objective function

        f(x) = sum_i sign(w_i dot (x - b_i))

    W and B have shapes (n_summands, 2)
    """
    import matplotlib.pyplot as plt
    xs = np.arange(-10., 10., step=0.1)

    xs = np.meshgrid(xs, xs)
    ps = np.concatenate([xs[0][..., np.newaxis], xs[1][..., np.newaxis]], axis=2)

    zs = apply_objective_fn(ps.reshape(1,40000,2), W, B).reshape(200,200)
    zs_min = np.min(zs)
    zs_max = np.max(zs)

    fig, ax = plt.subplots()

    for w, b in zip(W, B):
        d = np.array([-w[1], w[0]])
        norm_d = np.linalg.norm(d)
        t_span = np.minimum(100, (20 * 1.414 / (norm_d + 1e-8)))
        t_range = np.arange(-t_span, t_span)
        d = d[..., np.newaxis]
        line = b[..., np.newaxis] + d*t_range
        ax.plot(line[0], line[1], color='black')

    ax.set_xlim((-10,10))
    ax.set_ylim((-10,10))
    im = ax.imshow(zs, origin='lower', extent=(-10,10,-10,10))

    plt.colorbar(im)
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()



single_indent_str = ' ' * 4
double_indent_str = ' ' * 8

def generate(n_dim,
             n_summands,
             W_mean=0.,
             W_cov=1.,
             B_mean=0.,
             B_cov=5.,
             instance_name=None,
             render_objective_fn=False):
    """Generates a random RDDL instance file (values of the nonfluent
    and the definition of the domain instance).

    The objective function is given by

        f(a) = sum_i sign(W_i dot (a - B_i))

    where W_i, a, B_i are all n_dim-dimensional, and there are n_summands
    terms in the sum. The values of W_i and B_i are generated from
    a multivariable normal distribution.

    Arguments
    ---------
        n_dim : int
            Number of dimensions
        n_sumamnds : int
            Number of summands used to define the objective function
        W_mean, W_cov: float or np.array
            Parameters of the Multivariate Normal used to generate
            the W vector. If floats are passed, it is assumed that
            the means are uniform and the covariance is W_cov * Id
        B_mean, B_cov : float or np.array
            Similar to W_mean and W_cov, respectively
        instance_name : str
            The name of the instance. If None, automatically generated
            based on the values of the other parameters
        render_objective_fn : bool
            Whether or not to plot the graph (1d) or contour plot (2d)
            of the objective function. Only 1d and 2d are implemented.
    """

    if instance_name is None:
        instance_name = f'dim{n_dim}_nsummands{n_summands}'
    if isinstance(W_mean, float):
        W_mean = np.ones(shape=(n_dim,)) * W_mean
    if isinstance(W_cov, float):
        W_scale = np.ones(n_dim) * np.sqrt(W_cov)
    if isinstance(B_mean, float):
        B_mean = np.ones(shape=(n_dim,)) * B_mean
    if isinstance(B_cov, float):
        B_scale = np.ones(n_dim) * np.sqrt(B_cov)

    W = np.random.normal(
        loc=W_mean,
        scale=W_scale,
        size=(n_summands, n_dim))
    B = np.random.normal(
        loc=B_mean,
        scale=B_scale,
        size=(n_summands, n_dim))

    W_str = ''
    for i in range(n_summands):
        for j in range(n_dim):
            W_str += '\n' + double_indent_str + f'W(s{i},d{j}) = {W[i,j]:.16f};'

    B_str = ''
    for i in range(n_summands):
        for j in range(n_dim):
            B_str += '\n' + double_indent_str + f'B(s{i},d{j}) = {B[i,j]:.16f};'


    instance_str = f'non-fluents sum_of_half_spaces_{instance_name} {{'
    instance_str += ('\n' + single_indent_str).join((
        '',
        'domain = sum_of_half_spaces;',
        '',
        'objects {',
        single_indent_str + f'summand : {{{", ".join(f"s{i}" for i in range(n_summands))}}};',
        single_indent_str + f'dimension : {{{", ".join(f"d{i}" for i in range(n_dim))}}};',
        '};',
        '',
        'non-fluents {' + W_str + B_str,
        '};'
    ))
    instance_str += '\n}'
    instance_str += '\n'
    instance_str += ('\n' + single_indent_str).join((
        f'instance sum_of_half_spaces_{instance_name}_inst {{',
        f'domain = sum_of_half_spaces;',
        f'non-fluents = sum_of_half_spaces_{instance_name};',
        f'max-nondef-actions = {n_dim};',
        f'horizon = 1;',
        f'discount = 1.0;'))
    instance_str += '}'

    if render_objective_fn:
        if n_dim == 1:
            render_one_dim_objective(W, B)
        elif n_dim == 2:
            render_two_dim_objective(W, B)
        else:
            raise RuntimeError('[generate] For dim > 2, rendering the generated objective fn is not supported')

    return instance_str




if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Tool for automatically generating grid instances for the RDDL SumOfHalfSpaces domain')
    parser.add_argument('target_path', type=str, help='Path the generated rddl code will be saved to')
    parser.add_argument('-d', '--dims', type=int, help='Number of dimensions of the problem', required=True)
    parser.add_argument('-s', '--summands', type=int, help='Number of summands in the objective function', required=True)
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='By default the generator will not overwrite existing files. With this argument, it will')
    parser.add_argument('-n', '--instance-name', help='Name of instance')
    parser.add_argument('-r', '--render', action='store_true', help='If TRUE: Render the generated objective function. Only 1-d and 2-d supported.')
    args = parser.parse_args()

    if os.path.isfile(args.target_path) and not args.force_overwrite:
        raise RuntimeError('[generator.py] File with the requested path already exists. Pass a diffent path or add the -f argument to force overwrite')

    with open(args.target_path, 'w') as file:
        instance_rddl = generate(
            args.dims, args.summands,
            instance_name=args.instance_name,
            render_objective_fn=args.render)

        file.write(instance_rddl)
