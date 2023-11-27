from pyRDDLGym.Examples.SumOfHalfSpaces.instances.generator import render_two_dim_objective
import os
import numpy as np
from collections import defaultdict

for path in os.listdir():
    if path.endswith('rddl'):
        with open(path, 'r') as file:
            id = path.strip('instance').rstrip('.rddl')
            Ws = np.zeros(shape=(30,2))
            Bs = np.zeros(shape=(30,2))
            for line in file.read().split('\n'):
                line = line.strip()
                if line.startswith('W'):
                    params, val = line.split(' = ')
                    s, d = params[2:-1].split(',')
                    Ws[int(s[1:]), int(d[1:])] = float(val.rstrip(';'))
                if line.startswith('B'):
                    params, val = line.split(' = ')
                    s, d = params[2:-1].split(',')
                    Bs[int(s[1:]), int(d[1:])] = float(val.rstrip(';'))
                render_two_dim_objective(Ws, Bs, save_to=f'img/img{id}.png')

