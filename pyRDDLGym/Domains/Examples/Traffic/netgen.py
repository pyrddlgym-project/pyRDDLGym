# A tool for generating traffic instances

import numpy as np

def generate_NEMA8_phasing_data(center, north, east, south, west,
                                minor_phase_min, minor_phase_max, minor_red,
                                major_phase_min, major_phase_max, major_red,
                                p0, right_on_red=True):
    """ Generates the non-fluents for a 4-leg NEMA8 intersection.

        center, north, east, south, west (string):
            Intersection ids
        minor_phase_min, minor_phase_max, minor_red,
        major_phase_min, major_phase_max, major_red (int):
            Phase properties
            Minor phases have protected left turns
        p0 (int):
            The index of the initial phase for the traffic light
            Phases with indices p0, p0+1, p0+2, ..., p0+7 will be used
        right_on_red (bool):
            Whether right turns are permitted on red
    """
    nonfluents_str = f'\n        // NEMA8 scheme for intersection {center},{north},{east},{south},{west}'
    nonfluents_str += f'\n        NUM-ACTION-TOKENS({center}) = 5;'
    nonfluents_str += '\n        '.join( ('',) +
        tuple(f'PHASE-INDEX(p{p0+i}) = {i};'
             for i in range(8) ))

    nonfluents_str += '\n        '.join( ('',) +
        tuple(f'PHASE-OF(p{p0+i},{center}) = true;'
              for i in range(8) ))

    nonfluents_str += ''.join(('',
        f'\n        GREEN({north},{center},{west},p{p0}) = true;',
        f'\n        GREEN({south},{center},{east},p{p0}) = true;',
        f'\n        GREEN({north},{center},{west},p{p0+1}) = true;',
        f'\n        GREEN({north},{center},{south},p{p0+1}) = true;',
        f'\n        GREEN({south},{center},{north},p{p0+2}) = true;',
        f'\n        GREEN({south},{center},{east},p{p0+2}) = true;',
        f'\n        GREEN({north},{center},{south},p{p0+3}) = true;',
        f'\n        GREEN({south},{center},{north},p{p0+3}) = true;',
        f'\n        GREEN({east},{center},{north},p{p0+4}) = true;',
        f'\n        GREEN({west},{center},{south},p{p0+4}) = true;',
        f'\n        GREEN({west},{center},{east},p{p0+5}) = true;',
        f'\n        GREEN({west},{center},{south},p{p0+5}) = true;',
        f'\n        GREEN({east},{center},{north},p{p0+6}) = true;',
        f'\n        GREEN({east},{center},{west},p{p0+6}) = true;',
        f'\n        GREEN({west},{center},{east},p{p0+7}) = true;',
        f'\n        GREEN({east},{center},{west},p{p0+7}) = true;'))

    if right_on_red:
        nonfluents_str += ''.join(('',
            f'\n        GREEN({north},{center},{east},p{p0})=true; GREEN({north},{center},{east},p{p0+1})=true;',
            f'\n        GREEN({north},{center},{east},p{p0+2})=true; GREEN({north},{center},{east},p{p0+3})=true;',
            f'\n        GREEN({north},{center},{east},p{p0+4})=true; GREEN({north},{center},{east},p{p0+5})=true;',
            f'\n        GREEN({north},{center},{east},p{p0+6})=true; GREEN({north},{center},{east},p{p0+7})=true;',

            f'\n        GREEN({east},{center},{south},p{p0})=true; GREEN({east},{center},{south},p{p0+1})=true;',
            f'\n        GREEN({east},{center},{south},p{p0+2})=true; GREEN({east},{center},{south},p{p0+3})=true;',
            f'\n        GREEN({east},{center},{south},p{p0+4})=true; GREEN({east},{center},{south},p{p0+5})=true;',
            f'\n        GREEN({east},{center},{south},p{p0+6})=true; GREEN({east},{center},{south},p{p0+7})=true;',

            f'\n        GREEN({south},{center},{west},p{p0})=true; GREEN({south},{center},{west},p{p0+1})=true;',
            f'\n        GREEN({south},{center},{west},p{p0+2})=true; GREEN({south},{center},{west},p{p0+3})=true;',
            f'\n        GREEN({south},{center},{west},p{p0+4})=true; GREEN({south},{center},{west},p{p0+5})=true;',
            f'\n        GREEN({south},{center},{west},p{p0+6})=true; GREEN({south},{center},{west},p{p0+7})=true;',

            f'\n        GREEN({west},{center},{north},p{p0})=true; GREEN({west},{center},{north},p{p0+1})=true;',
            f'\n        GREEN({west},{center},{north},p{p0+2})=true; GREEN({west},{center},{north},p{p0+3})=true;',
            f'\n        GREEN({west},{center},{north},p{p0+4})=true; GREEN({west},{center},{north},p{p0+5})=true;',
            f'\n        GREEN({west},{center},{north},p{p0+6})=true; GREEN({west},{center},{north},p{p0+7})=true;'))

    nonfluents_str += ''.join(('',
        f'\n        TRANSITION(p{p0},a0) = 0;',
        f'\n        TRANSITION(p{p0},a1) = 1;',
        f'\n        TRANSITION(p{p0},a2) = 2;',
        f'\n        TRANSITION(p{p0},a3) = 3;',
        f'\n        TRANSITION(p{p0+1},a0) = 1;',
        f'\n        TRANSITION(p{p0+1},a1) = 3;',
        f'\n        TRANSITION(p{p0+2},a0) = 2;',
        f'\n        TRANSITION(p{p0+2},a1) = 3;',
        f'\n        TRANSITION(p{p0+3},a0) = 3;',
        f'\n        TRANSITION(p{p0+3},a1) = 4;',
        f'\n        TRANSITION(p{p0+3},a2) = 5;',
        f'\n        TRANSITION(p{p0+3},a3) = 6;',
        f'\n        TRANSITION(p{p0+3},a4) = 7;',
        f'\n        TRANSITION(p{p0+4},a0) = 4;',
        f'\n        TRANSITION(p{p0+4},a1) = 5;',
        f'\n        TRANSITION(p{p0+4},a2) = 6;',
        f'\n        TRANSITION(p{p0+4},a3) = 7;',
        f'\n        TRANSITION(p{p0+5},a0) = 5;',
        f'\n        TRANSITION(p{p0+5},a1) = 7;',
        f'\n        TRANSITION(p{p0+6},a0) = 6;',
        f'\n        TRANSITION(p{p0+6},a1) = 7;',
        f'\n        TRANSITION(p{p0+7},a0) = 7;',
        f'\n        TRANSITION(p{p0+7},a1) = 0;',
        f'\n        TRANSITION(p{p0+7},a2) = 1;',
        f'\n        TRANSITION(p{p0+7},a3) = 2;',
        f'\n        TRANSITION(p{p0+7},a4) = 3;'))

    nonfluents_str += ''.join(('',
        f'\n        PHASE-MIN(p{p0}) = {minor_phase_min};'
        f'\n        PHASE-MIN(p{p0+1}) = {minor_phase_min};'
        f'\n        PHASE-MIN(p{p0+2}) = {minor_phase_min};'
        f'\n        PHASE-MIN(p{p0+3}) = {major_phase_min};'
        f'\n        PHASE-MIN(p{p0+4}) = {minor_phase_min};'
        f'\n        PHASE-MIN(p{p0+5}) = {minor_phase_min};'
        f'\n        PHASE-MIN(p{p0+6}) = {minor_phase_min};'
        f'\n        PHASE-MIN(p{p0+7}) = {major_phase_min};'))

    nonfluents_str += ''.join(('',
        f'\n        PHASE-MAX(p{p0}) = {minor_phase_max};'
        f'\n        PHASE-MAX(p{p0+1}) = {minor_phase_max};'
        f'\n        PHASE-MAX(p{p0+2}) = {minor_phase_max};'
        f'\n        PHASE-MAX(p{p0+3}) = {major_phase_max};'
        f'\n        PHASE-MAX(p{p0+4}) = {minor_phase_max};'
        f'\n        PHASE-MAX(p{p0+5}) = {minor_phase_max};'
        f'\n        PHASE-MAX(p{p0+6}) = {minor_phase_max};'
        f'\n        PHASE-MAX(p{p0+7}) = {major_phase_max};'))

    nonfluents_str += ''.join(('',
        f'\n        PHASE-ALL-RED-DUR(p{p0}) = {minor_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+1}) = {minor_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+2}) = {minor_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+3}) = {major_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+4}) = {minor_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+5}) = {minor_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+6}) = {minor_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+7}) = {major_red};'))
    nonfluents_str += f'\n        //DONE NEMA8 scheme for intersection {center},{north},{east},{south},{west}\n'

    return nonfluents_str


def generate_NEMA8_grid(nrows,
                        ncols,
                        ew_link_len=(200,50),
                        ns_link_len=(200,50),
                        inflow_rate=(0.2,0.1),
                        V=13.8,
                        instance_name=None,
                        horizon=200,
                        discount=1.0):
    """ ,
    """
    # Sample link lengths uniformly
    ew_lens = np.random.uniform(ew_link_len[0]-ew_link_len[1], ew_link_len[0]+ew_link_len[1], ncols+1)
    ns_lens = np.random.uniform(ns_link_len[0]-ns_link_len[1], ns_link_len[0]+ns_link_len[1], nrows+1)
    max_len = np.max(np.concatenate((ew_lens, ns_lens)))

    # Derive the X and Y coordinates of the intersections
    Xs, Ys = np.zeros(ncols+2), np.zeros(nrows+2)
    Xs[1:] = np.cumsum(ew_lens)
    Ys[1:] = np.cumsum(ns_lens)
    Xs = np.tile(Xs, (nrows+2,1))
    Ys = np.tile(Ys[np.newaxis,:].transpose(), (1,ncols+2))
    coords = np.concatenate((Xs[:,:,np.newaxis],
                             Ys[:,:,np.newaxis]),
                             axis=2)
    coords = np.round(coords)


    num_intersections = (nrows+2)*(ncols+2) - 4
    num_bdry = 2*(nrows + ncols) #Intersections on the boundary (Sources/sinks)
    num_tls = num_intersections - num_bdry
    num_phases = num_tls * 8
    num_actions = 5
    num_ts = int(np.ceil(max_len/V))+1

    bdry_names = tuple(f's{i}' for i in range(num_bdry))
    tl_names = tuple(f'i{i}' for i in range(num_tls))
    phase_names = tuple(f'p{i}' for i in range(num_phases))
    t_names = tuple(f't{i}' for i in range(num_ts))

    inames = np.array(['EMPTY' for _ in range(num_intersections+4)]).reshape((nrows+2,ncols+2))

    for i in range(nrows+2):
        for j in range(ncols+2):

            if 0 < i < nrows+1 and 0 < j < ncols+1:
                # Traffic light
                inames[i,j] = f'i{(j-1) + (i-1)*ncols}'
            else:
                # Source/sink
                if 0 < j < ncols+1:
                    if i==0:
                        inames[i,j] = f's{j-1}'
                    elif i==(nrows+1):
                        inames[i,j] = f's{2*ncols + nrows - j}'
                if 0 < i < nrows+1:
                    if j==(ncols+1):
                        inames[i,j] = f's{ncols+(i-1)}'
                    elif j==0:
                        inames[i,j] = f's{num_bdry - i}'

    link_pairs = []
    for i in range(1,nrows+1):
        for j in range(1,ncols+1):
            link_pairs.extend([
                f'{inames[i,j]},{inames[i-1,j]}',
                f'{inames[i,j]},{inames[i,j+1]}',
                f'{inames[i,j]},{inames[i+1,j]}',
                f'{inames[i,j]},{inames[i,j-1]}',
                f'{inames[i-1,j]},{inames[i,j]}',
                f'{inames[i,j+1]},{inames[i,j]}',
                f'{inames[i+1,j]},{inames[i,j]}',
                f'{inames[i,j-1]},{inames[i,j]}',
            ])

    arrival_rates = np.round( np.random.uniform(inflow_rate[0]-inflow_rate[1],
                                                inflow_rate[0]+inflow_rate[1],
                                                num_bdry),
                              2)


    if instance_name is None:
        instance_name = f'grid_{nrows}x{ncols}'

    instance_str = '\n'.join((
        f'non-fluents {instance_name}' + ' {',
         '    domain = BLX_model;',
         '',
         '    objects {',
        f'        intersection : {{{", ".join(tl_names + bdry_names)}}};',
        f'        phase        : {{{", ".join(phase_names)}}};',
         '        action-token : {a0, a1, a2, a3, a4};',
        f'        time         : {{{", ".join(t_names)}}};',
         '    };',
         '',
         '    non-fluents {',
         '        //action token enumeration',
         '        '))
    instance_str += '\n        '.join((f'ACTION-TOKEN-INDEX(a{i}) = {i};' for i in range(num_actions)))
    instance_str += '\n'

    instance_str += '\n        '.join(('', '// cartesian coordinates', ''))
    instance_str += '\n        '.join((f'X({inames[i,j]}) = {coords[i,j,0]};    Y({inames[i,j]}) = {coords[i,j,1]};'
                                      for i in range(nrows+2) for j in range(ncols+2) if inames[i,j] != 'EMPTY'))
    instance_str += '\n'

    instance_str += '\n        '.join(('', '// intersection indices') +
                                      tuple(f'INTERSECTION-INDEX(i{i}) = {i};' for i in range(num_tls)) +
                                      tuple(f'INTERSECTION-INDEX(s{i}) = {num_tls+i};' for i in range(num_bdry)))
    instance_str += '\n'


    instance_str += '\n        '.join(('', '// source intersections', ''))
    instance_str += '\n        '.join((f'SOURCE(s{i}) = true;' for i in range(num_bdry)))

    instance_str += '\n        '.join(('', '// sink intersections', ''))
    instance_str += '\n        '.join((f'SINK(s{i}) = true;' for i in range(num_bdry)))

    instance_str += '\n        '.join(('', '// traffic lights', ''))
    instance_str += '\n        '.join((f'TL(i{i}) = true;' for i in range(num_tls)))
    instance_str += '\n'

    instance_str += '\n        '.join(('', '// arrival rates', ''))
    instance_str += '\n        '.join((f'ARRIVAL-RATE(s{i}) = {arrival_rates[i]};' for i in range(num_bdry)))
    instance_str += '\n'

    instance_str += '\n        '.join(('', '// roads between intersections', ''))
    instance_str += '\n        '.join((f'LINK({link_pair}) = true;' for link_pair in link_pairs))
    instance_str += '\n'

    phase_counter = 0
    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
            instance_str += generate_NEMA8_phasing_data(
                                  inames[i,j],
                                  inames[i-1,j],
                                  inames[i,j+1],
                                  inames[i+1,j],
                                  inames[i,j-1],
                                  4, 20, 4,
                                  20, 60, 6,
                                  phase_counter, right_on_red=True)
            phase_counter += 8

    instance_str += '\n        '.join(('', '// time-delay properties',
        'TIME-HEAD(t0) = true;',
       f'TIME-TAIL(t{num_ts-1}) = true;') +
        tuple(f'TIME-VAL(t{i}) = {i};' for i in range(num_ts)) +
        tuple(f'NEXT(t{i},t{i+1}) = true;' for i in range(num_ts-1)))


    instance_str += '\n'
    instance_str += '\n'.join((
         '    };',
         '}',
         '',
        f'instance {instance_name}' + '{',
         '    domain = BLX_model;',
        f'    non-fluents = {instance_name};',
         '    max-nondef-actions = pos-inf;',
        f'    horizon = {horizon};',
        f'    discount = {discount};',
         '}' ))

    return instance_str

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Tool for automatically generating grid instances for the RDDL traffic domain')
    parser.add_argument('target_path', type=str, help='Path the generated rddl code will be saved to')
    parser.add_argument('-r', '--rows', type=int, help='Number of rows in the grid', required=True)
    parser.add_argument('-c', '--cols', type=int, help='Number of columns in the grid', required=True)
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='By default the generator will not overwrite existing files. With this flag on, it will')
    args = parser.parse_args()

    if os.path.isfile(args.target_path) and not args.force_overwrite:
        raise RuntimeError('[netgen.py] File with the requested path already exists. Pass a diffent path or add the -f flag to force overwrite')

    with open(args.target_path, 'w') as file:
        file.write(generate_NEMA8_grid(args.rows, args.cols))
