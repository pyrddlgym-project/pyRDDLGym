# A tool for generating traffic instances

import numpy as np

def generate_NEMA8_phasing_data(center, north, east, south, west,
                                minor_min, minor_max, minor_red,
                                major_min, major_max, major_red,
                                p0, right_on_red=True):
    """ Generates the non-fluents for a four-leg NEMA8 intersection.

        center, north, east, south, west (string):
            Intersection ids
        minor_min, minor_max, minor_red,
        major_min, major_max, major_red (int):
            Phase properties
            Typically, minor phases are the ones that have protected left turns
               (phases 0,1,2 and 4,5,6)
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
        f'\n        GREEN({north},{center},{east},p{p0}) = true;',
        f'\n        GREEN({south},{center},{west},p{p0}) = true;',
        f'\n        GREEN({north},{center},{east},p{p0+1}) = true;',
        f'\n        GREEN({north},{center},{south},p{p0+1}) = true;',
        f'\n        GREEN({south},{center},{north},p{p0+2}) = true;',
        f'\n        GREEN({south},{center},{west},p{p0+2}) = true;',
        f'\n        GREEN({north},{center},{south},p{p0+3}) = true;',
        f'\n        GREEN({south},{center},{north},p{p0+3}) = true;',
        f'\n        GREEN({west},{center},{north},p{p0+4}) = true;',
        f'\n        GREEN({east},{center},{south},p{p0+4}) = true;',
        f'\n        GREEN({east},{center},{west},p{p0+5}) = true;',
        f'\n        GREEN({east},{center},{south},p{p0+5}) = true;',
        f'\n        GREEN({west},{center},{north},p{p0+6}) = true;',
        f'\n        GREEN({west},{center},{east},p{p0+6}) = true;',
        f'\n        GREEN({east},{center},{west},p{p0+7}) = true;',
        f'\n        GREEN({west},{center},{east},p{p0+7}) = true;'))

    if right_on_red:
        nonfluents_str += ''.join(('',
            f'\n        GREEN({north},{center},{west},p{p0})=true; GREEN({north},{center},{west},p{p0+1})=true;',
            f'\n        GREEN({north},{center},{west},p{p0+2})=true; GREEN({north},{center},{west},p{p0+3})=true;',
            f'\n        GREEN({north},{center},{west},p{p0+4})=true; GREEN({north},{center},{west},p{p0+5})=true;',
            f'\n        GREEN({north},{center},{west},p{p0+6})=true; GREEN({north},{center},{west},p{p0+7})=true;',

            f'\n        GREEN({west},{center},{south},p{p0})=true; GREEN({west},{center},{south},p{p0+1})=true;',
            f'\n        GREEN({west},{center},{south},p{p0+2})=true; GREEN({west},{center},{south},p{p0+3})=true;',
            f'\n        GREEN({west},{center},{south},p{p0+4})=true; GREEN({west},{center},{south},p{p0+5})=true;',
            f'\n        GREEN({west},{center},{south},p{p0+6})=true; GREEN({west},{center},{south},p{p0+7})=true;',

            f'\n        GREEN({south},{center},{east},p{p0})=true; GREEN({south},{center},{east},p{p0+1})=true;',
            f'\n        GREEN({south},{center},{east},p{p0+2})=true; GREEN({south},{center},{east},p{p0+3})=true;',
            f'\n        GREEN({south},{center},{east},p{p0+4})=true; GREEN({south},{center},{east},p{p0+5})=true;',
            f'\n        GREEN({south},{center},{east},p{p0+6})=true; GREEN({south},{center},{east},p{p0+7})=true;',

            f'\n        GREEN({east},{center},{north},p{p0})=true; GREEN({east},{center},{north},p{p0+1})=true;',
            f'\n        GREEN({east},{center},{north},p{p0+2})=true; GREEN({east},{center},{north},p{p0+3})=true;',
            f'\n        GREEN({east},{center},{north},p{p0+4})=true; GREEN({east},{center},{north},p{p0+5})=true;',
            f'\n        GREEN({east},{center},{north},p{p0+6})=true; GREEN({east},{center},{north},p{p0+7})=true;'))

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
        f'\n        PHASE-MIN(p{p0}) = {minor_min};'
        f'\n        PHASE-MIN(p{p0+1}) = {minor_min};'
        f'\n        PHASE-MIN(p{p0+2}) = {minor_min};'
        f'\n        PHASE-MIN(p{p0+3}) = {major_min};'
        f'\n        PHASE-MIN(p{p0+4}) = {minor_min};'
        f'\n        PHASE-MIN(p{p0+5}) = {minor_min};'
        f'\n        PHASE-MIN(p{p0+6}) = {minor_min};'
        f'\n        PHASE-MIN(p{p0+7}) = {major_min};'))

    nonfluents_str += ''.join(('',
        f'\n        PHASE-MAX(p{p0}) = {minor_max};'
        f'\n        PHASE-MAX(p{p0+1}) = {minor_max};'
        f'\n        PHASE-MAX(p{p0+2}) = {minor_max};'
        f'\n        PHASE-MAX(p{p0+3}) = {major_max};'
        f'\n        PHASE-MAX(p{p0+4}) = {minor_max};'
        f'\n        PHASE-MAX(p{p0+5}) = {minor_max};'
        f'\n        PHASE-MAX(p{p0+6}) = {minor_max};'
        f'\n        PHASE-MAX(p{p0+7}) = {major_max};'))

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

def generate_FIXED4_phasing_data(center, north, east, south, west,
                                 left_min, left_max, left_red,
                                 through_min, through_max, through_red,
                                 p0, right_on_red=True):
    """ Generates the non-fluents for a four-leg FIXED4 intersection.

        This is a fixed-order four-phase phasing scheme, with alternating
        protected left turns and through movements.

        center, north, east, south, west (string):
            Intersection ids
        left_min, left_max, left_red,
        through_min, through_max, through_red (int):
            Phase properties
        p0 (int):
            The index of the initial phase for the traffic light
            Phases with indices p0, p0+1, p0+2, p0+3 will be used
        right_on_red (bool):
            Whether right turns are permitted on red
    """
    nonfluents_str = f'\n        // FIXED4 scheme for intersection {center},{north},{east},{south},{west}'
    nonfluents_str += f'\n        NUM-ACTION-TOKENS({center}) = 2;'
    nonfluents_str += '\n        '.join( ('',) +
        tuple(f'PHASE-INDEX(p{p0+i}) = {i};'
             for i in range(4) ))

    nonfluents_str += '\n        '.join( ('',) +
        tuple(f'PHASE-OF(p{p0+i},{center}) = true;'
              for i in range(4) ))

    nonfluents_str += ''.join(('',
        f'\n        GREEN({north},{center},{east},p{p0}) = true;',
        f'\n        GREEN({south},{center},{west},p{p0}) = true;',
        f'\n        GREEN({north},{center},{south},p{p0+1}) = true;',
        f'\n        GREEN({south},{center},{north},p{p0+1}) = true;',
        f'\n        GREEN({west},{center},{north},p{p0+2}) = true;',
        f'\n        GREEN({east},{center},{south},p{p0+2}) = true;',
        f'\n        GREEN({east},{center},{west},p{p0+3}) = true;',
        f'\n        GREEN({west},{center},{east},p{p0+3}) = true;'))

    if right_on_red:
        nonfluents_str += ''.join(('',
            f'\n        GREEN({north},{center},{west},p{p0})=true; GREEN({north},{center},{west},p{p0+1})=true;',
            f'\n        GREEN({north},{center},{west},p{p0+2})=true; GREEN({north},{center},{west},p{p0+3})=true;',

            f'\n        GREEN({west},{center},{south},p{p0})=true; GREEN({west},{center},{south},p{p0+1})=true;',
            f'\n        GREEN({west},{center},{south},p{p0+2})=true; GREEN({west},{center},{south},p{p0+3})=true;',

            f'\n        GREEN({south},{center},{east},p{p0})=true; GREEN({south},{center},{east},p{p0+1})=true;',
            f'\n        GREEN({south},{center},{east},p{p0+2})=true; GREEN({south},{center},{east},p{p0+3})=true;',

            f'\n        GREEN({east},{center},{north},p{p0})=true; GREEN({east},{center},{north},p{p0+1})=true;',
            f'\n        GREEN({east},{center},{north},p{p0+2})=true; GREEN({east},{center},{north},p{p0+3})=true;'))

    nonfluents_str += ''.join(('',
        f'\n        TRANSITION(p{p0},a0) = 0;',
        f'\n        TRANSITION(p{p0},a1) = 1;',
        f'\n        TRANSITION(p{p0+1},a0) = 1;',
        f'\n        TRANSITION(p{p0+1},a1) = 2;',
        f'\n        TRANSITION(p{p0+2},a0) = 2;',
        f'\n        TRANSITION(p{p0+2},a1) = 3;',
        f'\n        TRANSITION(p{p0+3},a0) = 3;',
        f'\n        TRANSITION(p{p0+3},a1) = 0;'))

    nonfluents_str += ''.join(('',
        f'\n        PHASE-MIN(p{p0}) = {left_min};'
        f'\n        PHASE-MIN(p{p0+1}) = {through_min};'
        f'\n        PHASE-MIN(p{p0+2}) = {left_min};'
        f'\n        PHASE-MIN(p{p0+3}) = {through_min};'))

    nonfluents_str += ''.join(('',
        f'\n        PHASE-MAX(p{p0}) = {left_max};'
        f'\n        PHASE-MAX(p{p0+1}) = {through_max};'
        f'\n        PHASE-MAX(p{p0+2}) = {left_max};'
        f'\n        PHASE-MAX(p{p0+3}) = {through_max};'))

    nonfluents_str += ''.join(('',
        f'\n        PHASE-ALL-RED-DUR(p{p0}) = {left_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+1}) = {through_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+2}) = {left_red};'
        f'\n        PHASE-ALL-RED-DUR(p{p0+3}) = {through_red};'))
    nonfluents_str += f'\n        //DONE FIXED4 scheme for intersection {center},{north},{east},{south},{west}\n'

    return nonfluents_str



def generate_grid(nrows,
                  ncols,
                  phasing_scheme='NEMA8',
                  ew_link_len=(200,50),
                  ns_link_len=(200,50),
                  feeder_link_elongation_factor=1.5,
                  V=13.8,
                  inflow_rate_per_lane=(0.1,0.05),
                  satflow_per_lane=0.53,
                  num_lanes=4,
                  high_left_prob=0,
                  min_green=6,
                  max_green=60,
                  all_red=4,
                  instance_name=None,
                  horizon=200,
                  discount=1.0):
    """ Generates a grid network.

        The inflow rates are sampled from a uniform random distribution,
        and so are the link lengths. The feeder links can be elongated
        to fit more vehicles and provide more information to the boundary
        lights.

        Typically, through movements are assumed to get (2/4) of the lanes,
        and left and right turns 1/4 each. The saturation flow rate for a
        movement is obtained by multiplying the sat. flow rate per lane by
        the assumed number of lanes.

        There is a fixed probability for a left-turn to have higher demand than the
        through movement (defaults to 0). In this case, the left turns are assumed
        to have (2/4) of the lanes and the through movements (1/4) of the lanes.
    """
    # Sample link lengths uniformly
    # Optionally, make the feeder links longer to fit more vehicles
    feeder_elongation_ew, feeder_elongation_ns = np.ones(ncols+1), np.ones(nrows+1)
    feeder_elongation_ew[0] = feeder_elongation_ew[-1] = \
        feeder_elongation_ns[0] = feeder_elongation_ns[-1] = feeder_link_elongation_factor

    ew_lens = np.random.uniform(ew_link_len[0]-ew_link_len[1], ew_link_len[0]+ew_link_len[1], ncols+1)
    ns_lens = np.random.uniform(ns_link_len[0]-ns_link_len[1], ns_link_len[0]+ns_link_len[1], nrows+1)

    ew_lens *= feeder_elongation_ew
    ns_lens *= feeder_elongation_ns
    max_len = np.max(np.concatenate((ew_lens, ns_lens)))

    # Derive the X and Y coordinates of the intersections
    Xs, Ys = np.zeros(ncols+2), np.zeros(nrows+2)
    Xs[1:] = np.cumsum(ew_lens)
    Ys[1:] = np.cumsum(ns_lens)
    Ys = np.flip(Ys) # Want Ys to decrease with increasing i to be consistent with
                     # cartesian coords
    Xs = np.tile(Xs, (nrows+2,1))
    Ys = np.tile(Ys[np.newaxis,:].transpose(), (1,ncols+2))
    coords = np.concatenate((Xs[:,:,np.newaxis],
                             Ys[:,:,np.newaxis]),
                             axis=2)
    coords = np.round(coords)


    num_intersections = (nrows+2)*(ncols+2) - 4
    num_bdry = 2*(nrows + ncols) #Intersections on the boundary (Sources/sinks)
    num_tls = num_intersections - num_bdry
    num_ts = int(np.ceil(max_len/V))+1

    if phasing_scheme == 'NEMA8':
        generate_phasing_fn = generate_NEMA8_phasing_data
        phases_per_light = 8
        num_actions = 5
    elif phasing_scheme == 'FIXED4':
        generate_phasing_fn = generate_FIXED4_phasing_data
        phases_per_light = 4
        num_actions = 2

    num_phases = num_tls * phases_per_light


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
    left_turns, through_turns, right_turns = [], [], []

    for i in range(1,nrows+1):
        for j in range(1,ncols+1):
            link_pairs.extend([
                f'{inames[i,j]},{inames[i-1,j]}',
                f'{inames[i-1,j]},{inames[i,j]}',
                f'{inames[i,j]},{inames[i,j+1]}',
                f'{inames[i,j+1]},{inames[i,j]}',
                f'{inames[i,j]},{inames[i+1,j]}',
                f'{inames[i+1,j]},{inames[i,j]}',
                f'{inames[i,j]},{inames[i,j-1]}',
                f'{inames[i,j-1]},{inames[i,j]}',
            ])

            left_turns.extend([
                f'{inames[i,j-1]},{inames[i,j]},{inames[i-1,j]}',
                f'{inames[i-1,j]},{inames[i,j]},{inames[i,j+1]}',
                f'{inames[i,j+1]},{inames[i,j]},{inames[i+1,j]}',
                f'{inames[i+1,j]},{inames[i,j]},{inames[i,j-1]}',
            ])
            through_turns.extend([
                f'{inames[i,j-1]},{inames[i,j]},{inames[i,j+1]}',
                f'{inames[i,j+1]},{inames[i,j]},{inames[i,j-1]}',
                f'{inames[i-1,j]},{inames[i,j]},{inames[i+1,j]}',
                f'{inames[i+1,j]},{inames[i,j]},{inames[i-1,j]}',
            ])
            right_turns.extend([
                f'{inames[i,j-1]},{inames[i,j]},{inames[i+1,j]}',
                f'{inames[i+1,j]},{inames[i,j]},{inames[i,j+1]}',
                f'{inames[i,j+1]},{inames[i,j]},{inames[i-1,j]}',
                f'{inames[i-1,j]},{inames[i,j]},{inames[i,j-1]}',
            ])


    # Optionally, make some left turns have heavier demand than the
    # through movements
    high_left_turn = np.random.binomial(1, high_left_prob, size=len(left_turns))
    deltas = np.random.uniform(-0.1, 0.1, size=len(left_turns))

    satflow_rates, turn_probs = {}, {}
    for L, T, R, dp, high_left in zip(left_turns, through_turns, right_turns, deltas, high_left_turn):
        if high_left:
            high_turn, low_turn = L, T
        else:
            high_turn, low_turn = T, L

        turn_probs[high_turn] = 0.5 - dp
        turn_probs[low_turn] = 0.25 + dp
        turn_probs[R] = 1-turn_probs[high_turn]-turn_probs[low_turn]
        total_satflow = satflow_per_lane * num_lanes
        satflow_rates[high_turn] = 0.5 * total_satflow
        satflow_rates[low_turn] = 0.25 * total_satflow
        satflow_rates[R] = 0.25 * total_satflow

    inflow_lb = (inflow_rate_per_lane[0]-inflow_rate_per_lane[1]) * num_lanes
    inflow_ub = (inflow_rate_per_lane[0]+inflow_rate_per_lane[1]) * num_lanes
    arrival_rates = np.round( np.random.uniform(inflow_lb, inflow_ub, num_bdry),
                              2)


    if instance_name is None:
        instance_name = f'grid_{nrows}x{ncols}_{phasing_scheme}'

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

    if V != 13.8:
        instance_str += '\n        '.join(('', '// speeds', ''))
        instance_str += '\n        '.join((f'SPEED({link_pair}) = {V};' for link_pair in link_pairs))
        instance_str += '\n'

    if num_lanes != 1:
        instance_str += '\n        '.join(('', '// number of lanes', ''))
        instance_str += '\n        '.join((f'Nl({link_pair}) = {num_lanes};' for link_pair in link_pairs))
        instance_str += '\n'

    instance_str += '\n        '.join(('', '// satflow rates', ''))
    instance_str += '\n        '.join((f'MU({k}) = {v};' for k,v in satflow_rates.items()))
    instance_str += '\n'

    instance_str += '\n        '.join(('', '// turn probs', ''))
    instance_str += '\n        '.join((f'BETA({k}) = {v};' for k,v in turn_probs.items()))
    instance_str += '\n'

    phase_counter = 0
    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
            instance_str += generate_phasing_fn(
                                  inames[i,j],
                                  inames[i-1,j],
                                  inames[i,j+1],
                                  inames[i+1,j],
                                  inames[i,j-1],
                                  min_green, max_green, 2,
                                  min_green, max_green, all_red,
                                  phase_counter, right_on_red=True)
            phase_counter += phases_per_light


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
    parser.add_argument('-p', '--phasing-scheme', default='NEMA8', choices=['NEMA8', 'FIXED4'], help='The phasing scheme type to use. One of: NEMA8, FIXED4')
    parser.add_argument('-l', '--high-left-prob', default=0, help='Probability of having heavier demand on through than left from an approach')
    parser.add_argument('-n', '--instance-name', help='Name of instance')
    args = parser.parse_args()

    args.high_left_prob = float(args.high_left_prob)
    assert(0 <= args.high_left_prob <= 1)

    if os.path.isfile(args.target_path) and not args.force_overwrite:
        raise RuntimeError('[netgen.py] File with the requested path already exists. Pass a diffent path or add the -f flag to force overwrite')

    with open(args.target_path, 'w') as file:
        file.write(generate_grid(args.rows,
                                 args.cols,
                                 instance_name=args.instance_name,
                                 phasing_scheme=args.phasing_scheme,
                                 high_left_prob=args.high_left_prob))
