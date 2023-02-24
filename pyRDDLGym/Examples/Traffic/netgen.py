# A tool for generating traffic instances

import numpy as np
from itertools import product

indent_str = ' ' * 8
newline_indent_str = '\n' + indent_str

def dist(p0, p1): return np.linalg.norm(p1-p0)


def generate_4leg_intersection(i, Ein, Nout, Nin, Wout, Win, Sout, Sin, Eout,
                               min, max, red, right_on_red=True):
    """ Generates the non-fluents for a four-leg intersection """

    nonfluents_str = newline_indent_str*2 + f'//intersection {i}'
    nonfluents_str += newline_indent_str + '//turns' + newline_indent_str
    nonfluents_str += newline_indent_str.join((
        f'TURN({Ein},{Nout});',
        f'TURN({Ein},{Wout});',
        f'TURN({Ein},{Sout});',
        f'TURN({Nin},{Wout});',
        f'TURN({Nin},{Sout});',
        f'TURN({Nin},{Eout});',
        f'TURN({Win},{Sout});',
        f'TURN({Win},{Eout});',
        f'TURN({Win},{Nout});',
        f'TURN({Sin},{Eout});',
        f'TURN({Sin},{Nout});',
        f'TURN({Sin},{Wout});',
         '//link-to',
        f'LINK-TO({Ein},{i});',
        f'LINK-TO({Nin},{i});',
        f'LINK-TO({Win},{i});',
        f'LINK-TO({Sin},{i});',
         '//link-from',
        f'LINK-FROM({i},{Eout});',
        f'LINK-FROM({i},{Nout});',
        f'LINK-FROM({i},{Wout});',
        f'LINK-FROM({i},{Sout});',
         '//phase properties',
        f'PHASE-MIN({i}) = {min};',
        f'PHASE-MAX({i}) = {max};',
        f'PHASE-ALL-RED-DUR({i}) = {red};',
         '//green turns',
        f'GREEN({Ein},{Sout},@WEST-EAST-LEFT);',
        f'GREEN({Win},{Nout},@WEST-EAST-LEFT);',
        f'GREEN({Ein},{Wout},@WEST-EAST-THROUGH);',
        f'GREEN({Win},{Eout},@WEST-EAST-THROUGH);',
        f'GREEN({Nin},{Eout},@NORTH-SOUTH-LEFT);',
        f'GREEN({Sin},{Wout},@NORTH-SOUTH-LEFT);',
        f'GREEN({Nin},{Sout},@NORTH-SOUTH-THROUGH);',
        f'GREEN({Sin},{Nout},@NORTH-SOUTH-THROUGH);'))

    right_turn_pairs = ((Ein,Nout),
                        (Nin,Wout),
                        (Win,Sout),
                        (Sin,Eout))
    phases = ('@WEST-EAST-LEFT',
              '@WEST-EAST-THROUGH',
              '@NORTH-SOUTH-LEFT',
              '@NORTH-SOUTH-THROUGH')

    nonfluents_str += newline_indent_str
    nonfluents_str += newline_indent_str.join( (f'GREEN({i},{j},{p});'
                                                for (i,j) in right_turn_pairs for p in phases) )

    if right_on_red:
        red_phases = ('@ALL-RED',
                      '@ALL-RED2',
                      '@ALL-RED3',
                      '@ALL-RED4')
        nonfluents_str += newline_indent_str
        nonfluents_str += newline_indent_str.join( (f'GREEN({i},{j},{p});'
                                                    for (i,j) in right_turn_pairs for p in red_phases) )
    return nonfluents_str



def generate_webster_scenario(d,
                              dNSfrac,
                              dEWfrac,
                              betaL,
                              betaT,
                              link_lengths=250,
                              min_green=6,
                              max_green=60,
                              all_red=4,
                              mu=0.53,
                              instance_name=None,
                              horizon=1024,
                              discount=1.0):
    """ Generates a single intersection instance for a Webster timing experiment """
    if instance_name is None:
        instance_name = f'single_intersection_webster_experiment'

    num_ts = int(np.ceil(link_lengths/13.6))+1
    t_names = (f't{t}' for t in range(num_ts))
    instance_str = '\n'.join((
        f'// d={d}, dNS={dNSfrac:.2f}, dEW={dEWfrac:.2f}, bL={betaL:.3f}, bT={betaT:.3f}',
        f'',
        f'non-fluents {instance_name} {{',
        f'    domain = BLX_model;',
        f'',
        f'    objects {{',
        f'        intersection : {{i0}};',
        f'        link         : {{l0, l1, l2, l3, l4, l5, l6, l7}};',
        f'        time         : {{{", ".join(t_names)}}};',
        f'    }};',
        f'',
        f'    //             | |',
        f'    //             | |',
        f'    //             | |',
        f'    //            l2 l1',
        f'    //             | |',
        f'    //             v ^',
        f'    //             | |',
        f'    //             ____',
        f'    // --- l3 -<- | i0 | -<- l0 ---',
        f'    // --- l4 ->- |____| ->- l7 ---',
        f'    //             | |',
        f'    //             v ^',
        f'    //             | |',
        f'    //            l5 l6',
        f'    //             | |',
        f'    //             | |',
        f'    //             | |',
        f'',
        f'    non-fluents {{',
        f'        //cartesian coordinates',
        f'        X(i0) = 0;    Y(i0) = 0;',
        f'        SOURCE-X(l0) = {link_lengths};    SOURCE-Y(l0) = 0;',
        f'        SOURCE-X(l2) = 0;    SOURCE-Y(l2) = {link_lengths};',
        f'        SOURCE-X(l4) = -{link_lengths};   SOURCE-Y(l4) = 0;',
        f'        SOURCE-X(l6) = 0;   SOURCE-Y(l6) = -{link_lengths};',
        f'        SINK-X(l7) = {link_lengths};    SINK-Y(l7) = 0;',
        f'        SINK-X(l1) = 0;    SINK-Y(l1) = {link_lengths};',
        f'        SINK-X(l3) = -{link_lengths};   SINK-Y(l3) = 0;',
        f'        SINK-X(l5) = 0;   SINK-Y(l5) = -{link_lengths};',
        f'',
        f'        // source links',
        f'        SOURCE(l0);',
        f'        SOURCE(l2);',
        f'        SOURCE(l4);',
        f'        SOURCE(l6);',
        f'',
        f'        // sink links',
        f'        SINK(l1);',
        f'        SINK(l3);',
        f'        SINK(l5);',
        f'        SINK(l7);',
        f'',
        f'        // arrival rate from each source',
        f'        SOURCE-ARRIVAL-RATE(l0) = {d*dEWfrac/2};',
        f'        SOURCE-ARRIVAL-RATE(l2) = {d*dNSfrac/2};',
        f'        SOURCE-ARRIVAL-RATE(l4) = {d*dEWfrac/2};',
        f'        SOURCE-ARRIVAL-RATE(l6) = {d*dNSfrac/2};',
        f''
        f'        // link lengths',
        f'        Dl(l0) = {link_lengths};',
        f'        Dl(l1) = {link_lengths};',
        f'        Dl(l2) = {link_lengths};',
        f'        Dl(l3) = {link_lengths};',
        f'        Dl(l4) = {link_lengths};',
        f'        Dl(l5) = {link_lengths};',
        f'        Dl(l6) = {link_lengths};',
        f'        Dl(l7) = {link_lengths};',
        f'',
        f'        // satflow rates',
        f'        MU(l0,l3) = {2*mu};',
        f'        MU(l2,l5) = {2*mu};',
        f'        MU(l4,l7) = {2*mu};',
        f'        MU(l6,l1) = {2*mu};',
        f'        MU(l0,l5) = {mu};',
        f'        MU(l2,l7) = {mu};',
        f'        MU(l4,l1) = {mu};',
        f'        MU(l6,l3) = {mu};',
        f'',
        f'        // turn probabilities',
        f'        BETA(l0,l3) = {betaT};',
        f'        BETA(l2,l5) = {betaT};',
        f'        BETA(l4,l7) = {betaT};',
        f'        BETA(l6,l1) = {betaT};',
        f'        BETA(l0,l5) = {betaL};',
        f'        BETA(l2,l7) = {betaL};',
        f'        BETA(l4,l1) = {betaL};',
        f'        BETA(l6,l3) = {betaL};',
        f''))

    instance_str += generate_4leg_intersection('i0', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7',
                               min=min_green, max=max_green, red=all_red, right_on_red=True)


    instance_str += '\n        '.join(('', '// time-delay properties',
       f'TIME-HEAD(t0);',
       f'TIME-TAIL(t{num_ts-1});') +
        tuple(f'TIME-VAL(t{i}) = {i};' for i in range(num_ts)) +
        tuple(f'NEXT(t{i},t{i+1});' for i in range(num_ts-1)))

    instance_str += '\n'
    instance_str += '\n'.join((
        f'    }};',
        f'}}',
        f'',
        f'instance {instance_name} {{',
        f'    domain = BLX_model;',
        f'    non-fluents = {instance_name};',
        f'    max-nondef-actions = 1;',
        f'    horizon = {horizon};',
        f'    discount = {discount};',
        f'}}' ))
    return instance_str



def generate_grid(nrows,
                  ncols,
                  ew_link_len=(200,50), #(a,b) parsed as Uniform(a-b,a+b)
                  ns_link_len=(200,50),
                  feeder_link_elongation_factor=1.5,
                  Vl=13.8,
                  inflow_rate_per_lane=(0.08,0.02),
                  satflow_per_lane=0.53,
                  num_lanes=4,
                  high_left_prob=0,
                  min_green=7,
                  max_green=60,
                  all_red=4,
                  right_on_red=True,
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

    # Derive the X and Y coordinates of the intersections, sinks and sources
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


    num_intersections = nrows*ncols
    num_bdry = 2*(nrows + ncols)
    N = num_intersections + num_bdry
    num_ts = int(np.ceil(max_len/Vl))+2


    intersection_names = tuple(f'i{i}' for i in range(num_intersections))
    t_names = tuple(f't{i}' for i in range(num_ts))


    inames = np.array(['EMPTY' for _ in range(N+4)]).reshape((nrows+2,ncols+2))

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


    link_names = []
    link_lengths = []
    left_turns, through_turns, right_turns = [], [], []

    for i in range(1,nrows+1):
        for j in range(1,ncols+1):
            link_names.extend([
                f'l-{inames[i,j]}-{inames[i-1,j]}',
                f'l-{inames[i,j]}-{inames[i,j+1]}',
                f'l-{inames[i,j]}-{inames[i+1,j]}',
                f'l-{inames[i,j]}-{inames[i,j-1]}'])
            link_lengths.extend([
                dist(coords[i-1,j], coords[i,j]),
                dist(coords[i,j+1], coords[i,j]),
                dist(coords[i+1,j], coords[i,j]),
                dist(coords[i,j-1], coords[i,j])])

            left_turns.extend([
                f'l-{inames[i,j-1]}-{inames[i,j]},l-{inames[i,j]}-{inames[i-1,j]}',
                f'l-{inames[i-1,j]}-{inames[i,j]},l-{inames[i,j]}-{inames[i,j+1]}',
                f'l-{inames[i,j+1]}-{inames[i,j]},l-{inames[i,j]}-{inames[i+1,j]}',
                f'l-{inames[i+1,j]}-{inames[i,j]},l-{inames[i,j]}-{inames[i,j-1]}'])
            through_turns.extend([
                f'l-{inames[i,j-1]}-{inames[i,j]},l-{inames[i,j]}-{inames[i,j+1]}',
                f'l-{inames[i,j+1]}-{inames[i,j]},l-{inames[i,j]}-{inames[i,j-1]}',
                f'l-{inames[i-1,j]}-{inames[i,j]},l-{inames[i,j]}-{inames[i+1,j]}',
                f'l-{inames[i+1,j]}-{inames[i,j]},l-{inames[i,j]}-{inames[i-1,j]}'])
            right_turns.extend([
                f'l-{inames[i,j-1]}-{inames[i,j]},l-{inames[i,j]}-{inames[i+1,j]}',
                f'l-{inames[i+1,j]}-{inames[i,j]},l-{inames[i,j]}-{inames[i,j+1]}',
                f'l-{inames[i,j+1]}-{inames[i,j]},l-{inames[i,j]}-{inames[i-1,j]}',
                f'l-{inames[i-1,j]}-{inames[i,j]},l-{inames[i,j]}-{inames[i,j-1]}'])

    # Add missing source links
    source_link_names = []
    for j in range(1,ncols+1):
        new_links = [
            f'l-{inames[0,j]}-{inames[1,j]}',
            f'l-{inames[nrows+1,j]}-{inames[nrows,j]}']
        link_names.extend(new_links)
        source_link_names.extend(new_links)
        link_lengths.extend([
            dist(coords[1,j], coords[0,j]),
            dist(coords[nrows,j], coords[nrows+1,j]) ])

    for i in range(1,nrows+1):
        new_links = [
            f'l-{inames[i,0]}-{inames[i,1]}',
            f'l-{inames[i,ncols+1]}-{inames[i,ncols]}']
        link_names.extend(new_links)
        source_link_names.extend(new_links)
        link_lengths.extend([
            dist(coords[i,1], coords[i,0]),
            dist(coords[i,ncols], coords[i,ncols+1]) ])

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

        turn_probs[high_turn] = 0.7 - dp
        turn_probs[low_turn] = 0.2 + dp
        turn_probs[R] = 1-turn_probs[high_turn]-turn_probs[low_turn]
        total_satflow = satflow_per_lane * num_lanes
        satflow_rates[high_turn] = 0.5 * total_satflow
        satflow_rates[low_turn] = 0.3 * total_satflow
        satflow_rates[R] = 0.2 * total_satflow

    inflow_lb = (inflow_rate_per_lane[0]-inflow_rate_per_lane[1]) * num_lanes
    inflow_ub = (inflow_rate_per_lane[0]+inflow_rate_per_lane[1]) * num_lanes
    arrival_rates = np.round( np.random.uniform(inflow_lb, inflow_ub, num_bdry),
                              2)

    if instance_name is None:
        instance_name = f'grid_{nrows}x{ncols}'


    instance_str = '\n'.join((
        f'non-fluents {instance_name} {{',
        f'    domain = BLX_model;',
        f'',
        f'    objects {{',
        f'        intersection : {{{", ".join(intersection_names)}}};',
        f'        link         : {{{", ".join(link_names)}}};',
        f'        time         : {{{", ".join(t_names)}}};',
        f'    }};',
        f'',
        f'    non-fluents {{'))

    instance_str += newline_indent_str + '//sources'
    instance_str += newline_indent_str.join(('',)
        + tuple(f'SOURCE(l-{inames[i,0]}-{inames[i,1]});'
                + newline_indent_str
                + f'SOURCE(l-{inames[i,ncols+1]}-{inames[i,ncols]});'
                for i in range(1, nrows+1))
        + tuple(f'SOURCE(l-{inames[0,j]}-{inames[1,j]});'
                + newline_indent_str
                + f'SOURCE(l-{inames[nrows+1,j]}-{inames[nrows,j]});'
                for j in range(1, ncols+1)))

    instance_str += newline_indent_str + '//sinks'
    instance_str += newline_indent_str.join(('',)
        + tuple(f'SINK(l-{inames[i,1]}-{inames[i,0]});'
                + newline_indent_str
                + f'SINK(l-{inames[i,ncols]}-{inames[i,ncols+1]});'
                for i in range(1, nrows+1))
        + tuple(f'SINK(l-{inames[1,j]}-{inames[0,j]});'
                + newline_indent_str
                + f'SINK(l-{inames[nrows,j]}-{inames[nrows+1,j]});'
                for j in range(1, ncols+1)))


    if Vl != 13.8:
        instance_str += newline_indent_str + '//speeds'
        instance_str += newline_indent_str.join(('',) + tuple(f'SPEED({link}) = {Vl};' for link in link_names))

    if num_lanes != 4:
        instance_str += newline_indent_str + '//number of lanes'
        instance_str += newline_indent_str.join(('',) + tuple(f'Nl({link}) = {num_lanes};' for link in link_names))

    instance_str += newline_indent_str + '//satflow rates'
    instance_str += newline_indent_str.join(('',) + tuple(f'MU({k}) = {v};' for k,v in satflow_rates.items()))

    instance_str += newline_indent_str + '//turn probabilities'
    instance_str += newline_indent_str.join(('',) + tuple(f'BETA({k}) = {v};' for k,v in turn_probs.items()))

    instance_str += newline_indent_str + '//link lengths'
    instance_str += newline_indent_str.join(('',) + tuple(f'Dl({k}) = {v};' for k,v in zip(link_names, link_lengths)))

    instance_str += newline_indent_str + '//source arrival rates'
    instance_str += newline_indent_str.join(('',) + tuple(f'SOURCE-ARRIVAL-RATE({k}) = {v};' for k,v in zip(source_link_names, arrival_rates)))

    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
            instance_str += generate_4leg_intersection(
                                inames[i,j],
                                f'l-{inames[i,j+1]}-{inames[i,j]}', #Ein
                                f'l-{inames[i,j]}-{inames[i-1,j]}', #Nout
                                f'l-{inames[i-1,j]}-{inames[i,j]}', #Nin
                                f'l-{inames[i,j]}-{inames[i,j-1]}', #Wout
                                f'l-{inames[i,j-1]}-{inames[i,j]}', #Win
                                f'l-{inames[i,j]}-{inames[i+1,j]}', #Sout
                                f'l-{inames[i+1,j]}-{inames[i,j]}', #Sin
                                f'l-{inames[i,j]}-{inames[i,j+1]}', #Eout
                                min=min_green,
                                max=max_green,
                                red=all_red,
                                right_on_red=right_on_red)


    instance_str += '\n        '.join(('', '// time-delay properties',
       f'TIME-HEAD(t0);',
       f'TIME-TAIL(t{num_ts-1});') +
        tuple(f'TIME-VAL(t{i}) = {i};' for i in range(num_ts)) +
        tuple(f'NEXT(t{i},t{i+1});' for i in range(num_ts-1)))


    instance_str += newline_indent_str + '//cartesian coordinates (for visualization)'
    instance_str += newline_indent_str.join(('',) + tuple(
        f'X({inames[i,j]}) = {coords[i,j,0]}; Y({inames[i,j]}) = {coords[i,j,1]};'
        for i in range(1,nrows+1) for j in range(1,ncols+1) ))

    instance_str += newline_indent_str + newline_indent_str.join((
        f'SOURCE-X(l-{inames[i,0]}-{inames[i,1]}) = {coords[i,0,0]}; SOURCE-Y(l-{inames[i,0]}-{inames[i,1]}) = {coords[i,0,1]};'
        + newline_indent_str
        + f'SOURCE-X(l-{inames[i,ncols+1]}-{inames[i,ncols]}) = {coords[i,ncols+1,0]}; SOURCE-Y(l-{inames[i,ncols+1]}-{inames[i,ncols]}) = {coords[i,ncols+1,1]};'
        + newline_indent_str
        + f'SINK-X(l-{inames[i,1]}-{inames[i,0]}) = {coords[i,0,0]}; SINK-Y(l-{inames[i,1]}-{inames[i,0]}) = {coords[i,0,1]};'
        + newline_indent_str
        + f'SINK-X(l-{inames[i,ncols]}-{inames[i,ncols+1]}) = {coords[i,ncols+1,0]}; SINK-Y(l-{inames[i,ncols]}-{inames[i,ncols+1]}) = {coords[i,ncols+1,1]};'
        for i in range(1,nrows+1) ))

    instance_str += newline_indent_str + newline_indent_str.join((
        f'SOURCE-X(l-{inames[0,j]}-{inames[1,j]}) = {coords[0,j,0]}; SOURCE-Y(l-{inames[0,j]}-{inames[1,j]}) = {coords[0,j,1]};'
        + newline_indent_str
        + f'SOURCE-X(l-{inames[nrows+1,j]}-{inames[nrows,j]}) = {coords[nrows+1,j,0]}; SOURCE-Y(l-{inames[nrows+1,j]}-{inames[nrows,j]}) = {coords[nrows+1,j,1]};'
        + newline_indent_str
        + f'SINK-X(l-{inames[1,j]}-{inames[0,j]}) = {coords[0,j,0]}; SINK-Y(l-{inames[1,j]}-{inames[0,j]}) = {coords[0,j,1]};'
        + newline_indent_str
        + f'SINK-X(l-{inames[nrows,j]}-{inames[nrows+1,j]}) = {coords[nrows+1,j,0]}; SINK-Y(l-{inames[nrows,j]}-{inames[nrows+1,j]}) = {coords[nrows+1,j,1]};'
        for j in range(1,ncols+1) ))


    instance_str += '\n'
    instance_str += '\n'.join((
        f'    }};',
        f'}}',
        f'',
        f'instance {instance_name} {{',
        f'    domain = BLX_model;',
        f'    non-fluents = {instance_name};',
        f'    max-nondef-actions = {num_intersections};',
        f'    horizon = {horizon};',
        f'    discount = {discount};',
        f'}}' ))

    return instance_str





if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Tool for automatically generating grid instances for the RDDL traffic domain')
    parser.add_argument('target_path', type=str, help='Path the generated rddl code will be saved to')
    parser.add_argument('-r', '--rows', type=int, help='Number of rows in the network', required=True)
    parser.add_argument('-c', '--cols', type=int, help='Number of columns in the network', required=True)
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='By default the generator will not overwrite existing files. With this argument, it will')
    parser.add_argument('-l', '--high-left-prob', default=0, help='Probability of having heavier demand on through than left from an approach')
    parser.add_argument('-n', '--instance-name', help='Name of instance')
    args = parser.parse_args()


    args.high_left_prob = float(args.high_left_prob)
    assert(0 <= args.high_left_prob <= 1)

    if os.path.isfile(args.target_path) and not args.force_overwrite:
        raise RuntimeError('[netgen.py] File with the requested path already exists. Pass a diffent path or add the -f argument to force overwrite')

    with open(args.target_path, 'w') as file:
        network = generate_grid(
            args.rows, args.cols,
            instance_name=args.instance_name,
            high_left_prob=args.high_left_prob)

        file.write(network)
