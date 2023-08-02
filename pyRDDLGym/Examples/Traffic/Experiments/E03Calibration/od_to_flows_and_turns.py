import numpy as np
import jax.numpy as jnp
import networkx as nx

def prepare_shortest_path_assignment(myEnv):
    model = myEnv.model
    init_state_subs = myEnv.sampler.subs

    intersections = model.objects['intersection']
    links = model.objects['link']
    link_from = init_state_subs['LINK-FROM'].T
    link_to = init_state_subs['LINK-TO']
    link_len = init_state_subs['Dl']
    link_turns = init_state_subs['TURN']
    sink = init_state_subs['SINK']
    source = init_state_subs['SOURCE']

    DG = nx.DiGraph()
    DG.add_nodes_from(intersections)

    srcs, sinks = [], []
    for Li, (L, fr, to, len_, sink, src) in enumerate(zip(links, link_from, link_to, link_len, sink, source)):
        if sink:
            DG.add_node(L)
            fr_idx = fr.nonzero()[0][0]
            DG.add_edge(f'i{fr_idx}', L, idx=Li, len=len_)
            sinks.append(L)
        elif src:
            DG.add_node(L)
            to_idx = to.nonzero()[0][0]
            DG.add_edge(L, f'i{to_idx}', idx=Li, len=len_)
            srcs.append(L)
        else:
            fr_idx = fr.nonzero()[0][0]
            to_idx = to.nonzero()[0][0]
            DG.add_edge(f'i{fr_idx}', f'i{to_idx}', idx=Li, len=len_)


    def parse_as_edge_path(path, link_turns):
        edge_path = tuple(DG[v0][v1]['idx'] for v0, v1 in zip(path, path[1:]))
        for ei0, ei1 in zip(edge_path, edge_path[1:]):
            if not link_turns[ei0, ei1]:
                # A path in the graph may not be valid if, for example, it contains
                # disallowed U-turns
                return None
        return edge_path

    turn_flow_matrix = np.zeros(shape=(len(srcs), len(sinks), len(links), len(links)))
    for srci, src in enumerate(srcs):
        for snki, snk in enumerate(sinks):
            valid_edge_paths = []
            for path in nx.all_shortest_paths(DG, src, snk, weight='len'):
                edge_path = parse_as_edge_path(path, link_turns)
                if edge_path is not None:
                    valid_edge_paths.append(edge_path)

            if len(valid_edge_paths) == 0:
                continue

            alpha = 1/len(valid_edge_paths)
            for edge_path in valid_edge_paths:
                for e0i, e1i in zip(edge_path, edge_path[1:]):
                    turn_flow_matrix[srci, snki, e0i, e1i] += alpha

    return srcs, sinks, turn_flow_matrix

def convert_od_to_flows_and_turn_props(od, turn_flow_matrix):
    inflows = np.sum(od, axis=1)
    turn_flows = np.tensordot(od, turn_flow_matrix, axes=([0,1],[0,1]))
    turn_props = turn_flows / (np.sum(turn_flows, axis=1) + 1e-8)[:, np.newaxis]
    return inflows, turn_props

def convert_od_to_flows_and_turn_props_jax(od, turn_flow_matrix):
    inflows = jnp.sum(od, axis=1)
    turn_flows = jnp.tensordot(od, turn_flow_matrix, axes=([0,1],[0,1]))
    turn_props = turn_flows / (jnp.sum(turn_flows, axis=0) + 1e-8)[:, jnp.newaxis]
    return inflows, turn_props

if __name__ == '__main__':
    from pyRDDLGym import ExampleManager
    from pyRDDLGym import RDDLEnv

    np.set_printoptions(
        linewidth=9999)

    # specify the model
    EnvInfo = ExampleManager.GetEnvInfo('traffic4phase')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                        instance='instances/instance_2x2.rddl')


    srcs, sinks, turn_flow_matrix = prepare_shortest_path_assignment(myEnv)

    # Tests
    # The network is a 2x2 grid with link labels organized by intersection
    # and source node. Unfortunately the order of sinks and sources does not
    # match the naming

    print(srcs)
    print(sinks)

    def run_test(target_inflows, target_turn_props, inflows, turn_props, test_name):
        if not np.isclose(target_inflows, inflows).all(): raise AssertionError(f'{test_name} inflows failed to match')
        elif not np.isclose(target_turn_props, turn_props).all(): raise AssertionError(f'{test_name} turning props failed to match')
        else: print(f'{test_name} PASSED')

    # Test 1. Simple straight flow
    od = np.zeros(shape=(len(srcs), len(sinks)))
    od[0,4] = 0.3

    inflows, turn_props = convert_od_to_flows_and_turn_props(od, turn_flow_matrix)

    test1_inflows = np.array([0.3, 0., 0., 0., 0., 0., 0., 0.])
    test1_turn_props = np.zeros(shape=(24,24))
    test1_turn_props[16,2] = 1.
    test1_turn_props[2,10] = 1.
    run_test(test1_inflows, test1_turn_props, inflows, turn_props, 'Test 1')

    # Test 2. Branch into three
    od = np.zeros(shape=(len(srcs), len(sinks)))
    od[0,4] = 0.2
    od[0,1] = 0.2
    od[0,3] = 0.2

    inflows, turn_props = convert_od_to_flows_and_turn_props(od, turn_flow_matrix)

    test2_inflows = np.array([0.6, 0., 0., 0., 0., 0., 0., 0.])
    test2_turn_props = np.zeros(shape=(24,24))
    test2_turn_props[16,2] = 1/3
    test2_turn_props[16,3] = 1/3
    test2_turn_props[16,1] = 1/3
    test2_turn_props[2,10] = 1.
    test2_turn_props[1,5] = 1.
    run_test(test2_inflows, test2_turn_props, inflows, turn_props, 'Test 2')

    # Test 3. Branch into three with unequal flows
    od = np.zeros(shape=(len(srcs), len(sinks)))
    od[0,4] = 0.2
    od[0,1] = 0.1
    od[0,3] = 0.3

    inflows, turn_props = convert_od_to_flows_and_turn_props(od, turn_flow_matrix)

    test3_inflows = np.array([0.6, 0., 0., 0., 0., 0., 0., 0.])
    test3_turn_props = np.zeros(shape=(24,24))
    test3_turn_props[16,2] = 1/3
    test3_turn_props[16,3] = 1/6
    test3_turn_props[16,1] = 1/2
    test3_turn_props[2,10] = 1.
    test3_turn_props[1,5] = 1.
    run_test(test3_inflows, test3_turn_props, inflows, turn_props, 'Test 3')

    # Test 4. Branching at several nodes
    od = np.zeros(shape=(len(srcs), len(sinks)))
    od[0,4] = 0.2
    od[0,1] = 0.1
    od[0,3] = 0.15
    od[0,2] = 0.15

    inflows, turn_props = convert_od_to_flows_and_turn_props(od, turn_flow_matrix)

    test4_inflows = np.array([0.6, 0., 0., 0., 0., 0., 0., 0.])
    test4_turn_props = np.zeros(shape=(24,24))
    test4_turn_props[16,2] = 1/3
    test4_turn_props[16,3] = 1/6
    test4_turn_props[16,1] = 1/2
    test4_turn_props[2,10] = 1.
    test4_turn_props[1,5] = 1/2
    test4_turn_props[1,4] = 1/2
    run_test(test4_inflows, test4_turn_props, inflows, turn_props, 'Test 4')


    # Test 5. Several sources
    od = np.zeros(shape=(len(srcs), len(sinks)))
    od[0,4] = 0.2
    od[0,1] = 0.1
    od[0,3] = 0.15
    od[0,2] = 0.15
    od[4,3] = 0.3

    inflows, turn_props = convert_od_to_flows_and_turn_props(od, turn_flow_matrix)

    test5_inflows = np.array([0.6, 0., 0., 0., 0.3, 0., 0., 0.])
    test5_turn_props = np.zeros(shape=(24,24))
    test5_turn_props[16,2] = 1/3
    test5_turn_props[16,3] = 1/6
    test5_turn_props[16,1] = 1/2
    test5_turn_props[2,10] = 1.
    test5_turn_props[20,1] = 1.
    test5_turn_props[1,5] = 3/4
    test5_turn_props[1,4] = 1/4
    run_test(test5_inflows, test5_turn_props, inflows, turn_props, 'Test 5')
