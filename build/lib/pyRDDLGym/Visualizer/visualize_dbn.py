import pygraphviz as pgv
from pathlib import Path
from typing import Optional, Tuple, Dict, Union, List, Set

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Examples.ExampleManager import ExampleManager

COLOR = dict(
    action='olivedrab1',
    interm='sandybrown',
    derived='orchid',
    observ='orangered',
    state='lightblue',
    next_state='gold1',
    reward='firebrick1'
)

SHAPE = dict(
    action='box',
    interm='ellipse',
    derived='octagon',
    observ='ellipse',
    state='ellipse',
    next_state='ellipse',
    reward='diamond'
)

STYLE = dict(
    action='filled',
    interm='filled',
    derived='filled',
    observ='filled',
    state='filled',
    next_state='filled',
    reward='filled'
)


class Graph(pgv.AGraph):

    def __init__(self,
                 thing=None,
                 filename=None,
                 data=None,
                 string=None,
                 handle=None,
                 name="",
                 strict=True,
                 directed=False,
                 **attr):
        super().__init__(thing, filename, data, string, handle, name, 
                         strict, directed, **attr)
        self._supress_rank = False
        self._same_rank: List[List[str]] = []
    
    def set_suppress_rank(self, suppress_rank: bool):
        self._suppress_rank = suppress_rank
    
    def add_same_rank(self, nodes: List[str]):
        self._same_rank.append(nodes)
    
    def configure_graph_attributes(self, **attrs):
        self.graph_attr.update(attrs)

    def configure_node_attributes(self, **attrs):
        self.node_attr.update(attrs)

    def configure_edge_attributes(self, **attrs):
        self.edge_attr.update(attrs)

    def set_ranks(self):
        if not self._supress_rank:
            for r in self._same_rank:
                self.add_subgraph(r, rank='same')


class RDDL2Graph:

    def __init__(self, domain: str='Wildfire',
                 instance: int=0,
                 directed: bool=True,
                 strict_grouping: bool=False,
                 simulation: bool=False):

        # Read the domain and instance files
        self._domain, self._instance = domain, str(instance)
        env_info = ExampleManager.GetEnvInfo(domain)
        domain = env_info.get_domain()
        instance = env_info.get_instance(instance)
        
        # Read and parse domain and instance
        reader = RDDLReader(domain, instance)
        domain = reader.rddltxt
        parser = RDDLParser(None, False)
        parser.build()

        # Parse RDDL file
        rddl_ast = parser.parse(domain)

        # Ground domain
        grounder = RDDLGrounder(rddl_ast)
        model = grounder.Ground()

        # XADD compilation
        self.model = RDDLModelWXADD(model, simulation=simulation)
        self.model.compile()
        
        self.cpfs: Dict[str, Union[int, List[int]]] = self.model.cpfs
        self.cpfs.update({'reward': self.model.reward})
        self.cpfs.update({'terminals': self.model.terminals})
        self._directed = directed
        self._strict_grouping = strict_grouping
        self._gvar_to_node_name = {}
        self._node_name_to_gvar = {}
        self._action: Set[str] = set()
        self._state = set()
        self._next_state = set()
        self._observ = set()
        self._interm = set()
        self._derived = set()
        self._parents = {}
    
    def configure_graph(self, graph: Graph):
        graph.configure_graph_attributes(
            fontname="Helvetica",
            fontsize="16",
            ratio="auto",
            size="7.5,10",
            rankdir="LR",
            ranksep="2.00"
        )
    
    def configure_nodes(self, graph: Graph):
        graph.configure_node_attributes(
            fontsize="16",
            color='black',
        )
        for node in graph.nodes():
            color, style, shape = [''] * 3
            if node in self._state:
                color, shape, style = COLOR['state'], SHAPE['state'], STYLE['state']
            elif node in self._next_state:
                color, shape, style = COLOR['next_state'], SHAPE['next_state'], STYLE['next_state']
            elif node in self._action:
                color, shape, style = COLOR['action'], SHAPE['action'], STYLE['action']
            elif node in self._interm:
                color, shape, style = COLOR['interm'], SHAPE['interm'], STYLE['interm']
            elif node in self._observ:
                color, shape, style = COLOR['observ'], SHAPE['observ'], STYLE['observ']
            elif node in self._derived:
                color, shape, style = COLOR['derived'], SHAPE['derived'], STYLE['derived']
            elif node == 'Reward Function':
                color, shape, style = COLOR['reward'], SHAPE['reward'], STYLE['reward']
            if color or style or shape:
                node.attr['color'] = color
                node.attr['style'] = style
                node.attr['shape'] = shape
        
    def configure_edges(self, graph: Graph):
        if not self._directed:
            graph.configure_edge_attributes(fontsize="16")

    def save_dbn(self, file_name: str='example',
                 fluent: Optional[str]=None,
                 gfluent: Optional[str]=None,
                 file_type: str='pdf'):
        self.clear_cache()
        f_dir = Path(f"tmp/{self._domain}")
        f_dir.mkdir(exist_ok=True, parents=True)
        f_path = f_dir / \
            f"{file_name}{('_' + fluent) if fluent else ''}{('_' + gfluent) if gfluent else ''}_inst_{self._instance}"
        if fluent is None and gfluent is None:
            graph = self.get_graph_of_instance()
        else:
            graph = self.get_graph_of_cpf(fluent, gfluent)
        self.configure_graph(graph)
        self.configure_nodes(graph)
        self.configure_edges(graph)
        graph.set_ranks()
        graph.draw(f"{str(f_path)}.{file_type}", prog='dot')

        # Output a file
        with open(f_dir / f"{file_name}_inst_{self._instance}.txt", "w") as f:
            txt = self.get_text_repr()
            f.write(txt)
    
    def get_text_repr(self):
        """Returns a string that summarizes the DBN dependency structure of a given RDDL instance"""
        res = f"DBN dependency analysis\n\n" \
              f"Domain: {self._domain}\n" \
              f"Instance: {self._instance}\n\n"
        
        res += f"{'Grounded fluent':40}{'Parent variables':100}\n\n"
        # Ground fluents (including terminals and reward)
        for cpf, node_id in self.model.cpfs.items():
            if not node_id or cpf == 'terminals':
                continue
            parents = self.model.collect_vars(node_id)
            parents = sorted([self.get_node_name(p) for p in parents])
            node_name = self.get_node_name(cpf) if cpf != 'reward' else cpf
            cpf_str = f"{node_name:40}{', '.join(parents):100}".strip(',')
            res += f"{cpf_str}\n"
        return res

    def get_graph_of_cpf(self, fluent: str, gfluent: Optional[str]=None) -> Graph:
        assert fluent in self.model.pvar_to_type or fluent.lower() == 'reward', \
            f"Fluent {fluent} not recognized"
        assert not gfluent or (gfluent and f"{fluent}_{gfluent}" in self.model.gvar_to_type), \
            f"Grounded fluent provided but cannot be resolved"

        graph = Graph(directed=self._directed)
        if not self._strict_grouping:
            graph.set_suppress_rank(True)
        
        if fluent.lower() == 'reward':
            gvars = ['reward']
        else:
            gvars = {
                gvar for gvar, pvar in self.model.gvar_to_pvar.items() 
                    if pvar == fluent and 
                        ((not gfluent) or (gfluent and f"{fluent}_{gfluent}" == gvar))
            }
        
        for gvar in gvars:
            if gvar in self.model.interm:
                self.add_single_interm_fluent_to_graph(graph, gvar)
            elif gvar in self.model.derived:
                self.add_single_derived_fluent_to_graph(graph, gvar)
            elif gvar in self.model.states:
                self.add_single_state_fluent_to_graph(graph, gvar)
            elif gvar in self.model.observ:
                self.add_single_observ_fluent_to_graph(graph, gvar)
            elif gvar in self.model.actions:
                raise NotImplementedError("For action variables, generate the entire DBN")
            elif gvar == 'reward':
                self.add_reward_to_graph(graph)

        # Handle action nodes
        action_nodes = list(self._action)
        for act in action_nodes:
            graph.add_node(act)
        
        # Handle interm & derived fluents
        interm_derived_nodes = list(self._interm) + list(self._derived)
        interm_derived_nodes = [f"Intermediate"] + interm_derived_nodes
        graph.add_same_rank(interm_derived_nodes)
        
        # Handle observations
        obs_nodes = ["Observations"] + list(self._observ)
        if len(obs_nodes) > 1:
            graph.add_same_rank(obs_nodes)
        
        state_nodes = list(self._state)
        action_states = ["Current State and Actions"] + action_nodes + state_nodes
        graph.add_same_rank(action_states)

        # Handle reward node
        if fluent.lower() != 'reward':
            self.add_reward_to_graph(graph, fluent=fluent, gfluent=gfluent)
        
        # Put next state and reward nodes at same rank
        next_state_nodes = list(self._next_state)
        rew_ns_nodes = ['Reward Function', 'Next State and Reward'] + next_state_nodes
        graph.add_same_rank(rew_ns_nodes)

        # Setup some stratification nodes and links for strict levels
        if self._strict_grouping:
            prev_level = "Current State and Actions"
            graph.add_node(prev_level, color='white', shape='plaintext', style='bold')

            # interm & derived
            level_node = f"Intermediate"
            graph.add_edge(prev_level, level_node, color='black', style='invis', label='')
            graph.add_node(level_node, color='white', shape='plaintext', style='bold')
            prev_level = level_node
            
            node_name = "Next State and Reward"
            graph.add_node(node_name, color='white', shape='plaintext', style='bold')
            graph.add_edge(prev_level, node_name, color='black', style='invis', label='')

            # Observations (at the right-most column)
            if len(obs_nodes) > 1:  # will always contain default "Observations" node
                graph.add_node("Observations", color='white', shape='plaintext', style='bold')
                graph.add_edge(prev_level, "Observations", color='black', style='invis', label='')
                prev_level = "Observations"
        
        return graph
        
    def get_graph_of_instance(self) -> Graph:
        graph = Graph(directed=self._directed)
        if not self._strict_grouping:
            graph.set_suppress_rank(True)
        
        # Go through all actions
        for act in self.model.actions:
            node_name = self.get_node_name(act)
            self.add_node(node_name)
            graph.add_node(node_name)
        
        # Handle intermediate & derived variables
        self.add_interm_fluents_to_graph(graph)
        self.add_derived_fluents_to_graph(graph)
        interm_derived_nodes = list(self._interm) + list(self._derived)
        interm_derived_nodes = [f"Intermediate"] + interm_derived_nodes
        graph.add_same_rank(interm_derived_nodes)
        
        # Handle observations
        self.add_observ_fluents_to_graph(graph)
        obs_nodes = ["Observations"] + list(self._observ)
        if len(obs_nodes) > 1:
            graph.add_same_rank(obs_nodes)
        
        # Handle current and next state CPFs
        self.add_state_fluents_to_graph(graph)
        
        # Put all current state and action nodes at same rank
        action_nodes = list(self._action)
        state_nodes = list(self._state)
        action_states = ["Current State and Actions"] + action_nodes + state_nodes
        graph.add_same_rank(action_states)

        # Handle reward node
        self.add_reward_to_graph(graph)

        # Put Next State and Reward nodes at same rank
        next_state_nodes = list(self._next_state)
        rew_ns_nodes = ['Reward Function', 'Next State and Reward'] + next_state_nodes
        graph.add_same_rank(rew_ns_nodes)

        # Setup some stratification nodes and links if strict levels
        if self._strict_grouping:
            prev_level = "Current State and Actions"
            graph.add_node(prev_level, color='white', shape='plaintext', style='bold')

            level_node = f"Intermediate"
            graph.add_edge(prev_level, level_node, color='black', style='invis', label='')
            graph.add_node(level_node, color='white', shape='plaintext', style='bold')
            prev_level = level_node
            
            node_name = "Next State and Reward"
            graph.add_node(node_name, color='white', shape='plaintext', style='bold')
            graph.add_edge(prev_level, node_name, color='black', style='invis', label='')

            # Observations
            if len(obs_nodes) > 1:  # will always contain default "Observations" node
                graph.add_node("Observations", color='white', shape='plaintext', style='bold')
                graph.add_edge(prev_level, "Observations", color='black', style='invis', label='')
                prev_level = "Observations"
        
        return graph
            
    def add_reward_to_graph(
            self, graph: Graph, fluent: Optional[str]=None, gfluent: Optional[str]=None
    ):
        graph.add_node("Reward Function")
        parents = self.model.vars_in_rew
        for p in parents:
            pvar = self.model.gvar_to_pvar[p]
            if not fluent or \
                (fluent and pvar == fluent and not gfluent) or \
                    (gfluent and (p == f"{fluent}_{gfluent}" or p == f"{fluent}_{gfluent}'")):
                p_node_name = self.get_node_name(p)
                self.add_node(p_node_name)
                graph.add_edge(p_node_name, 'Reward Function')

    def add_single_state_fluent_to_graph(self, graph: Graph, state: str):        
        ns = self.model.next_state[state]  # Primed state variable
        
        # Get node names and add to the graph
        s_node_name = self.get_node_name(state)
        ns_node_name = self.get_node_name(ns, primed=True)
        self.add_node(s_node_name); self.add_node(ns_node_name)
        graph.add_node(s_node_name); graph.add_node(ns_node_name)
        
        # Get parents and add links
        parents = self.model.collect_vars(self.cpfs[ns])
        for p in parents:
            p_node_name = self.get_node_name(
                p, primed=p in self.model.next_state.values())
            self.add_node(p_node_name)
            graph.add_edge(
                p_node_name,
                ns_node_name,
                color='black',
                style='solid',
                label=''
            )
    
    def add_state_fluents_to_graph(self, graph: Graph):
        for st in self.model.states:
            self.add_single_state_fluent_to_graph(graph, st)

    def add_single_observ_fluent_to_graph(self, graph: Graph, observ: str):
        node_name = self.get_node_name(observ)
        self._gvar_to_node_name[observ] = node_name
        graph.add_node(node_name)
        self.add_node(node_name)

        # Get parent nodes and add links
        parents = self.model.collect_vars(self.cpfs[observ])
        for p in parents:
            p_node_name = self.get_node_name(p)
            self.add_node(p_node_name)
            graph.add_edge(p_node_name, node_name)
    
    def add_observ_fluents_to_graph(self, graph: Graph):
        for observ in self.model.observ:
            self.add_single_observ_fluent_to_graph(graph, observ)
    
    def add_single_interm_fluent_to_graph(self, graph: Graph, interm: str):
        node_name = self.get_node_name(interm)
        self._gvar_to_node_name[interm] = node_name
        self.add_node(node_name)
        graph.add_node(node_name)

        # Get parent nodes and add links
        parents = self.model.collect_vars(self.cpfs[interm])
        for p in parents:
            p_node_name = self.get_node_name(p)
            self.add_node(p_node_name)
            graph.add_edge(p_node_name, node_name)

    def add_interm_fluents_to_graph(self, graph: Graph):
        for interm in self.model.interm:
            self.add_single_interm_fluent_to_graph(graph, interm)
    
    def add_single_derived_fluent_to_graph(self, graph: Graph, derived: str):
        node_name = self.get_node_name(derived)
        self._gvar_to_node_name[derived] = node_name
        self.add_node(node_name)
        graph.add_node(node_name)

        # Get parent nodes and add links
        parents = self.model.collect_vars(self.cpfs[derived])
        for p in parents:
            p_node_name = self.get_node_name(p)
            self.add_node(p_node_name)
            graph.add_edge(p_node_name, node_name)

    def add_derived_fluents_to_graph(self, graph: Graph):
        for derived in self.model.derived:
            self.add_single_derived_fluent_to_graph(graph, derived)
    
    @staticmethod
    def get_objects(gvar: str, pvar: str) -> List[str]:
        return list(map(lambda x: '?' + x, 
                        [v_str for v_str in gvar.split(pvar)[1].split('_')[1:] if v_str]))
    
    def add_node(self, node_name: str):
        gvar = self._node_name_to_gvar[node_name]
        if gvar in self.model.prev_state.values():
            self._state.add(node_name)
        elif gvar in self.model.next_state.values():
            self._next_state.add(node_name)
        elif gvar in self.model.interm:
            self._interm.add(node_name)
        elif gvar in self.model.derived:
            self._derived.add(node_name)
        elif gvar in self.model.observ:
            self._observ.add(node_name)
        elif gvar in self.model.actions:
            self._action.add(node_name)

    def clear_cache(self):
        self._state.clear()
        self._next_state.clear()
        self._interm.clear()
        self._derived.clear()
        self._observ.clear()
        self._action.clear()
        self._gvar_to_node_name.clear()
        self._node_name_to_gvar.clear()
    
    def get_node_name(self, gvar: str, primed: bool=False) -> str:
        if gvar in self._gvar_to_node_name:
            return self._gvar_to_node_name[gvar]

        pvar = self.get_pvar_from_gvar(gvar)
        objects = RDDL2Graph.get_objects(gvar if not primed else gvar[:-1], pvar)
        node_name = f"{pvar}" + ("'" if primed else '') + \
                    f"{('(' + ','.join(objects) + ')') if len(objects) > 0 else ''}"
        self._gvar_to_node_name[gvar] = node_name
        self._node_name_to_gvar[node_name] = gvar
        return node_name
        
    def get_pvar_from_gvar(self, gvar: str):
        pvar = self.model.gvar_to_pvar.get(gvar)
        assert pvar is not None
        return pvar


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default='Wildfire')
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--undirected", action='store_true')
    parser.add_argument("--strict_grouping", action='store_true')
    parser.add_argument("--fluent", type=str, default=None)
    parser.add_argument("--gfluent", type=str, default=None)
    parser.add_argument("--simulation", action='store_true')

    args = parser.parse_args()

    r2g = RDDL2Graph(
        domain=args.domain,
        instance=args.instance,
        directed=not args.undirected,
        strict_grouping=args.strict_grouping,
        simulation=args.simulation,
    )
    if args.fluent and args.gfluent:
        r2g.save_dbn(file_name=args.domain, fluent=args.fluent, gfluent=args.gfluent)
    elif args.fluent:
        r2g.save_dbn(file_name=args.domain, fluent=args.fluent)
    else:
        r2g.save_dbn(file_name=args.domain)
