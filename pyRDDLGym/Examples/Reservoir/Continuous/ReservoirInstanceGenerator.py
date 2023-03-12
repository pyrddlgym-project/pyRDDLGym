import os

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class ReservoirInstanceGenerator(InstanceGenerator):
    
    def get_env_name(self) -> str:
        return os.path.join('Reservoir', 'Continuous')
    
    def generate_instance(self, instance: int) -> str:
        
        # two reservoirs
        if instance == 1:
            nodes, edges = self.generate_dag([(1, 2)])
            rlevels = [50., 60.]
            MIN_LEVEL = [20.0] * len(nodes)
            MAX_LEVEL = [80.0] * len(nodes)
            TOP_RES = [100.0] * len(nodes)
            horizon = 100
            
        # three reservoirs
        elif instance == 2:
            nodes, edges = self.generate_dag([(1, 3), (2, 3)])
            rlevels = [45., 50., 50.]
            MIN_LEVEL = [20.0] * len(nodes)
            MAX_LEVEL = [80.0] * len(nodes)
            TOP_RES = [100.0] * len(nodes)
            horizon = 100
        
        # five reservoirs
        elif instance == 3:
            nodes, edges = self.generate_dag([(1, 3), (2, 3), (3, 5), (4, 5)])
            rlevels = [45., 50., 50., 45., 60.]
            MIN_LEVEL = [20.0] * len(nodes)
            MAX_LEVEL = [80.0] * len(nodes)
            TOP_RES = [100.0] * len(nodes)
            horizon = 100
        
        # ten reservoirs
        elif instance == 4:
            nodes, edges = self.generate_dag([(1, 5), (2, 5), (2, 6), (3, 6), 
                                              (3, 7), (4, 7), (5, 8), (6, 8), 
                                              (6, 9), (7, 9), (8, 10), (9, 10)])
            rlevels = [45., 50., 50., 60., 50., 50., 50., 40., 50., 95.]
            MIN_LEVEL = [20.0] * len(nodes)
            MAX_LEVEL = [80.0] * len(nodes)
            TOP_RES = [100.0] * len(nodes)
            horizon = 100
        
        # twenty reservoirs
        elif instance == 5:
            nodes, edges = self.generate_dag([(1, 2), (2, 3), (3, 4), (4, 5), 
                                              (5, 6), (6, 7), (7, 8), (8, 9),
                                              (9, 10), (10, 11), (11, 12), (12, 13),
                                              (13, 14), (14, 15), (15, 16), (16, 17),
                                              (17, 18), (18, 19), (19, 20)])
            MIN_LEVEL = [60.0] * len(nodes)
            MAX_LEVEL = [140.0] * len(nodes)
            TOP_RES = [200.0] * len(nodes)
            rlevels = [80.] * 20
            horizon = 100
            
        else:
            raise Exception(f'Invalid instance {instance} for Reservoir.')
        
        value = f'non-fluents reservoir_{instance}' + ' {'
        value += '\n' + 'domain = reservoir_control_dis;'
        value += '\n' + 'objects {'
        value += '\n\t' + 'reservoir : {' + ','.join(nodes) + '};'
        value += '\n' + '};'
        value += '\n' + 'non-fluents {' + '\n\t'
        nfs = []
        for (e1, e2) in edges:
            nfs.append(f'RES_CONNECT({e1},{e2});')
        nfs.append(f'CONNECTED_TO_SEA({nodes[-1]});')    
        for (node, minl) in zip(nodes, MIN_LEVEL):
            nfs.append(f'MIN_LEVEL({node}) = {minl};')
        for (node, maxl) in zip(nodes, MAX_LEVEL):
            nfs.append(f'MAX_LEVEL({node}) = {maxl};')
        for (node, topl) in zip(nodes, TOP_RES):
            nfs.append(f'TOP_RES({node}) = {topl};')
        value += '\n\t'.join(nfs)
        value += '\n\t' + '};'
        value += '\n' + '}'
        
        value += '\n' + f'instance inst_reservoir_{instance}' + ' {'
        value += '\n' + 'domain = reservoir_control_dis;'
        value += '\n' + f'non-fluents = reservoir_{instance};'
        value += '\n' + 'init-state {'
        for (node, rlevel) in zip(nodes, rlevels):
            value += '\n\t' + f'rlevel({node}) = {rlevel};'
        value += '\n' + '};'
        value += '\n' + 'max-nondef-actions = pos-inf;'
        value += '\n' + f'horizon = {horizon};'
        value += '\n' + f'discount = 1.0;'
        value += '\n' + '}'
        return value
    
    def generate_dag(self, edges):
        nodes = sorted({i for tup in edges for i in tup})
        nodes = [f't{node}' for node in nodes]
        edges = [(f't{e1}', f't{e2}') for (e1, e2) in edges]
        return nodes, edges

inst = ReservoirInstanceGenerator()
inst.save_instance(5)
        