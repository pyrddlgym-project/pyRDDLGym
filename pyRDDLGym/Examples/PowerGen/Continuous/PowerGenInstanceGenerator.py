import os

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class PowerGenInstanceGenerator(InstanceGenerator):
    
    def get_env_name(self) -> str:
        return os.path.join('PowerGen', 'Continuous')
    
    def generate_instance(self, instance: int) -> str:
        
        # three generators, lower variance
        if instance == 1:
            plants = ['p1', 'p2', 'p3']
            TEMP_VARIANCE = 4.0
            PROD_UNITS_MIN = [0.0] * len(plants)
            PROD_UNITS_MAX = [10.0] * len(plants)
            EXP_COEFF = 0.01
            MIN_DEMAND_TEMP = 11.7
            MIN_CONSUMPTION = 2.0
            temperature = 10.
            horizon = 40
        
        # five generators, lower variance
        elif instance == 2:
            plants = ['p1', 'p2', 'p3', 'p4', 'p5']
            TEMP_VARIANCE = 4.0
            PROD_UNITS_MIN = [0.0] * len(plants)
            PROD_UNITS_MAX = [10.0] * len(plants)
            EXP_COEFF = 0.01
            MIN_DEMAND_TEMP = 11.7
            MIN_CONSUMPTION = 2.0
            temperature = 10.
            horizon = 40
        
        # ten generators, mid variance
        elif instance == 3:
            plants = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
            TEMP_VARIANCE = 6.0
            PROD_UNITS_MIN = [0.0] * len(plants)
            PROD_UNITS_MAX = [10.0] * len(plants)
            EXP_COEFF = 0.015
            MIN_DEMAND_TEMP = 11.7
            MIN_CONSUMPTION = 2.0
            temperature = 10.
            horizon = 40
        
        # fifteen generators, mid variance
        elif instance == 4:
            plants = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
                      'p11', 'p12', 'p13', 'p14', 'p15']
            TEMP_VARIANCE = 6.0
            PROD_UNITS_MIN = [0.0] * len(plants)
            PROD_UNITS_MAX = [10.0] * len(plants)
            EXP_COEFF = 0.02
            MIN_DEMAND_TEMP = 11.7
            MIN_CONSUMPTION = 2.0
            temperature = 10.
            horizon = 40
        
        # twenty generators, high variance
        elif instance == 5:
            plants = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
                      'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 
                      'p19', 'p20']
            TEMP_VARIANCE = 8.0
            PROD_UNITS_MIN = [0.0] * len(plants)
            PROD_UNITS_MAX = [10.0] * len(plants)
            EXP_COEFF = 0.02
            MIN_DEMAND_TEMP = 11.7
            MIN_CONSUMPTION = 2.0
            temperature = 10.
            horizon = 40
        
        else:
            raise Exception(f'Invalid instance {instance} for PowerGen.')
        
        value = f'non-fluents power_gen_{instance}' + ' {'
        value += '\n' + 'domain = power_gen;'
        value += '\n' + 'objects {'
        value += '\n\t' + 'plant : {' + ','.join(plants) + '};'
        value += '\n' + '};'
        value += '\n' + 'non-fluents {' + '\n\t'
        nfs = []
        for (node, minl) in zip(plants, PROD_UNITS_MIN):
            nfs.append(f'PROD-UNITS-MIN({node}) = {minl};')
        for (node, maxl) in zip(plants, PROD_UNITS_MAX):
            nfs.append(f'PROD-UNITS-MAX({node}) = {maxl};')
        nfs.append(f'TEMP-VARIANCE = {TEMP_VARIANCE};')
        nfs.append(f'DEMAND-EXP-COEF = {EXP_COEFF};')
        nfs.append(f'MIN-DEMAND-TEMP = {MIN_DEMAND_TEMP};')
        nfs.append(f'MIN-CONSUMPTION = {MIN_CONSUMPTION};')
        value += '\n\t'.join(nfs)
        value += '\n\t' + '};'
        value += '\n' + '}'
        
        value += '\n' + f'instance inst_power_gen_{instance}' + ' {'
        value += '\n' + 'domain = power_gen;'
        value += '\n' + f'non-fluents = power_gen_{instance};'
        value += '\n' + 'init-state {'
        value += '\n\t' + f'temperature = {temperature};'
        value += '\n' + '};'
        value += '\n' + 'max-nondef-actions = pos-inf;'
        value += '\n' + f'horizon = {horizon};'
        value += '\n' + f'discount = 1.0;'
        value += '\n' + '}'
        return value
    

inst = PowerGenInstanceGenerator()
inst.save_instance(5)