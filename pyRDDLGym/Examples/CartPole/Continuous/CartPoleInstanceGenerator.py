import os

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class CartPoleInstanceGenerator(InstanceGenerator):
    
    def get_env_name(self) -> str:
        return os.path.join('CartPole', 'Continuous')
    
    def generate_instance(self, instance: int) -> str:
        
        # regular cart-pole
        if instance == 1:
            nonfluents = {'GRAVITY': 9.8, 'CART-MASS': 1.0, 'POLE-MASS': 0.1,
                          'POLE-LEN': 0.5, 'CART-FRICTION': 0.0, 'POLE-FRICTION': 0.0,
                          'IMPULSE-VAR': 0.0, 'ANGLE-VAR': 0.0}
            state = {'pos': 0.0, 'vel': 0.05, 'ang-pos': 0.0, 'ang-vel': -0.02}
            horizon = 200
        
        # cart-pole with long pole + reduced cart mass
        elif instance == 2:
            nonfluents = {'GRAVITY': 9.8, 'CART-MASS': 0.5, 'POLE-MASS': 0.1,
                          'POLE-LEN': 3.0, 'CART-FRICTION': 0.0, 
                          'POLE-FRICTION': 0.0, 
                          'IMPULSE-VAR': 0.0, 'ANGLE-VAR': 0.0}
            state = {'pos': 0.0, 'vel': 0.05, 'ang-pos': 0.0, 'ang-vel': -0.02}
            horizon = 200
            
        # cart-pole with friction + impulse noise
        elif instance == 3:
            nonfluents = {'GRAVITY': 9.8, 'CART-MASS': 1.0, 'POLE-MASS': 0.1,
                          'POLE-LEN': 0.5, 'CART-FRICTION': 0.0005, 
                          'POLE-FRICTION': 0.000002, 
                          'IMPULSE-VAR': 16.0, 'ANGLE-VAR': 0.0}
            state = {'pos': 0.0, 'vel': 0.05, 'ang-pos': 0.0, 'ang-vel': -0.02}
            horizon = 200
            
        # cart-pole with friction + sensor noise
        elif instance == 4:
            nonfluents = {'GRAVITY': 9.8, 'CART-MASS': 1.0, 'POLE-MASS': 0.1,
                          'POLE-LEN': 0.5, 'CART-FRICTION': 0.0005, 
                          'POLE-FRICTION': 0.000002, 
                          'IMPULSE-VAR': 0.0, 'ANGLE-VAR': 0.001}
            state = {'pos': 0.0, 'vel': 0.05, 'ang-pos': 0.0, 'ang-vel': -0.02}
            horizon = 200
        
        # cart-pole with friction + sensor noise + impulse noise
        elif instance == 5:
            nonfluents = {'GRAVITY': 9.8, 'CART-MASS': 1.0, 'POLE-MASS': 0.1,
                          'POLE-LEN': 0.5, 'CART-FRICTION': 0.0005, 
                          'POLE-FRICTION': 0.000002, 
                          'IMPULSE-VAR': 16.0, 'ANGLE-VAR': 0.001}
            state = {'pos': 0.0, 'vel': 0.05, 'ang-pos': 0.0, 'ang-vel': -0.02}
            horizon = 200
        
        else:
            raise Exception(f'Invalid instance {instance} for CartPole.')
        
        value = f'non-fluents cart_pole_{instance}' + ' {'
        value += '\n' + 'domain = cart_pole_continuous;'
        value += '\n' + 'non-fluents {' + '\n\t'
        nfs = []
        for (name, nfvalue) in nonfluents.items():
            nfvalue = '{:.10f}'.format(nfvalue)
            nfs.append(f'{name} = {nfvalue};')
        value += '\n\t'.join(nfs)
        value += '\n\t' + '};'
        value += '\n' + '}'
        
        value += '\n' + f'instance inst_cart_pole_{instance}' + ' {'
        value += '\n' + 'domain = cart_pole_continuous;'
        value += '\n' + f'non-fluents = cart_pole_{instance};'
        value += '\n' + 'init-state {'
        for (name, statevalue) in state.items():
            statevalue = '{:.10f}'.format(statevalue)
            value += '\n\t' + f'{name} = {statevalue};'
        value += '\n' + '};'
        value += '\n' + 'max-nondef-actions = pos-inf;'
        value += '\n' + f'horizon = {horizon};'
        value += '\n' + f'discount = 1.0;'
        value += '\n' + '}'
        return value
            

inst = CartPoleInstanceGenerator()
inst.save_instances(5)
