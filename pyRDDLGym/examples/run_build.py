from pyRDDLGym.core.builder import RDDLBuilder


def main():
    builder = RDDLBuilder()

    # add objects
    builder.add_object_type('x_pos')
    builder.add_object_type('y_pos')

    # add non-fluents
    builder.add_pvariable('COST_CUTOUT', [], 'non-fluent', 'real', -5.0)
    builder.add_pvariable('COST_PUTOUT', [], 'non-fluent', 'real', -10.0)
    builder.add_pvariable('PENALTY_TARGET_BURN', [], 'non-fluent', 'real', -100.0)
    builder.add_pvariable('PENALTY_NONTARGET_BURN', [], 'non-fluent', 'real', -5.0)
    builder.add_pvariable('NEIGHBOR', ['x_pos', 'y_pos', 'x_pos', 'y_pos'], 'non-fluent', 'bool', 'false')
    builder.add_pvariable('TARGET', ['x_pos', 'y_pos'], 'non-fluent', 'bool', 'false')

    # add fluents
    builder.add_pvariable('burning', ['x_pos', 'y_pos'], 'state-fluent', 'bool', 'false')
    builder.add_pvariable('out-of-fuel', ['x_pos', 'y_pos'], 'state-fluent', 'bool', 'false')
    builder.add_pvariable('put-out', ['x_pos', 'y_pos'], 'action-fluent', 'bool', 'false')
    builder.add_pvariable('cut-out', ['x_pos', 'y_pos'], 'action-fluent', 'bool', 'false')

    # add cpfs and reward
    builder.add_cpf('burning', ['?x', '?y'],
        '''if ( put-out(?x, ?y) )
				then false
            else if (~out-of-fuel(?x, ?y) ^ ~burning(?x, ?y))
              then [if (TARGET(?x, ?y) ^ ~(exists_{?x2: x_pos, ?y2: y_pos} (NEIGHBOR(?x, ?y, ?x2, ?y2) ^ burning(?x2, ?y2))))
                    then false
                    else Bernoulli( 1.0 / (1.0 + exp[4.5 - (sum_{?x2: x_pos, ?y2: y_pos} (NEIGHBOR(?x, ?y, ?x2, ?y2) ^ burning(?x2, ?y2)))]) ) ]
			else 
				burning(?x, ?y)'''
    )
    builder.add_cpf('out-of-fuel', ['?x', '?y'],
        'out-of-fuel(?x, ?y) | burning(?x,?y) | (~TARGET(?x, ?y) ^ cut-out(?x, ?y))'                
    )
    builder.add_reward(
        '''[sum_{?x: x_pos, ?y: y_pos} [ COST_CUTOUT*cut-out(?x, ?y) ]]
        + [sum_{?x: x_pos, ?y: y_pos} [ COST_PUTOUT*put-out(?x, ?y) ]]
        + [sum_{?x: x_pos, ?y: y_pos} [ PENALTY_TARGET_BURN*[ (burning(?x, ?y) | out-of-fuel(?x, ?y)) ^ TARGET(?x, ?y) ]]]
        + [sum_{?x: x_pos, ?y: y_pos} [ PENALTY_NONTARGET_BURN*[ burning(?x, ?y) ^ ~TARGET(?x, ?y) ]]]'''
    )

    # add objects
    numx, numy = 30, 30
    builder.add_object_values('x_pos', [f'x{i + 1}' for i in range(numx)])
    builder.add_object_values('y_pos', [f'y{i + 1}' for i in range(numy)])

    # add non-fluent values
    for x1 in range(numx):
        for y1 in range(numy):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x2, y2 = x1 + dx, y1 + dy
                    if 0 <= x2 < numx and 0 <= y2 < numy and not (dx == 0 and dy == 0):
                        builder.add_nonfluent_init(
                            'NEIGHBOR', [f'x{x1 + 1}', f'y{y1 + 1}', f'x{x2 + 1}', f'y{y2 + 1}'], 'true')
                        
    for x0, y0, k in [(25, 25, 3), (27, 19, 2), (5, 23, 2)]:
        for x in range(max(0, x0 - k), min(numx, x0 + k + 1)):
            for y in range(max(0, y0 - k), min(numy, y0 + k + 1)):
                if (x - x0) ** 2 + (y - y0) ** 2 < k ** 2:
                    builder.add_nonfluent_init(
                        'TARGET', [f'x{x + 1}', f'y{y + 1}'], 'true')

    # add init-state values
    x0, y0, k = 6, 6, 2
    for x in range(max(0, x0 - k), min(numx, x0 + k + 1)):
        for y in range(max(0, y0 - k), min(numy, y0 + k + 1)):
            if (x - x0) ** 2 + (y - y0) ** 2 < k ** 2:
                builder.add_init_state(
                    'burning', [f'x{x + 1}', f'y{y + 1}'], 'true')

    # add constants
    builder.add_max_nondef_actions(3)
    builder.add_horizon(60)
    builder.add_discount(1.0)
    
    # save
    domtxt = builder.build_domain('wildfire')
    instxt = builder.build_instance('wildfire', 'wildfire_large', 'nf_wildfire_large')
    with open('domain.rddl', 'w') as domfile, open('instance0.rddl', 'w') as insfile:
        domfile.write(domtxt)
        insfile.write(instxt)

 
if __name__ == '__main__':
    main()
