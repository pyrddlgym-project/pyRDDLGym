from pyRDDLGym.Visualizer.visualize_dbn import RDDL2Graph


def test_dbn_visualization():
    domains = [
        'CartPole discrete', 'CartPole continuous', 'MarsRover', 
        'MountainCar', 'PowerGeneration', 'RaceCar', 'UAV continuous', 
        'UAV discrete', 'UAV mixed', 'Wildfire', 'SupplyChain'
    ]
    
    for domain in domains:
        r2g = RDDL2Graph(
            domain=domain,
            instance=0,
            directed=True,
            strict_grouping=True
        )
        r2g.save_dbn(file_name=domain)


if __name__ == "__main__":
    test_dbn_visualization()
