import argparse

from pyRDDLGym.Visualizer.visualize_dbn import RDDL2Graph


def test_dbn_visualization(args: argparse.Namespace):
    domains = [
        'cartpole discrete',
        'cartpole continuous',
        'marsrover', 
        # 'mountaincar',    # Takes too long to create DBN graph for this
        'powergen continuous',
        'powergen discrete',
        'racecar',
        'uavcontinuous', 
        'uavdiscrete',
        'uavmixed',
        'wildfire',
        'supplychain', 
    ]
    
    for domain in domains:
        print(f"Creating DBN graph for {domain}...")
        r2g = RDDL2Graph(
            domain=domain,
            instance=0,
            directed=True,
            strict_grouping=True,
            simulation=args.simulation,
        )
        r2g.save_dbn(file_name=domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', action='store_true')
    args = parser.parse_args()

    test_dbn_visualization(args)
