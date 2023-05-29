from sys import argv
from pyRDDLGym.Examples.Traffic.netgen import generate_webster_scenario

if __name__ == '__main__':

    with open(argv[1], 'w') as file:
        network = generate_webster_scenario(
            d=1.87,
            dEWfrac=1/3,
            dNSfrac=2/3,
            betaL=1/3,
            betaT=2/3,
            link_lengths=250,
            min_green=6,
            max_green=60,
            all_red=4,
            mu=0.53,
            instance_name=None,
            horizon=1024,
            discount=1.0)
        file.write(network)

