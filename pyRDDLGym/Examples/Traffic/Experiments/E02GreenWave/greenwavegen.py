from sys import argv
from pyRDDLGym.Examples.Traffic.netgen import generate_green_wave_scenario

if __name__ == '__main__':

    with open(argv[1], 'w') as file:
        network = generate_green_wave_scenario(
            N=10)
        file.write(network)

