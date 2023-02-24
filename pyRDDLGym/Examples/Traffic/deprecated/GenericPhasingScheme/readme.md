To generate instances of a grid network with r rows and c columns:

    python netgen.py <file-path> -r <rows> -c <cols>

Other optional flags:
    -f # Will overwrite existing instance files if a file already exists at the path. Defaults to false
    -p <name> # Where <name> == "NEMA8" or "FIXED4". Determines the phasing scheme used. Defaults to NEMA8.
    -l <number> # Forces left turns to have some probability of having higher demand than thorugh turns. Defaults to 0

Other parameters of the network may be changed by modifying the arguments to the generate_grid function in netgen.py
 (or modifying the code for even lower-level details!).
