#!/bin/bash
# usage: source runprost.sh <container name> <rddl path> <rounds> <prost arguments> <destination>

# run the container
docker run --name $1 --mount type=bind,source=$2,target=/RDDL prost $3 $4
	
# copy the log files from container to local system
docker cp $1:/OUTPUTS/ $5
