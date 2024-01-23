#!/bin/bash

# Confirm 2 parameters
if [ $# -ne 2 ]; then
    echo
    echo "  Usage: prost.sh <rounds> <arguments>"
	echo "  Given: <$@> of size $#"
    echo
    exit 1
fi

echo "Starting RDDL gym server..."
( python rddlsim.py "$1" ) > $PROST_OUT/rddlsim.log 2>&1 &
sleep 5

echo "Starting PROST with arguments $2..."
( cd $WORKSPACE/prost && ./prost.py domain.rddl "$2" ) > $PROST_OUT/prost.log 2>&1
