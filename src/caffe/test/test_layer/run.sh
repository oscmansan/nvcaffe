#!/bin/bash
PROGRAM=test_layer

make $PROGRAM.bin
 
sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:../../../../.build_release/lib ./$PROGRAM.bin

