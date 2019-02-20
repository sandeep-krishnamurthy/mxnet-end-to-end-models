#!/bin/bash

set -ex

CURR_DIR=$(pwd)
CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/dependency/*:$CLASSPATH:$CURR_DIR/target/classes/lib/*

java -Xmx8G  -cp $CLASSPATH mxnet.EndToEndModelWoPreprocessing \
--model-path-prefix models/resnet18_v1_end_to_end \
--num-runs 1 \
--batchsize 1 \
--warm-up 0 \
--end-to-end
