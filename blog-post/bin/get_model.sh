#!/bin/bash

set -ex

wget https://s3.us-east-2.amazonaws.com/mxnet-public/end_to_end_models/resnet18_v1_end_to_end-symbol.json
wget https://s3.us-east-2.amazonaws.com/mxnet-public/end_to_end_models/resnet18_v1_end_to_end-0000.params
wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/synset.txt
wget https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg