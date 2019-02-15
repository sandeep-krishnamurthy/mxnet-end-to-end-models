# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#!/bin/bash
set -ex

hw_type=cpu
if [[ $1 = gpu ]]
then
    hw_type=gpu
fi

SCALA_VERSION_PROFILE=2.11
MXNET_VERSION="[1.5.0-SNAPSHOT,)"

# use maven wrapper to avoid dead lock when using the apt install maven but sudo apt update still running on background
# ./mvnw clean install dependency:copy-dependencies package -Dmxnet.hw_type=$hw_type -Dmxnet.scalaprofile=$SCALA_VERSION_PROFILE -Dmxnet.version=$MXNET_VERSION

CURR_DIR=$(cd $(dirname $0)/../; pwd)

CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/dependency/*:$CLASSPATH:$CURR_DIR/target/classes/lib/*

java -Xmx8G  -cp $CLASSPATH mxnet.EndToEndModelWoPreprocessing \
--model-path-prefix ../models/end_to_end_model/resnet18_end_to_end \
--num-runs 5 \
--batchsize 25 \
--warm-up 2 \
--end-to-end
