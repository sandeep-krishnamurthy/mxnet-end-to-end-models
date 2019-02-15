#!/bin/bash
set -e

if [[ $2 = gpu ]]
then
    hw_type=gpu
    use_gpu="--use-gpu"
else
    hw_type=cpu
    use_gpu=""
fi

if [[ $1 = e2e ]]
then
    model_path="../models/end_to_end_model/resnet18_end_to_end"
    end_to_end="--end-to-end"
else
    model_path="../models/not_end_to_end_model/resnet18_v1"
    end_to_end=""
fi

SCALA_VERSION_PROFILE=2.11
MXNET_VERSION="[1.5.0-SNAPSHOT,)"

# use maven wrapper to avoid dead lock when using the apt install maven but sudo apt update still running on background
# ./mvnw clean install dependency:copy-dependencies package -Dmxnet.hw_type=$hw_type -Dmxnet.scalaprofile=$SCALA_VERSION_PROFILE -Dmxnet.version=$MXNET_VERSION

CURR_DIR=$(cd $(dirname $0)/../; pwd)

CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/dependency/*:$CLASSPATH:$CURR_DIR/target/classes/lib/*

output_single=$(java -Xmx8G  -cp $CLASSPATH mxnet.EndToEndModelWoPreprocessing \
--model-path-prefix $model_path \
--num-runs $3 \
--batchsize 1 \
--warm-up 5 \
$end_to_end \
$use_gpu)

sum=0.0
count=0
# the defualt value is 20 so tha we have enough CPU and GPU memory
num_iter=$(echo $3/25 | bc)
if [ $num_iter = 0 ]; then num_iter=1; fi
for n in `seq 1 $num_iter`
do
    output_batch=$(java -Xmx8G  -cp $CLASSPATH mxnet.EndToEndModelWoPreprocessing \
    --model-path-prefix $model_path \
    --num-runs 25 \
    --batchsize 25 \
    --warm-up 1 \
    $end_to_end \
    $use_gpu)
    value=$(echo $output_batch | ggrep -oP '(E2E|Non E2E) (single|batch)_inference_average \K(\d+.\d+)(?=ms)')
    sum=$(echo $sum+$value | bc)
    ((count++))
    echo $value
done

metrix=$(echo $output_batch | ggrep -oE '(single|batch)_inference_average')
echo "$output_single $metrix $(echo "scale=2; $sum/$count" | bc)ms"
