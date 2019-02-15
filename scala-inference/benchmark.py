import argparse
import re
import subprocess
import sys
import os
import mxnet as mx


CURRE_DIR = os.getcwd()
CLASSPATH= '$CLASSPATH:{}/target/*:$CLASSPATH:{}/target/classes/lib/*'.format(CURRE_DIR, CURRE_DIR)
SCALA_VERSION_PROFILE='2.11'
MXNET_VERSION='[1.5.0-SNAPSHOT,)'

def download_model(model_name, model_path):
    model_json_path = "{}/{}-symbol.json".format(model_path, model_name)
    model_params_path = "{}/{}-0000.params".format(model_path, model_name)
    print("Downloading the following model files...")
    print(model_json_path)
    print(model_params_path)
    try:
        mx.test_utils.download(model_json_path)
        mx.test_utils.download(model_params_path)
    except Exception as e:
        print("ERROR: Failed to download the models. {}".format(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a E2E bechmark task.")
    parser.add_argument('--model-path', type=str, help='URL to download the model')
    parser.add_argument('--model-name', type=str, help='Name of the model. This will be used to download the right symbol and params files from model_path')
    parser.add_argument('--iterations', type=int, help='Number of times to run the benchmark')
    parser.add_argument('--use_gpus', type=int, help='Number of gpu the benchmark will use')
    parser.add_argument('--end_to_end', type=bool, help='flag to benchmark end to end model or non end to end')

    args = parser.parse_args()
    # prepare the model
    # download_model(args.model_name, args.model_path)
    # setup the maven
    # hw_type = 'gpu' if int(args.use_gpus) > 0 else 'cpu'
    # subprocess.run(['./mvnw', 'clean install dependency:copy-dependencies package -Dmxnet.hw_type={} -Dmxnet.scalaprofile={} -Dmxnet.version={}'.format(hw_type, SCALA_VERSION_PROFILE, MXNET_VERSION)])
    
    sum_result = 0.0
    # the defualt value is 20 so tha we have enough CPU and GPU memory
    num_iter_batch = 20 if args.num_runs > 20 else args.num_runs
    num_iter = args.num_runs // num_iter_batch if args.num_runs > num_iter_batch else 1
    for i in range(num_iter):
        output = subprocess.check_output('java -Xmx8G  -cp {} '
        'mxnet.EndToEndModelWoPreprocessing '
        '--model-path-prefix {} '
        '--num-runs {}'
        '--batchsize {}'
        '--warm-up {}'
        ' {}'.format(CLASSPATH, args.model_path, args.num_runs, 1, 5, '--end_to_end' if args.end_to_end else ''),
        stderr=subprocess.STDOUT,
        shell=True).decode(sys.stdout.encoding)
        res = re.search('(E2E| Non E2E)\n(single|batch)_inference_average (\d+.\d+)ms', output)
        sum_result += float(res.group(3))

    print('{} {}_inference_average {}'
        .format(res.group(1), res.group(2), sum_result / num_iter))