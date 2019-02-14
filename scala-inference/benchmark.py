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
    parser.add_argument('--batch-size', type=int, help='Number of times to run the benchmark')

    args = parser.parse_args()
    # prepare the model
    download_model(args.model_name, args.model_path)
    # setup the maven
    
    # single inference
    output = subprocess.check_output('java -Xmx8G  -cp {} '
        'mxnet.EndToEndModelWoPreprocessing '
        '--num-runs {} '
        '--use-batch {}'
        '--batchsize {}'.format(CLASSPATH, args.iterations, 'false', args.batch_size),
        stderr=subprocess.STDOUT,
        shell=True).decode(sys.stdout.encoding)
    single_res = re.search('E2E\nsingle_inference_average (\d+.\d+)ms\nNon E2E\nsingle_inference_average (\d+.\d+)ms', output)

    # batch inference
    sum_e2e = 0.0
    sum_non_e2e = 0.0
    # the defualt value is 20 so tha we have enough CPU and GPU memory
    num_iter_batch = 20 if args.num_runs > 20 else args.num_runs
    num_iter = args.num_runs // num_iter_batch if args.num_runs > num_iter_batch else 1
    print(num_iter)
    print(num_iter_batch)
    for i in range(num_iter):
        output = subprocess.check_output('java -Xmx8G  -cp {} '
        'mxnet.EndToEndModelWoPreprocessing '
        '--num-runs {} '
        '--use-batch {}'
        '--batchsize {}'.format(CLASSPATH, args.iterations, 'false', args.batch_size),
        stderr=subprocess.STDOUT,
        shell=True).decode(sys.stdout.encoding)
        res = re.search('E2E\nbatch_inference_average (\d+.\d+)ms\nNon E2E\nbatch_inference_average (\d+.\d+)ms', output)
        sum_e2e += float(res.group(1))
        sum_non_e2e += float(res.group(2))

    print('E2E single_inference_average {} Non E2E single_inference_average {} '
        'E2E batch_inference_average {} Non E2E batch_inference_average {}'
        .format(single_res.group(1), single_res.group(2), sum_e2e / num_iter, sum_non_e2e / num_iter))