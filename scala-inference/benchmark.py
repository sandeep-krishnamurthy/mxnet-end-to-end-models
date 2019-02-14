import argparse
import re
import subprocess
import sys
import os


CURRE_DIR = os.getcwd()
CLASSPATH= '$CLASSPATH:{}/target/*:$CLASSPATH:{}/target/classes/lib/*'.format(CURRE_DIR, CURRE_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a E2E bechmark task.")
    parser.add_argument('--num-runs', type=int, help='Number of times to run the benchmark')

    args = parser.parse_args()

    # single inference
    output = subprocess.check_output('java -Xmx8G  -cp {} '
        'mxnet.EndToEndModelWoPreprocessing '
        '--num-runs {} '
        '--use-batch {}'.format(CLASSPATH, args.num_runs, 'false'),
        stderr=subprocess.STDOUT,
        shell=True).decode(sys.stdout.encoding)
    res = re.search('E2E\nsingle_inference_average (\d+.\d+)ms\nNon E2E\nsingle_inference_average (\d+.\d+)ms', output)
    print('E2E single_inference_average {} Non E2E single_inference_average {}'.format(res.group(1), res.group(2)))

    # batch inference
    sum_e2e = 0.0
    sum_non_e2e = 0.0
    num_iter_batch = 20 if args.num_runs > 20 else args.num_runs
    num_iter = args.num_runs // num_iter_batch if args.num_runs > num_iter_batch else 1
    print(num_iter)
    print(num_iter_batch)
    for i in range(num_iter):
        output = subprocess.check_output('java -Xmx8G  -cp {} '
        'mxnet.EndToEndModelWoPreprocessing '
        '--num-runs {} '
        '--use-batch {}'.format(CLASSPATH, num_iter_batch, 'true'),
        stderr=subprocess.STDOUT,
        shell=True).decode(sys.stdout.encoding)
        res = re.search('E2E\nbatch_inference_average (\d+.\d+)ms\nNon E2E\nbatch_inference_average (\d+.\d+)ms', output)
        sum_e2e += float(res.group(1))
        sum_non_e2e += float(res.group(2))
    print('E2E batch_inference_average {} Non E2E batch_inference_average {}'.format(sum_e2e / num_iter, sum_non_e2e / num_iter))