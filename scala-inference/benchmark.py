import subprocess
import re
import sys


sum_e2e = 0.0
sum_non_e2e = 0.0
num_iter = 1
for i in range(num_iter):
    output = subprocess.check_output('bash bin/run_benchmark.sh',
    stderr=subprocess.STDOUT,
    shell=True).decode(sys.stdout.encoding)
    res = re.search('E2E\n(single|batch)_inference_average (\d+.\d+)ms\nNon E2E\n(single|batch)_inference_average (\d+.\d+)ms', output)
    sum_e2e += float(res.group(2))
    sum_non_e2e += float(res.group(4))
print('E2E {} Non E2E {}'.format(sum_e2e / num_iter, sum_non_e2e / num_iter))