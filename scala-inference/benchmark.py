import subprocess
import re
import sys


sum_e2e = 0.0
sum_non_e2e = 0.0
for i in range(10):
    output = subprocess.check_output('bash bin/run_im.sh',
    stderr=subprocess.STDOUT,
    shell=True).decode(sys.stdout.encoding)
    res = re.search('E2E\nsingle_inference_average (\d+.\d+)ms\nNon E2E\nsingle_inference_average (\d+.\d+)ms', output)
    sum_e2e += float(res.group(1))
    sum_non_e2e += float(res.group(2))
print('E2E {} Non E2E {}'.format(sum_e2e / 10.0, sum_non_e2e / 10.0))