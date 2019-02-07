import numpy as np
import mxnet as mx
from mxnet import nd
import time

total_time = 0.0

for i in range(500):
    image = mx.nd.random.uniform(0, 255, (1, 3, 512,512)).astype(dtype=np.uint8)
    tic = time.time()
    image.as_in_context(mx.gpu(0))
    res = image.asnumpy()
    tac = time.time()
    total_time += (tac - tic) * 1000

print("Total time for 512, 512, 3 - ", total_time)

total_time = 0.0

for i in range(500):
    image = mx.nd.random.uniform(0, 255, (1, 3, 224, 224)).astype(dtype=np.uint8)
    tic = time.time()
    image.as_in_context(mx.gpu(1))
    res = image.asnumpy()
    tac = time.time()
    total_time += (tac - tic) * 1000

print("Total time for 3, 224, 224 - ", total_time)

total_time = 0.0

for i in range(500):
    image = mx.nd.random.uniform(0, 255, (1, 300, 300, 3)).astype(dtype=np.uint8)
    tic = time.time()
    image.as_in_context(mx.gpu(2))
    res = image.asnumpy()
    tac = time.time()
    total_time += (tac - tic) * 1000

print("Total time for 1, 300, 300, 3 - ", total_time)
