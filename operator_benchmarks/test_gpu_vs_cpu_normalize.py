import numpy as np
import mxnet as mx
from mxnet import nd
import time
from mxnet.gluon.data.vision import transforms

gpu_total_time = 0.0
transformer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(.229, 0.224, 0.225))
count = 500

for i in range(count):
    image = mx.nd.random.uniform(0, 255, (1, 3, 512,512), ctx=mx.gpu(0))
    tic = time.time()
    #image.as_in_context(mx.gpu(0))
    res = transformer(image)
    # To force the calculation
    #a = res.shape
    res.wait_to_read()
    tac = time.time()
    gpu_total_time += (tac - tic) * 1000

print("Total time for GPU ToTensor - ", gpu_total_time)
print("Average time per Normalize 1,3,512,512 - ", gpu_total_time/count)

cpu_total_time = 0.0
transformer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(.229, 0.224, 0.225))
count = 1000

for i in range(count):
    image = mx.nd.random.uniform(0, 255, (1, 3, 512,512))
    tic = time.time()
    #image.as_in_context(mx.gpu(0))
    res = transformer(image)
    res.wait_to_read()
    tac = time.time()
    cpu_total_time += (tac - tic) * 1000

print("Total time for CPU ToTensor - ", cpu_total_time)
print("Average time per Normalize 1,3,512,512 - ", cpu_total_time/count)
