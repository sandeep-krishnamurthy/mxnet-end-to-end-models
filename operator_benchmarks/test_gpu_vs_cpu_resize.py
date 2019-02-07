import numpy as np
import mxnet as mx
from mxnet import nd
import time
from mxnet.gluon.data.vision import transforms

gpu_total_time = 0.0
transformer = transforms.Resize(size=(224, 224))
count = 1000

for i in range(count):
    image = mx.nd.random.uniform(0, 255, (1, 512,512,3), ctx=mx.gpu(0))
    tic = time.time()
    #image.as_in_context(mx.gpu(0))
    res = transformer(image)
    # To force the calculation
    #a = res.shape
    res.wait_to_read()
    tac = time.time()
    gpu_total_time += (tac - tic) * 1000

print("Total time for GPU resize - ", gpu_total_time)
print("Average time per resize 1,512,512,3 to 1,224,224,3 - ", gpu_total_time/count)

cpu_total_time = 0.0
transformer = transforms.Resize(size=(224, 224))
count = 1000

for i in range(count):
    image = mx.nd.random.uniform(0, 255, (1, 512,512,3))
    tic = time.time()
    #image.as_in_context(mx.gpu(0))
    res = transformer(image)
    res.wait_to_read()
    tac = time.time()
    cpu_total_time += (tac - tic) * 1000

print("Total time for CPU resize - ", cpu_total_time)
print("Average time per resize 1,512,512,3 to 1,224,224,3 - ", cpu_total_time/count)
