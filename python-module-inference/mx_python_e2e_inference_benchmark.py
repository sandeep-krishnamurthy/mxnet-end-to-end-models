# # Predict with pre-trained models

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import mxnet as mx
import cv2
from collections import namedtuple

import os
os.environ['MXNET_EXEC_ENABLE_INPLACE']='0'

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
class InferenceTesting(object):
    def __init__(self, opt):
        self.input_shape = (224, 224)
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (.229, 0.224, 0.225)

        self.model_path = opt.model_path
        self.model_name = opt.model_name
        self.iterations = opt.iterations
        self.use_gpus = opt.use_gpus
        self.pre_process = False #opt.preprocess

    def preprocess_data(self, data):
        """
        This method considers only one input data

        :param data: NDArray input
        """


        # We are assuming input shape is NCHW
        data = mx.image.imresize(data, self.input_shape[0], self.input_shape[1])
        data = data.astype(np.float32)
        data /= 255
        data = mx.image.color_normalize(data,
                                           mean=mx.nd.array([0.485, 0.456, 0.406]),
                                           std=mx.nd.array([0.229, 0.224, 0.225]))
        data = mx.nd.transpose(data, (2, 0, 1))
        data = data.expand_dims(axis=0)
        return data

    def downloadModel(self):
        # TODO - Sandeep - If model_path ends with / don't include it below.
        model_json_path = "{}/{}-symbol.json".format(self.model_path, self.model_name)
        model_params_path = "{}/{}-0000.params".format(self.model_path, self.model_name)
        model_synset_path = "{}/synset.txt".format(self.model_path, self.model_name)
        print("Downloading following model files")
        print(model_synset_path)
        print(model_json_path)
        print(model_params_path)

        try:
            mx.test_utils.download(model_json_path)
            mx.test_utils.download(model_params_path)
            mx.test_utils.download(model_synset_path)
        except Exception as e:
            print("Error in downloading the models {}".format(e))

    def __loadModel(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.model_name, 0)
        cont = mx.gpu() if self.use_gpus > 0 else mx.cpu()
        self.mod = mx.mod.Module(symbol=sym, context=cont, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 224, 224, 3))],
                 label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)

    def predict(self):
        self.__loadModel()
        total_time_ms = 0.0
        # compute the predict probabilities
        for i in range(self.iterations):
            # Generate a synthetic data
            image = mx.nd.random.uniform(0, 255, (1, 512, 512, 3)).astype(dtype=np.uint8)
            #print("Raw data shape - ", image.shape)
            tic = time.time()
            if self.pre_process:
                print("Preprocessing")
                image = self.preprocess_data(image)
            self.mod.forward(Batch([image]))
            res = self.mod.get_outputs()[0].asnumpy()
            pred_time_ms = (time.time() - tic) * 1000
            total_time_ms += pred_time_ms

        print ("Total Prediction-Time: {} milliseconds".format(total_time_ms))
        print ("Average Prediction-Time: {} milliseconds".format(total_time_ms/self.iterations))

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference speed test for image classification.')
    parser.add_argument('--model_path', type=str,
                        help='Path to download the model')
    parser.add_argument('--model_name', type=str, default='resnet18_v1',
                        help='Name of the model. This will be used to download the right symbol and params files from model_path')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of times inference is generated for a single image.')
    parser.add_argument('--use_gpus', type=int, default=0,
                        help="Indicate whether to use gpus")
    parser.add_argument('--preprocess', type=bool, default=True, help="To do pre-processing or not")
    opt = parser.parse_args()

    print(opt)
                                                                                                                                                     106,1         85%
    # Following sleep is added so that process runs until cpu-gpu profiler process starts.
    infer = InferenceTesting(opt)
    #infer.downloadModel()
    infer.predict()
    time.sleep(10)
    print ("Done")
    exit()

