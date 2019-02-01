# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mxnet as mx
import numpy as np
import os
import json


class GluonBaseService(object):
    """GluonBaseService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-5 labels are returned.
    """

    def __init__(self):
        self.param_filename = None
        self.model_name = None
        self.initialized = False
        self.ctx = None
        self.net = None
        self._signature = None
        self.labels = None
        self.signature = None

    def initialize(self, params):
        """
        Initialization of the network
        :param params: This is the :func `Context` object
        :return:
        """
        if self.net is None:
            raise NotImplementedError("Gluon network not defined")
        sys_prop = params.system_properties
        gpu_id = sys_prop.get("gpu_id")
        model_dir = sys_prop.get("model_dir")
        self.model_name = params.manifest["model"]["modelName"]
        self.ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)

        if self.param_filename is not None:
            param_file_path = os.path.join(model_dir, self.param_filename)
            if not os.path.isfile(param_file_path):
                raise OSError("Parameter file not found {}".format(param_file_path))
            self.net.load_parameters(param_file_path, self.ctx)

        synset_file = os.path.join(model_dir, "synset.txt")
        signature_file_path = os.path.join(model_dir, "signature.json")

        if not os.path.isfile(signature_file_path):
            raise OSError("Signature file not found {}".format(signature_file_path))

        if not os.path.isfile(synset_file):
            raise OSError("synset file not available {}".format(synset_file))

        with open(signature_file_path) as sig_file:
            self.signature = json.load(sig_file)

        self.labels = [line.strip() for line in open(synset_file).readlines()]
        self.initialized = True

    def preprocess(self, data):
        """
        This method considers only one input data

        :param data: Data is list of map
        format is
        [
        {
            "parameterName": name
            "parameterValue": data
        },
        {...}
        ]
        :return:
        """

        param_name = self.signature['inputs'][0]['data_name']
        input_shape = self.signature['inputs'][0]['data_shape']

        img = data[0].get(param_name)
        if img is None:
            img = data[0].get("body")
        if img is None:
            img = data[0].get("data")
        if img is None or len(img) == 0:
            raise IOError("Invalid parameter given")

        # We are assuming input shape is NCHW
        [h, w] = input_shape[2:]
        img_arr = mx.img.imdecode(img)
        img_arr = mx.image.imresize(img_arr, w, h)
        img_arr = img_arr.astype(np.float32)
        img_arr /= 255
        img_arr = mx.image.color_normalize(img_arr,
                                           mean=mx.nd.array([0.485, 0.456, 0.406]),
                                           std=mx.nd.array([0.229, 0.224, 0.225]))
        img_arr = mx.nd.transpose(img_arr, (2, 0, 1))
        img_arr = img_arr.expand_dims(axis=0)
        return img_arr

    def inference(self, data):
        """
        Internal inference methods for MMS service. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
               Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        """
        model_input = data.as_in_context(self.ctx)
        output = self.net(model_input)
        return output.softmax()

    def postprocess(self, data):
        assert hasattr(self, 'labels'), \
            "Can't find labels attribute. Did you put synset.txt file into " \
            "model archive or manually load class label file in __init__?"
        return [[top_probability(d, self.labels, top=5) for d in data]]

    def predict(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        return self.postprocess(data)


def top_probability(data, labels, top=5):
    """
    Get top probability prediction from NDArray.

    :param data: NDArray
        Data to be predicted
    :param labels: List
        List of class labels
    :param top:
    :return: List
        List of probability: class pairs in sorted order
    """
    dim = len(data.shape)
    if dim > 2:
        data = mx.nd.array(
            np.squeeze(data.asnumpy(), axis=tuple(range(dim)[2:])))
    sorted_prob = mx.nd.argsort(data[0], is_ascend=False)
    # pylint: disable=deprecated-lambda
    top_prob = map(lambda x: int(x.asscalar()), sorted_prob[0:top])
    return [{'probability': float(data[0, i].asscalar()), 'class': labels[i]}
            for i in top_prob]


"""
Gluon Pretrained Resnet model
"""


class PretrainedResnetService(GluonBaseService):
    """
    Pretrained Resnet Service
    """

    def initialize(self, params):
        """
        Initialize the model
        :param params: This is the same as the Context object
        :return:
        """
        sys_prop = params.system_properties
        gpu_id = sys_prop.get("gpu_id")
        ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)
        self.net = mx.gluon.model_zoo.vision.resnet18_v1(pretrained=True, ctx=ctx)
        super(PretrainedResnetService, self).initialize(params)

    def postprocess(self, data):
        """
        Post process for the Gluon Resnet model
        :param data:
        :return:
        """
        idx = data.topk(k=5)[0]
        return [[{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
            float(data[0, int(i.asscalar())].asscalar())} for i in idx]]


svc = PretrainedResnetService()


def pretrained_gluon_resnet(data, params):
    """
    This is the handler that needs to be registerd in the model-archive.
    :param data:
    :param params:
    :return:
    """
    res = None
    if not svc.initialized:
        svc.initialize(params)

    if data is not None:
        res = svc.predict(data)

    return res
