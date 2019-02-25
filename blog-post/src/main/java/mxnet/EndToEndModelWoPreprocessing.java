/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package mxnet;

import org.apache.mxnet.ResourceScope;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.util.*;
import java.io.IOException;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;

public class EndToEndModelWoPreprocessing {
    static NDArray$ NDArray = NDArray$.MODULE$;

    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = "resnet18_v1";
    @Option(name = "--use-gpu", usage = "use gpu or cpu")
    private boolean useGPU = false;

    private static String getClassName(double[] scores,
                                            String modelPathPrefix) throws IOException {                                     
        String synsetFilePath = modelPathPrefix.substring(0,
                1 + modelPathPrefix.lastIndexOf(File.separator)) + "/synset.txt";
        BufferedReader reader = new BufferedReader(new FileReader(synsetFilePath));
        ArrayList<String> list = new ArrayList<>();
        String line = reader.readLine();

        while (line != null){
            list.add(line);
            line = reader.readLine();
        }
        reader.close();

        int maxIdx = 0;
        for (int i = 1;i < scores.length; i++) {
            if (scores[i] > scores[maxIdx]) {
                maxIdx = i;
            }
        }

        return "Class : " + list.get(maxIdx) ;
    }

    private static void runInference(String modelPathPrefix, List<Context> context) {
        Shape inputShape;
        List<DataDesc> inputDescriptors = new ArrayList<>();
        inputShape = new Shape(new int[]{1, 576, 1024, 3});
        inputDescriptors.add(new DataDesc("data", inputShape, DType.UInt8(), "NHWC"));
        Predictor predictor = new Predictor(modelPathPrefix, inputDescriptors, context, 0);

        try(ResourceScope scope = new ResourceScope()) {
            NDArray img = Image.imRead("Pug-Cookie.jpg", 1, true);
            img = NDArray.expand_dims(img, 0, null)[0];    
            img.asInContext(context.get(0));
            List<NDArray> input = new ArrayList<>();
            input.add(img);
            List<NDArray> output = predictor.predictWithNDArray(input);
            output.get(0).waitToRead();
            try {
                System.out.println(getClassName(output.get(0).toFloat64Array(), modelPathPrefix));
            } catch (IOException e) {
                System.err.println(e);
            }
        }
    }

    public static void main(String[] args) {
        EndToEndModelWoPreprocessing inst = new EndToEndModelWoPreprocessing();
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            parser.printUsage(System.err);
            System.exit(1);
        }

        List<Context> context = new ArrayList<Context>();
        context.add((inst.useGPU) ? Context.gpu() : Context.cpu());

        runInference(inst.modelPathPrefix, context);
    }
}
