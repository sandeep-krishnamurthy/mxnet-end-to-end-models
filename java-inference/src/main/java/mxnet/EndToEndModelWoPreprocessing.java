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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * This Class is a demo to show how users can use Predictor APIs to do
 * Image Classification with all hand-crafted Pre-processing.
 * All helper functions for image pre-processing are
 * currently available in ObjectDetector class.
 */
public class EndToEndModelWoPreprocessing {
    static NDArray$ NDArray = NDArray$.MODULE$;

    @Option(name = "--model-e2e-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefixE2E = "model/resnet18_end_to_end";
    @Option(name = "--model-non_e2e-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefixNonE2E = "model/resnet18_v1";
    @Option(name = "--input-image", usage = "the input image")
    private String inputImagePath = "/images/dog.jpg";
    @Option(name = "--input-dir", usage = "the input batch of images directory")
    private String inputImageDir = "/images/";
    @Option(name = "--num-runs", usage = "Number of runs")
    private String numRuns = "1";
    @Option(name = "--batchsize", usage = "batch size")
    private String batchsize = "1";

    private static NDArray preprocessImage(NDArray nd) {
        NDArray resizeImg = Image.imResize(nd, 224, 224);
        resizeImg = NDArray.cast(resizeImg, "float32", null)[0];
        resizeImg = resizeImg.divInplace(255.0);
        NDArray totensorImg = (NDArray.swapaxes(NDArray.new swapaxesParam(resizeImg).setDim1(0).setDim2(2)))[0];
        totensorImg = totensorImg.divInplace(0.456);
        NDArray preprocessedImg = totensorImg.divInplace(0.224);

        return preprocessedImg;
    }

    private static void printStatistics(long[] inferenceTimesRaw, String metricsPrefix)  {
        long[] inferenceTimes = inferenceTimesRaw;
        // remove head and tail
        if (inferenceTimes.length > 2) {
            inferenceTimes = Arrays.copyOfRange(inferenceTimesRaw,
                    1, inferenceTimesRaw.length - 1);
        }
        long sum = 0;
        for (long time: inferenceTimes) sum += time;
        double average = sum / (inferenceTimes.length * 1.0e6);

        System.out.println(String.format("%s_average %1.2fms",metricsPrefix, average));
    }

    /**
     * Helper class to print the maximum prediction result
     * @param probabilities The float array of probability
     * @param modelPathPrefix model Path needs to load the synset.txt
     */
    private static String printMaximumClass(float[] probabilities,
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
        for (int i = 1;i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIdx]) {
                maxIdx = i;
            }
        }

        return "Probability : " + probabilities[maxIdx] + " Class : " + list.get(maxIdx) ;
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

        int batchSize = Integer.parseInt(inst.batchsize);
        int numberOfRuns = Integer.parseInt(inst.numRuns);

        String imgPath = inst.inputImagePath;
        String imgDir = inst.inputImageDir;

        // Prepare the model
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }

        Shape inputShapeE2E = new Shape(new int[]{1, 300, 300, 3});
        Shape inputShapeNonE2E = new Shape(new int[]{1, 3, 224, 224});
        List<DataDesc> inputDescriptorsE2E = new ArrayList<>();
        List<DataDesc> inputDescriptorsNonE2E = new ArrayList<>();
        inputDescriptorsE2E.add(new DataDesc("data", inputShapeE2E, DType.UInt8(), "NHWC"));
        inputDescriptorsNonE2E.add(new DataDesc("data", inputShapeNonE2E, DType.Float32(), "NCHW"));
        Predictor predictorE2E = new Predictor(inst.modelPathPrefixE2E, inputDescriptorsE2E, context,0);
        Predictor predictorNonE2E = new Predictor(inst.modelPathPrefixNonE2E, inputDescriptorsNonE2E, context,0);

        long[] currTimeE2E = new long[numberOfRuns];
        long[] currTimeNonE2E = new long[numberOfRuns];
        long[] timesE2E = new long[numberOfRuns];
        long[] timesNonE2E = new long[numberOfRuns];

        for (int n = 0; n < numberOfRuns; n++) {
            try (ResourceScope scope = new ResourceScope()) {
                NDArray nd = NDArray.random_uniform(
                        NDArray.new random_uniformParam()
                                .setLow(0f)
                                .setHigh(255f)
                                .setShape(new Shape(new int[]{300, 300, 3})))[0];
                NDArray img = NDArray.cast(nd, "uint8", null)[0];
                //E2E
                currTimeE2E[n] = System.nanoTime();
                if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                        Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
                    img.asInContext(Context.gpu());
                }
                NDArray imgWithBatchNumE2E = NDArray.expand_dims(img, 0, null)[0];
                List<NDArray> inputE2E = new ArrayList<>();
                inputE2E.add(imgWithBatchNumE2E);

                List<NDArray> resE2E = predictorE2E.predictWithNDArray(inputE2E);
                resE2E.get(0).waitToRead();
                timesE2E[n] = System.nanoTime() - currTimeE2E[n];

                // Non E2E
                img.asInContext(Context.cpu());
                currTimeNonE2E[n] = System.nanoTime();
                NDArray preprocessedImage = preprocessImage(img);
                if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                        Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
                    preprocessedImage.asInContext(Context.gpu());
                }
                NDArray imgWithBatchNumNonE2E = NDArray.expand_dims(preprocessedImage, 0, null)[0];
                List<NDArray> inputNonE2E = new ArrayList<>();
                inputNonE2E.add(imgWithBatchNumNonE2E);
                List<NDArray> resNonE2E = predictorNonE2E.predictWithNDArray(inputNonE2E);
                resNonE2E.get(0).waitToRead();
                timesNonE2E[n] = System.nanoTime() - currTimeNonE2E[n];
            }
        }
        System.out.println("E2E");
        printStatistics(timesE2E, "single_inference");
        System.out.println("Non E2E");
        printStatistics(timesNonE2E, "single_inference");
    }
}
