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

package mxnet


import org.apache.mxnet._
import org.apache.mxnet.infer.Classifier
import org.kohsuke.args4j.{CmdLineException, CmdLineParser, Option}

import collection.JavaConverters._

/**
  * Benchmark resnet18 and resnet_end_to_end model for single / batch inference
  * and CPU / GPU
  */
object EndToEndModelWoPreprocessing {

  @Option(name = "--model-e2e-path-prefix", usage = "input model directory and prefix of the model")
  val modelPathPrefixE2E: String = "../models/end_to_end_model/resnet18_end_to_end"
  @Option(name = "--model-non_e2e-path-prefix", usage = "input model directory and prefix of the model")
  val modelPathPrefixNonE2E: String = "../models/not_end_to_end_model/resnet18_v1"
  @Option(name = "--num-runs", usage = "Number of runs")
  val numOfRuns: Int = 1
  @Option(name = "--batchsize", usage = "batch size")
  val batchSize: Int = 25
  @Option(name = "--use-batch", usage = "flag to use batch inference")
  val isBatch: Boolean = false
  @Option(name = "--warm-up", usage = "warm up iteration")
  val timesOfWarmUp: Int = 5

  // process the image explicitly Resize -> ToTensor -> Normalize
  def preprocessImage(nd: NDArray, isBatch: Boolean): NDArray = {
    ResourceScope.using() {
      var resizedImg: NDArray = null
      if (isBatch) {
        val arr: Array[NDArray] = new Array[NDArray](nd.shape.get(0))
        for (i <- 0 until nd.shape.get(0)) {
          arr(i) = Image.imResize(nd.at(i), 224, 224)
        }
        resizedImg = NDArray.api.stack(arr, Some(0), arr.length)
      } else {
        resizedImg = Image.imResize(nd, 224, 224)
      }

      resizedImg = NDArray.api.cast(resizedImg, "float32")
      resizedImg /= 255
      var totensorImg: NDArray = null
      if (isBatch) {
        totensorImg = NDArray.api.swapaxes(resizedImg, Some(1), Some(3))
      } else {
        totensorImg = NDArray.api.swapaxes(resizedImg, Some(0), Some(2))
      }
      val preprocessedImg = (totensorImg - 0.456) / 0.224

      preprocessedImg
    }
  }

  def printAvg(inferenceTimesRaw: Array[Long], metricsPrefix: String, timesOfWarmUp: Int): Unit = {
    // remove warmup
    val inferenceTimes = inferenceTimesRaw.slice(timesOfWarmUp, inferenceTimesRaw.length)
    var sum: Long = 0
    for (time <- inferenceTimes) {
      sum += time
    }
    val average = sum / (inferenceTimes.length * 1.0e6)
    println(f"$metricsPrefix%s_average $average%1.2fms")
  }

  def main(args: Array[String]): Unit = {

    val parser = new CmdLineParser(EndToEndModelWoPreprocessing)

    try {
      parser.parseArgument(args.toList.asJava)
    } catch {
      case e: CmdLineException =>
        print(s"Error:${e.getMessage}\n Usage:\n")
        parser.printUsage(System.out)
        System.exit(1)
    }

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }

    val inputShapeE2E = Shape(1, 300, 300, 3)
    val inputDescriptorE2E = IndexedSeq(DataDesc("data", inputShapeE2E, DType.UInt8, "NHWC"))
    val inputShapeNonE2E = Shape(1, 3, 224, 224)
    val inputDescriptorNonE2E = IndexedSeq(DataDesc("data", inputShapeNonE2E, DType.Float32, "NCHW"))

    val classifierE2E = new
        Classifier(modelPathPrefixE2E, inputDescriptorE2E, context)

    val classifierNonE2E = new
        Classifier(modelPathPrefixNonE2E, inputDescriptorNonE2E, context)

    val currTimeE2E: Array[Long] = Array.fill(numOfRuns + timesOfWarmUp){0}
    val currTimeNonE2E: Array[Long] = Array.fill(numOfRuns + timesOfWarmUp){0}
    val timesE2E: Array[Long] = Array.fill(numOfRuns + timesOfWarmUp){0}
    val timesNonE2E: Array[Long] = Array.fill(numOfRuns + timesOfWarmUp){0}

    for (n <- 0 until numOfRuns + timesOfWarmUp) {
      ResourceScope.using() {
        var nd:NDArray = null
        if (isBatch) {
          nd = NDArray.api.random_uniform(Some(0), Some(255), Some(Shape(batchSize, 300, 300, 3)))
        } else {
          nd = NDArray.api.random_uniform(Some(0), Some(255), Some(Shape(300, 300, 3)))
        }

        val img = NDArray.api.cast(nd, "uint8")
        // E2E
        currTimeE2E(n) = System.nanoTime()
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
          System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
          img.asInContext(Context.gpu())
        }
        var imgWithBatchNumE2E:NDArray = null
        if (isBatch) {
          imgWithBatchNumE2E = img
        } else {
          imgWithBatchNumE2E = NDArray.api.expand_dims(img, 0)
        }

        val outputE2E = classifierE2E.classifyWithNDArray(IndexedSeq(imgWithBatchNumE2E), Some(5))
        timesE2E(n) = System.nanoTime() - currTimeE2E(n)

        // Non E2E
        // If the img is in GPU, copy back to CPU for preprocess
        img.asInContext(Context.cpu())
        currTimeNonE2E(n) = System.nanoTime()
        val preprocessedImage = preprocessImage(img, isBatch)
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
          System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
          preprocessedImage.asInContext(Context.gpu())
        }
        var imgWithBatchNumNonE2E: NDArray = null
        if (isBatch) {
          imgWithBatchNumNonE2E = preprocessedImage
        } else {
          imgWithBatchNumNonE2E = NDArray.api.expand_dims(preprocessedImage, 0)
        }
        val outputNonE2E = classifierNonE2E.classifyWithNDArray(IndexedSeq(imgWithBatchNumNonE2E), Some(5))
        timesNonE2E(n) = System.nanoTime() - currTimeNonE2E(n)
      }
    }
    println("E2E")
    printAvg(timesE2E, if (isBatch) "batch_inference" else "single_inference", timesOfWarmUp)
    println("Non E2E")
    printAvg(timesNonE2E, if (isBatch) "batch_inference" else "single_inference", timesOfWarmUp)
  }
}