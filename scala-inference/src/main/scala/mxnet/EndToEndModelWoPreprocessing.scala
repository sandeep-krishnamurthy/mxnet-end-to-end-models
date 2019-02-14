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
import org.apache.mxnet.infer.Predictor
import org.kohsuke.args4j.{CmdLineException, CmdLineParser, Option}

import collection.JavaConverters._

/**
  * Benchmark resnet18 and resnet_end_to_end model for single / batch inference
  * and CPU / GPU
  */
object EndToEndModelWoPreprocessing {

  @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
  val modelPathPrefix: String = "resnet18_v1"
  @Option(name = "--num-runs", usage = "Number of runs")
  val numOfRuns: Int = 1
  @Option(name = "--batchsize", usage = "batch size")
  val batchSize: Int = 25
  @Option(name = "--end_to_end", usage = "benchmark with e2e / non e2e")
  val isE2E: Boolean = false
  @Option(name = "--warm-up", usage = "warm up iteration")
  val timesOfWarmUp: Int = 5

  // process the image explicitly Resize -> ToTensor -> Normalize
  def preprocessImage(nd: NDArray): NDArray = {
    ResourceScope.using() {
      var resizedImg: NDArray = null
      val arr: Array[NDArray] = new Array[NDArray](nd.shape.get(0))
      for (i <- 0 until nd.shape.get(0)) {
        arr(i) = Image.imResize(nd.at(i), 224, 224)
      }
      resizedImg = NDArray.api.stack(arr, Some(0), arr.length)

      resizedImg = NDArray.api.cast(resizedImg, "float32")
      resizedImg /= 255
      val totensorImg = NDArray.api.swapaxes(resizedImg, Some(1), Some(3))
      val preprocessedImg = (totensorImg - 0.456) / 0.224
      preprocessedImg
    }
  }

  def printAvg(inferenceTimes: Array[Long], metricsPrefix: String, timesOfWarmUp: Int): Unit = {
    var sum: Long = 0
    for (time <- inferenceTimes) {
      sum += time
    }
    val average = sum / (batchSize * inferenceTimes.length * 1.0e6)
    println(f"$metricsPrefix%s_average $average%1.2fms")
  }

  def runInference(modelPathPrefix: String, context: Context, batchSize: Int, isE2E: Boolean): Unit = {
    var inputShape: Shape = null
    var inputDescriptor: IndexedSeq[DataDesc] = null
    if (isE2E) {
      inputShape = Shape(1, 300, 300, 3)
      inputDescriptor = IndexedSeq(DataDesc("data", inputShape, DType.UInt8, "NHWC"))
    } else {
      inputShape = Shape(1, 3, 224, 224)
      inputDescriptor = IndexedSeq(DataDesc("data", inputShape, DType.Float32, "NCHW"))
    }

    val predictor = new
        Predictor(modelPathPrefix, inputDescriptor, context)

    val times: Array[Long] = Array.fill(numOfRuns){0}

    for (n <- 0 until numOfRuns + timesOfWarmUp) {
      NDArrayCollector.auto().withScope {

        val nd = NDArray.api.random_uniform(Some(0), Some(255), Some(Shape(batchSize, 300, 300, 3)))
        val img = NDArray.api.cast(nd, "uint8")
        var imgWithBatchNum: NDArray = null
        var preprocessedImage: NDArray = null
        // time the latency after warmup
        if (n >= timesOfWarmUp) {
          times(n - timesOfWarmUp) = System.nanoTime()
        }
        if (isE2E) {
          img.asInContext(context)
        } else {
          preprocessedImage = preprocessImage(img)
          preprocessedImage.asInContext(context)
        }
        imgWithBatchNum = if (isE2E) img else preprocessedImage
        val output = predictor.predictWithNDArray(IndexedSeq(imgWithBatchNum))
        output(0).waitToRead()
        if (n >= timesOfWarmUp) {
          times(n - timesOfWarmUp) = System.nanoTime() - times(n - timesOfWarmUp)
        }
      }
    }
    println(if (isE2E) "E2E" else "Non E2E")
    printAvg(times, if (batchSize > 1) "batch_inference" else "single_inference", timesOfWarmUp)
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

    runInference(modelPathPrefix, context, batchSize, isE2E)
  }
}