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


import java.awt.image.BufferedImage
import java.io.File

import javax.imageio.ImageIO
import org.apache.mxnet._
import org.apache.mxnet.infer.Classifier
import org.kohsuke.args4j.{CmdLineException, CmdLineParser, Option}

import collection.JavaConverters._

/**
  * Example showing usage of Infer package to do inference on resnet-18 model
  * Follow instructions in README.md to run this example.
  */
object EndToEndModelWoPreprocessing {

  @Option(name = "--model-e2e-path-prefix", usage = "input model directory and prefix of the model")
  private val modelPathPrefixE2E = "model/resnet18_end_to_end"
  @Option(name = "--model-non_e2e-path-prefix", usage = "input model directory and prefix of the model")
  private val modelPathPrefixNonE2E = "model/resnet18_v1"
  @Option(name = "--input-image", usage = "the input image")
  private val inputImagePath = "/images/kitten.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  private val inputImageDir = "/images/"
  @Option(name = "--num-runs", usage = "Number of runs")
  private val numRuns = "1"
  @Option(name = "--batchsize", usage = "batch size")
  private val batchsize = "1"


  def loadImageFromFile(inputImagePath: String): BufferedImage = {
    val img = ImageIO.read(new File(inputImagePath))
    img
  }

  def reshapeImage(img: BufferedImage, newWidth: Int, newHeight: Int): BufferedImage = {
    val resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB)
    val g = resizedImage.createGraphics()
    g.drawImage(img, 0, 0, newWidth, newHeight, null)
    g.dispose()

    resizedImage
  }


  def imagePreprocessJavaWay(buf: BufferedImage): Array[Float] = {

    val w = buf.getWidth()
    val h = buf.getHeight()

    // get an array of integer pixels in the default RGB color mode
    val pixels = buf.getRGB(0, 0, w, h, null, 0, w)

    // 3 times height and width for R,G,B channels
    val result = new Array[Float](3 * h * w)
    var row = 0
    // copy pixels to array vertically
    while (row < h) { //
      var col = 0
      // copy pixels to array horizontally
      while (col < w) {
        val rgb = pixels(row * w + col)
        // getting red color
        result(0 * h * w + row * w + col) = ((rgb >> 16) & 0xFF) / 255.0f
        // getting green color
        result(1 * h * w + row * w + col) = ((rgb >> 8) & 0xFF) / 255.0f
        // getting blue color
        result(2 * h * w + row * w + col) = (rgb & 0xFF) / 255.0f
        col += 1
      }
      row += 1
    }

    buf.flush()

    normalize(result, h, w)
  }

  def imageToNDArray(buf: BufferedImage): NDArray = {
    val w = buf.getWidth()
    val h = buf.getHeight()
    // get an array of integer pixels in the default RGB color mode
    val pixels = buf.getRGB(0, 0, w, h, null, 0, w)

    // 3 times height and width for R,G,B channels
    val result = new Array[Float](3 * h * w)
    var row = 0
    // copy pixels to array vertically
    while (row < h) {
      var col = 0
      // copy pixels to array horizontally
      while (col < w) {
        val rgb = pixels(row * w + col)
        // getting red color
        result(row * col + 0) = (rgb >> 16) & 0xFF
        // getting green color
        result(row * col + 1) = (rgb >> 8) & 0xFF
        // getting blue color
        result(row * col + 2) = rgb & 0xFF
        col += 1
      }
      row += 1
    }
    NDArray.array(result, shape = Shape(1, h, w, 3))
  }

  def imagePreprocess(arr: NDArray): NDArray = {
    var resizedImg = Image.imResize(arr, 224, 224)
    resizedImg = NDArray.api.cast(resizedImg, "float32")
    resizedImg /= 255
    val totensorImg = NDArray.api.swapaxes(resizedImg, Some(0), Some(2))
    val preprocessedImg = (totensorImg - 0.456) / 0.224

    preprocessedImg
  }

  def normalize(img: Array[Float], h: Int, w: Int): Array[Float] = {
    val mean = Array(0.485f, 0.456f, 0.406f)
    val std = Array(0.229f, 0.224f, 0.225f)

    var row = 0
    // copy pixels to array vertically
    while (row < h) {
      var col = 0
      // copy pixels to array horizontally
      while (col < w) { // getting red color
        img(0 * h * w + row * w + col) = (img(0 * h * w + row * w + col) - mean(0)) / std(0)
        // getting green color
        img(1 * h * w + row * w + col) = (img(1 * h * w + row * w + col) - mean(1)) / std(1)
        // getting blue color
        img(2 * h * w + row * w + col) = (img(2 * h * w + row * w + col) - mean(2)) / std(2)
        col += 1
      }
      row += 1
    }
    img
  }

  private def percentile(p: Int, seq: Array[Long]) = {
    scala.util.Sorting.quickSort(seq)
    val k = Math.ceil(seq.length * (p / 100.0)).toInt
    seq(k - 1)
  }

  def printStatistics(inferenceTimesRaw: Array[Long], metricsPrefix: String): Unit = {
    var inferenceTimes = inferenceTimesRaw
    // remove head and tail
    if (inferenceTimes.length > 2) inferenceTimes = inferenceTimesRaw.slice(1, inferenceTimesRaw.length - 1)
    val p50 = percentile(50, inferenceTimes) / 1.0e6
    val p99 = percentile(99, inferenceTimes) / 1.0e6
    val p90 = percentile(90, inferenceTimes) / 1.0e6
    var sum: Long = 0
    for (time <- inferenceTimes) {
      sum += time
    }
    val average = sum / (inferenceTimes.length * 1.0e6)
    println(f"\n$metricsPrefix%s_p99 $p99%fms\n$metricsPrefix%s_p90 $p90%fms\n$metricsPrefix%s_p50 $p50%fms\n$metricsPrefix%s_average $average%1.2fms")
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
    val numOfRuns = numRuns.toInt

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }

    val inputShapeE2E = Shape(1, 300, 300, 3)
    val inputDescriptorE2E = IndexedSeq(DataDesc("data", inputShapeE2E, DType.UInt8, "NHWC"))
    val inputShapeNonE2E = Shape(1, 3, 224, 224)
    val inputDescriptorNonE2E = IndexedSeq(DataDesc("data", inputShapeNonE2E, DType.Float32, "NCHW"))

    // Create object of ImageClassifier class
    val classifierE2E = new
        Classifier(modelPathPrefixE2E, inputDescriptorE2E, context)

    val classifierNonE2E = new
        Classifier(modelPathPrefixNonE2E, inputDescriptorNonE2E, context)

    val currTimeE2E: Array[Long] = Array.fill(numOfRuns){0}
    val currTimeNonE2E: Array[Long] = Array.fill(numOfRuns){0}
    val timesE2E: Array[Long] = Array.fill(numOfRuns){0}
    val timesNonE2E: Array[Long] = Array.fill(numOfRuns){0}

    for (n <- 0 until numOfRuns) {
      val nd = NDArray.api.random_uniform(Some(0), Some(255), Some(Shape(300, 300, 3)))
      val img = NDArray.api.cast(nd, "uint8")
      // E2E
      currTimeE2E(n) = System.nanoTime()
      val imgWithBatchNumE2E = NDArray.api.expand_dims(img, 0)
      val outputE2E = classifierE2E.classifyWithNDArray(IndexedSeq(imgWithBatchNumE2E), Some(5))
      timesE2E(n) = System.nanoTime() - currTimeE2E(n)

      // Non E2E
      currTimeNonE2E(n) = System.nanoTime()
      val preprocessedImage = imagePreprocess(img)
      var imgWithBatchNumNonE2E = NDArray.api.expand_dims(preprocessedImage, 0)
      val outputNonE2E = classifierNonE2E.classifyWithNDArray(IndexedSeq(imgWithBatchNumNonE2E), Some(5))
      timesNonE2E(n) = System.nanoTime() - currTimeNonE2E(n)
    }
    println("E2E")
    printStatistics(timesE2E, "single_inference")
    println("Non E2E")
    printStatistics(timesNonE2E, "single_inference")
  }
}