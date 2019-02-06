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
import java.net.URL

import javax.imageio.ImageIO
import org.apache.commons.io._
import org.apache.mxnet._
import org.apache.mxnet.infer.Classifier
import org.kohsuke.args4j.{CmdLineException, CmdLineParser, Option}
import collection.JavaConverters._

/**
  * Example showing usage of Infer package to do inference on resnet-18 model
  * Follow instructions in README.md to run this example.
  */
object EndToEndModelWoPreprocessing {

  @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
  private val modelPathPrefix = "/model/resnet18_v1"
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


  def imagePreprocess(buf: BufferedImage): Array[Float] = {

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

    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)
    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val classifier = new
        Classifier(modelPathPrefix, inputDescriptor, context)

    val currTime: Array[Long] = Array.fill(numOfRuns){0}
    val times: Array[Long] = Array.fill(numOfRuns){0}
    for (n <- 0 until numOfRuns) {
      // Loading single image from file and getting BufferedImage
      val img = loadImageFromFile(inputImagePath)
      currTime(n) = System.nanoTime()
      // Resize the image
      val resizedImg = reshapeImage(img, 224, 224)
      // Preprocess the image
      val prepossesedImg = imagePreprocess(resizedImg)

      val imgWithBatchNum = NDArray.array(prepossesedImg, shape = inputShape)

      val output = classifier.classifyWithNDArray(IndexedSeq(imgWithBatchNum), Some(5))
      times(n) = System.nanoTime() - currTime(n)
      // Printing top 5 class probabilities
      for (i <- output) {
        printf("Classes with top 5 probability = %s \n", i)
      }
    }
    printStatistics(times, "single_inference")

  }
}