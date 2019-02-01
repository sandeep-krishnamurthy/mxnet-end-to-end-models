# JAVA Inference

Ubuntu v20 or 21 - DL AMI

Default - Not an End to End Model

1. Setup instructions - Gluon-CV, MXNet, Shell scripts

2. Python script - download pre-trained ResNet18 model which was trained with transforms but model does not contain it as part of sym,params

3. Run tests =- shell scripts, Java file


========================

Replace sym, params with the fused one.
Remove transformations from Java file and run inference.

=======================

## MXNet Java Image Classification Project

This is a project created to use Maven-published Scala/Java package.

### Steps to run:

1. Build the package using Maven with the following commands

   1. CPU

      ```shell
      sudo make javaclean
      sudo make javademo
      ```

   2. GPU

      ```shell
      sudo make javaclean
      sudo make javademo USE_CUDA=1
      ```

2. Activate virtual Environment

   ```shell
   source activate mxnet_p36
   ```

3. Install gluoncv package

   ```shell
   pip install gluoncv
   ```

4. Download the pre-trained Resnet_18 hybrid model. Files are saved in `$PROJECT_DIR/models/gluoncv-resnet-18` directory.

   ```shell
   bash bin/get_gluoncv_resnet_18_data.sh
   ```

5. Run the test

   1. CPU

   ```shell
   bash bin/run_gl_ete.sh
   ```

   2. GPU

   ```shell
   export SCALA_TEST_ON_GPU=1 USE_GPU=1
   bash bin/run_gl_ete.sh
   ```

6. Clean up:

   ```shell
   sudo make javaclean
   ```

NOTE:

1. You will find detailed java command list parameter in  `bin/run_gl_ete.sh` file.
2. The complete java inference code resides in `./src/main/java/mxnet/EndToEndModelWoPreprocessing.java` file.



### ERROR:

1. If test failed while executing the java code with:

   ```shell
   SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
   SLF4J: Defaulting to no-operation (NOP) logger implementation
   SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
   ```

**Solution:** Add the below line of code to `pom.xml` file

```xml
<dependencies>
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-api</artifactId>
        <version>1.7.5</version>
    </dependency>
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-log4j12</artifactId>
        <version>1.7.5</version>
    </dependency>
</dependencies>
```

 

2. If test failed while executing the java code with:

   ```shell
   Exception in thread "main" java.lang.UnsatisfiedLinkError:
   /tmp/mxnet1613124969478303655/mxnet-scala: libcudart.so.9.2: 
   cannot open shared object file: No such file or directory
   ```

   

**Solution:** You have to set the CUDA library path:

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64/
OR
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64/' >> ~/.bashrc
source ~/.bashrc
```

### More Information:

1. https://github.com/karan6181/incubator-mxnet/tree/java_inference_ete_model/scala-package/mxnet-demo/java-demo
