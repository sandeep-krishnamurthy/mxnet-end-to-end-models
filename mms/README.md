# MMS

1. Setup Instruction - CPU / GPU - mms - a/b test

3. Gluon Python script - inference - Not end to end (existing) - ResNet - 18

4. synset.txt

5. signature.json


-----

Above all for end to end model (In Python script remove pre-processing, point to a new model with sym,params having transforms (resize, totensor, normalize)

-----

## Image Classification inference using Python bindings on MMS

### Steps to run:

1. Activate virtual Environment

   ```shell
   source activate mxnet_p36
   ```

2. Install pre-requisites if missing libraries/packages:

   ```shell
   # Install latest released version of mxnet-model-server 
   pip install mxnet-model-server
   
   # Install Apache utils to run Apache Benchmarking
   sudo apt install apache2-utils
   ```

3. Jump to `mms` directory

4. Create a directory to package all model artifacts into a single model archive file

   ```shell
   mkdir /tmp/resnet18_v1
   ```

5. Copy the main python file, signature and synset file to the directory(/tmp/resnet18_v1) which we created.It does't require you to provide `symbols` and `params` files locally. 

   ```shell
   cp gluon_pretrained_image_classification.py signature.json synset.txt /tmp/resnet18_v1/
   ```

6. Create a .mar file by executing below command

   ```shell
   model-archiver --model-name resnet18_v1 --model-path /tmp/resnet18_v1/ --handler gluon_pretrained_image_classification:pretrained_gluon_resnet --runtime python --export-path /tmp
   ```

7. Start the server

   ```shell
   mxnet-model-server --start --models resnet18_v1.mar --model-store /tmp
   ```

8. Open a new similar session(ssh into same EC2 instance) to schedule an inference request.

9. Download the sample image

   ```shell
   curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
   ```

10. Perform the inference by sending a POST request to a server which is currently running a image classification model

    ```shell
    curl -X POST http://127.0.0.1:8080/resnet18_v1/predict -T kitten.jpg
    curl -X POST http://127.0.0.1:8080/resnet18_v1/predict -F "data=@kitten.jpg"
    ```

11. Apache Benchmark(AB) testing command

    ```shell
    ab -k -n 5000 -c 10 -p ~/kitten.jpg -T image/jpeg http://127.0.0.1:8080/resnet18_v1/predict
    ```

    1. - **-k:** Use HTTP KeepAlive feature
       - **-n:** Number of requests to perform
       - **-c:** Number of multiple requests to make at a time
       - **-p:** File containing data to POST. Remember also to set -T
       - **-T:** content-type

12. Stop the server

    ```
    mxnet-model-server --stop
    ```

    
