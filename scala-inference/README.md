# Install Maven package
```
sudo apt install maven
```
# Add dependency
```
<repositories>
    <repository>
        <id>Apache Snapshot</id>
        <url>https://repository.apache.org/content/groups/snapshots</url>
    </repository>
</repositories>
```
```
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
    <version>[1.5.0-SNAPSHOT,)</version>
</dependency>
```
# Install OpenCV
```
sudo add-apt-repository ppa:timsc/opencv-3.4
sudo apt-get update
sudo apt install libopencv-imgcodecs3.4
```

## Setup
You are required to use maven to build the package, by running the following:
```
mvn package
```

## Run

```Bash
bash bin/run_im.sh
```

If you want to test run on GPU, you can set a environment variable as follows:
```Bash
export SCALA_TEST_ON_GPU=1
```
## Clean up
To clean up a Maven package, run the following:
```Bash
mvn clean
```
