#0. Dataset: Sina Weibo Actors Following Relationship Matrix
The following files are in `dataset` folder,
1. actorId.txt  
This file contains profiles of movie actors who has an account in Sina Weibo. Following information are provided for each actor: the number of fans(10k), Sina Weibo verified name, Sina Weibo Account Id, Sina Weibo Nickname, actor self description. All actors have at least 10,000 of fans, it shows that all users in this dataset are famous to some point.  

2. weibo-actors-adjacent.npz  
This file is a compressed adjacent NumPy matrix of 8508 actors. The adjacent matrix `m` stores connection for each node pairs. For all node `u`, set `m[u,u]=0`, and for any node `u` follows node `v`, then set `m[u, v]=1`, other values in matrix `m` are set to `m[u, v]=3.4028235e+38` which is the maximum value of floating-point in 32bit.

3. matrix.npy
Use `unzip weibo-actors-adjacent.npz` to get `matrix.npy` file. This file is the NumPy matrix format which used as computations input.


#1. Environment  

##1.1 Hardware  
Experiments are performed on a server with CPU of Intel Xeon CPU E5-2620 v4 @ 2.10GHz * 2 and GPU of NVIDIA GeForce GTX 1080 Ti GPU, the server has a total of 128GB memories and 11G GPU memories.

##1.2 OS  
Prepare a linux distribution os, e.g. CentOS Linux release 7.2.1511.

#2. Library Requirement  

##2.1 Programming Environment
Firstly, installation of python 3.6+, NVIDIA CUDA10.0, JDK 1.8.0, Maven 3.6.1 is required.

##2.2 Python Libraries 
Secondly, python libraries need to be installed, install dependencies using command: `pip install cupy-cuda100==7.3.0 numpy==1.18.2 scipy==1.4.1`, full list as follows: cupy-cuda100==7.3.0, numpy==1.18.2, scipy==1.4.1.

##2.3 Java Libraries
Thirdly, java libraries need to be installed, install dependencies using command: `mvn install:install-file -DpomFile=pom[-gpu].xml`.  
The installed full list is as follows:  
`
org.nd4j:nd4j-cuda-10.0-platform:1.0.0-beta6
org.nd4j:nd4j-native-platform:1.0.0-beta6
org.nd4j:nd4j-native:windows-x86_64-avx2:1.0.0-beta6
org.nd4j:nd4j-native:linux-x86_64-avx2:1.0.0-beta6
org.slf4j:slf4j-api:1.7.25
ch.qos.logback:logback-classic:1.2.3
org.projectlombok:lombok:1.18.10
org.springframework.boot:spring-boot-starter-web:2.2.5.RELEASE
org.springframework.boot:spring-boot-starter-test:2.2.5.RELEASE
`

##2.4 Java jar build.
Lastly, use maven to build executing jar. Using command: `mvn package -f pom[-gpu].xml`, then get the result jar file `apsp-cpu.jar` or `apsp-gpu.jar`.

#3. Execution Guidline

##3.1 Floyd-Warshal
Command parameters explained as follows:
1) input matrix in numpy format
2) diameter of the network
3) output matrix 
4) algorithm name: floydwarshall for Floyd-Warshal all pairs shortest path algorithm, and powerlandbound for this paper's algorithm.  
Full command as follows:
`java -jar apsp-cpu.jar matrix.npy 8508 apsp.npy floydwarshall`

##3.2 Alon N
Command parameters explained as follows:
1) -u for computation hardware: gpu | cpu
2) -d for diameter of the network
3) -s for sparseness judgement
4) -t for sparseness threshold 0.1 for 10% of the total elements is valid.
5) -m for input matrix file in compressed matrix format.
`python DP-MM-app.py -u gpu -d 8058 -s True -t 0.1 -m dataset\weibo-actors-adjacent.npz`

##3.3 PowerLawBound
##3.3.1 PowerLawBound-CPU-NumPy
`python DP-MM-app.py -u cpu -d 8 -s False -m dataset\weibo-actors-adjacent.npz`

##3.3.2 PowerLawBound-CPU-SciPy-sparse-Numpy
`python DP-MM-app.py -u cpu -d 8 -s True -t 0.1 -m dataset\weibo-actors-adjacent.npz`

##3.3.3 PowerLawBound-GPU-CUBLAS
`java -jar apsp-gpu.jar matrix.npy 8 apsp.npy`

##3.3.4 PowerLawBound-CPU-OPENBLAS
`java -jar apsp-cpu.jar matrix.npy 8 apsp.npy`

##3.3.5 PowerLawBound-GPU-CuPy
`python DP-MM-app.py -u gpu -d 8 -s False -m dataset\weibo-actors-adjacent.npz`

##3.3.6 PowerLawBound-GPU-CuPy-cuSparse-Cupy
`python DP-MM-app.py -u gpu -d 8 -s True -t 0.1 -m dataset\weibo-actors-adjacent.npz`

#4. Experimental Results

| Algorithm  |     Total Execution Time(seconds) |
|------------|-----------------------------------|
|Floyd-Warshal[1-2]|1055880.0|
|Alon N[3] |594.7 |
|PowerLawBound-CPU-NumPy|427.9 |
|PowerLawBound-CPU-SciPy-sparse-Numpy|328.4 |
|PowerLawBound-GPU-CUBLAS|95.0 |
|PowerLawBound-CPU-OPENBLAS|45.0 |
|PowerLawBound-GPU-CuPy|19.32 |
|PowerLawBound-GPU-CuPy-cuSparse-Cupy|15.98 |

#5. Parameters

Parameter used to get state-of-the-art for PowerLawBound algorithm in computation of all pairs shortest path of actors social network.  

| Parameter  |     Value          |       Range    |
|------------|-----------------|------------------|
|Sparseness|True| True/False|
|Sparseness-Threshold|0.1|0-1|
|Diameter |8|  1-8508      |
|Hardware |gpu| cpu/gpu |


#6. Reference
[1] Floyd R W. Algorithm 97: Shortest path[J]. Communications of the ACo, 1962, 5(6,:345.  
[2] Warshall S. A Theorem on Boolean oatrices[J]. Journal of the ACo, 1962, 9(1,:11-12.  
[3] Alon N, Galil Z, oargalit O. On the exponent of the all pairs shortest path problem[J]. Journal of Computer and System Sciences, 1997, 54(2,:255-262.  
