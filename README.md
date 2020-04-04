# APSP
An algorithm to calculate any graph's All-Pairs Shortest Path efficiently.

# 1.Introduction
APSP is a common problem discussed many years. With GPU this algorithm can get much faster performance than previous SOTA algorithms.

# 2.Usage
Prerequisite:

Graph's Adjacent Matrix preparition:
```
adj = numpy.ones((3,3))
index=[(0,0),(1,1),(2,2)]
for t in index:
    adj[t]=0
index=[(0,2),(2,0)]
for t in index:
    adj[t]*=3.4028235e+38
```
before writing each script, you need to add following scripts to detect use CPU/GPU, by default use GPU with CUDA support.
```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--use', type=str, default='gpu', help='cpu or gpu to use?')
args = parser.parse_known_args()[0]
if args.use =='cpu':
    import numpy as cupy
else:
    import cupy
``` 

##2.1 use GPU
You need install `cupy` for this demo use `pip install cupy-cuda100`  
```
import AllPairsShortestPath
import time

adj_matrix = cupy.load('weibo_dic/weibo.npz')['matrix']
DIAMETER=adj_matrix.shape[0]

t = time.process_time()
apsp = AllPairsShortestPath.AllPairsShortestPath(adj_matrix, use_dynamic=True)
mr = apsp.apsp(g_diameter=DIAMETER)
te = time.process_time()
print('time:', te-t)
```

run script using `python DP-MM.py` shell command.


##2.2 use CPU if you don't have CUDA GPU
You need install `numpy` 
```
import AllPairsShortestPath
import time

adj_matrix = cupy.load('weibo_dic/weibo.npz')['matrix']
DIAMETER=adj_matrix.shape[0]

t = time.process_time()
apsp = AllPairsShortestPath.AllPairsShortestPath(adj_matrix, use_dynamic=True)
mr = apsp.apsp(g_diameter=DIAMETER)
te = time.process_time()
print('time:', te-t)
```
run script using `python DP-MM.py -u cpu` shell command.

#3. Evaluation
|Device|Time|
| --------------- | ------------------------------------------------------------ |
|CPU|429.26s|
|GPU|19.10s|

#4. Conclusion
With CUDA support, this algorithm can calculate a graph of 8508 nodes with a lot of edges in less than 20s, while SOTA algorithm should take 4 days.

#5. Legislation
This Software is PROTECTED by CHINA Government, PATENT NO: CN2019112728633. All RIGHTS RESERVED.