# Copyright (c) 2020-2040 Dezhou Shen, Tsinghua University
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.extend(['.','..'])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--use', type=str, default='gpu', help='cpu or gpu to use?')
args = parser.parse_known_args()[0]

#19.319744405999998 GPU
#427.90335372500004 CPU

if args.use =='cpu':
    import numpy as cupy
else:
    import cupy

from src.AllPairsShortestPath import AllPairsShortestPath
import time

adj_matrix = cupy.load('../dataset/weibo-actors-adjacent.npz')['matrix']
DIAMETER=9
print(adj_matrix.shape)

t = time.process_time()
apsp = AllPairsShortestPath(adj_matrix)


counter = 1
for mat in apsp.apsp_iter(g_diameter=8):
    print(mat)
    cupy.savez_compressed('target/weibo-%d.npz'%counter, mat=mat)
    counter+=1

te = time.process_time()
print('time:', te-t)
print('FIN')