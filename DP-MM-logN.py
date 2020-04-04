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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--use', type=str, default='gpu', help='cpu or gpu to use?')
args = parser.parse_known_args()[0]

#67.317963493 GPU 14/4
#1549.034912551 CPU 14/4

#with dynamic technology GPU time 19.104662661
#dynamic CPU time 429.26328946

if args.use =='cpu':
    import numpy as cupy
else:
    import cupy

import AllPairsShortestPath
import time

adj_matrix = cupy.load('weibo_dic/weibo.npz')['matrix']
DIAMETER=adj_matrix.shape[0]

t = time.process_time()
apsp = AllPairsShortestPath.AllPairsShortestPath(adj_matrix, use_dynamic=True)
mr = apsp.apsp(g_diameter=DIAMETER)
te = time.process_time()
print('time:', te-t)
print('FIN')