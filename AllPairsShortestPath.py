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

if args.use =='cpu':
    import numpy as cupy
else:
    import cupy

import math

class AllPairsShortestPath:
    adj_matrix = None
    e_max = None
    maxd = None
    use_dynamic = False

    def __init__(self, adj, g_diameter=9, use_dynamic=False):
        self.adj_matrix = adj
        self.e_max = cupy.max(self.adj_matrix)
        self.maxd = g_diameter
        self.use_dynamic = use_dynamic
        print('shape:', adj.shape, 'element_max', self.e_max, 'diameter:', self.maxd, 'use_dynamic:', self.use_dynamic)

    def stat(self, op):
        stat = [len(op[cupy.where(op <= i)]) for i in (1,self.maxd, self.e_max)]
        print('stat:',(stat[0], stat[1]-stat[0], stat[2]-stat[1]))
        return stat

    def mmax(self, op):
        print('mmax')
        index_op = cupy.where(op < self.e_max)
        op_min = cupy.min(op[index_op])
        op_max = cupy.max(op[index_op])
        print('mmaxmin is ', op_min , 'max is ', op_max)
        return op_max

    def exponent(self, op, base, current_mx):
        print('exp')
        index_op = cupy.where(op < self.e_max)
        rindex_op = cupy.where(op >= self.e_max)
        print('expi:',len(op[index_op]),'ri:',len(op[rindex_op]))
        op[index_op] = current_mx - op[index_op]
        op[index_op] = cupy.power(base + 1, op[index_op])
        op[rindex_op] = 0
        print('exp:',op)
        return op

    def logarithm(self, op, base, current_mx):
        print('log')
        index_zero = cupy.where(op>0)
        rindex_zero = cupy.where(op==0)
        print('logi:', len(op[index_zero]), 'ri:', len(op[rindex_zero]))
        op[index_zero] = 2 * current_mx - cupy.floor(cupy.log(op[index_zero]) // cupy.log(base + 1))
        op[rindex_zero] = self.e_max
        print('log',op)
        # self.stat(op)
        return op

    def distanceP(self,op):
        print('distp')
        m = op.shape[0]
        print('m,',m)
        op_max = self.mmax(op)
        op = self.exponent(op, m, op_max)
        op = cupy.matmul(op,op)
        op = self.logarithm(op, m, op_max)
        print('distancep:',op)
        return op

    def apsp(self, g_diameter=9):
        print('apsp')
        adj = self.adj_matrix
        counter = math.ceil(math.log(g_diameter, 2))
        print('loop,',counter)
        for i in range(counter):
            print('loop:', i)
            print('apsp,a:', adj)
            wr = self.distanceP(adj.copy())
            print('apsp,b:', wr)
            post = cupy.minimum(adj, wr)
            print('apsp,c:', adj)
            if self.use_dynamic and cupy.all(cupy.equal(adj, post)):
                print('LOOP EXIT.')
                break
            adj = post

        print('apsp:', adj)
        return adj
