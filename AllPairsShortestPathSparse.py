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

from cupy import cusparse

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--use', type=str, default='gpu', help='cpu or gpu to use?')
parser.add_argument('-s', '--sparse', type=bool, default=True, help='use scipy.sparse?')
args = parser.parse_known_args()[0]

if args.sparse:
    #should be stay inside: (0.008536974741436418, 0.31259434136382935), cannot be too large.
    THRESHOLD = 0.1
    if args.use == 'cpu':
        from scipy.sparse import csr_matrix
    else:
        from cupyx.scipy.sparse import csr_matrix

if args.use =='cpu':
    import numpy as cupy
else:
    import cupy

import math

class AllPairsShortestPathSparse:
    adj_matrix = None
    e_max = None
    g_diameter = None
    use_dynamic = False

    #dense acc
    use_sparse = True
    density = 0


    def __init__(self, adj, g_diameter=9, use_dynamic=False, use_sparse=True):
        self.adj_matrix = adj
        self.e_max = cupy.max(self.adj_matrix)
        self.g_diameter = g_diameter
        self.use_dynamic = use_dynamic
        self.use_sparse = use_sparse
        print('shape:', adj.shape, 'element_max', self.e_max, 'diameter:', self.g_diameter, 'use_dynamic:', self.use_dynamic)

    def stat(self, op):
        stat = [len(op[cupy.where(op <= i)]) for i in (1, self.g_diameter, self.e_max)]
        print('stat:',(stat[0], stat[1]-stat[0], stat[2]-stat[1]))
        return stat

    def max(self, op):
        print('mmax')
        index_op = cupy.where(op < self.e_max)
        op_min = cupy.min(op[index_op])
        op_max = cupy.max(op[index_op])
        print('minv is ', op_min , 'maxv is ', op_max)
        return op_max

    def exponent(self, op, base, current_maxv):
        print('exp')
        index_op = cupy.where(op < self.e_max)
        rindex_op = cupy.where(op >= self.e_max)
        print('expi:',len(index_op[0]),'ri:',len(rindex_op[0]))
        op[index_op] = current_maxv - op[index_op]
        op[index_op] = cupy.power(base + 1, op[index_op])
        op[rindex_op] = 0
        print('exp:',op)
        if self.use_sparse:
            self.density = float(len(index_op[0]))/float(len(index_op[0])+len(rindex_op[0]))
        return op

    def logarithm(self, op, base, current_maxv):
        print('log')
        index_zero = cupy.where(op>0)
        rindex_zero = cupy.where(op==0)
        print('logi:', len(index_zero[0]), 'ri:', len(rindex_zero[0]))
        op[index_zero] = 2 * current_maxv - cupy.log(op[index_zero]) // cupy.log(base + 1)
        op[rindex_zero] = self.e_max
        print('log',op)
        return op

    def dp(self, op):
        print('dp')
        m = op.shape[0]
        op_max = self.max(op)
        op = self.exponent(op, m, op_max)
        print('dense', self.density)
        # check op is dense or not, within THRESHOLD such as 10% sparse, then decide MM or SPMM to use.
        if self.use_sparse and self.density < THRESHOLD:
            sop = csr_matrix(op)
            print('sparse nnz:', sop.nnz)
            if args.use == 'cpu':#cpu use @ after python 3
                sop = sop @ sop
            else:#gpu use cusparse.csrgemm
                sop = cusparse.csrgemm(sop, sop)
            print('sparse nnz2:', sop.nnz)
            op = sop.todense()
        else:
            op = cupy.matmul(op, op)
        op = self.logarithm(op, m, op_max)
        print('dp:',op)
        return op

    def apsp(self, g_diameter=9):
        print('apsp')
        adj = self.adj_matrix
        counter = math.ceil(math.log(g_diameter, 2))
        print('LOOP N:',counter)
        for i in range(counter):
            print('loop index:', i)
            print('apsp,a:', adj)
            wr = self.dp(adj.copy())
            print('apsp,b:', wr)
            post = cupy.minimum(adj, wr)
            print('apsp,c:', adj)
            if self.use_dynamic and cupy.all(cupy.equal(adj, post)):
                print('LOOP EXIT by dynamic decision.')
                break
            adj = post
        print('apsp:', adj)
        return adj

    def apsp_iter(self, g_diameter=9):
        print('apsp')
        adj = self.adj_matrix
        counter = math.ceil(math.log(g_diameter, 2))
        print('LOOP N:',counter)
        for i in range(counter):
            print('loop index:', i)
            print('apsp,a:', adj)
            wr = self.dp(adj.copy())
            print('apsp,b:', wr)
            post = cupy.minimum(adj, wr)
            print('apsp,c:', adj)
            if self.use_dynamic and cupy.all(cupy.equal(adj, post)):
                yield adj
                print('LOOP EXIT by dynamic decision.')
                break
            adj = post
            yield  post
        print('apsp:FIN')


