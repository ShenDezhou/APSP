import numpy

adj = numpy.ones((3,3))
index=[(0,0),(1,1),(2,2)]
for t in index:
    adj[t]=0
index=[(0,2),(2,0)]
for t in index:
    adj[t]*=3.4028235e+38
numpy.savez_compressed('test.npz',matrix=adj)
print('FIN')