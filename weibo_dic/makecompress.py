import numpy

m = numpy.load('weibo.npy')
numpy.savez_compressed('weibo.npz',matrix=m)
print('FIN')