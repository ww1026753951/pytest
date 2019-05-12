import numpy

from numpy import *


def cos_sim(vector_a, vector_b):

    vector_a = numpy.mat(vector_a)
    vector_b = numpy.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = numpy.linalg.norm(vector_a) * numpy.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


vec1 = array([12, 34, 5, 6, 17, 8])
vec2 = array([21, 3, 114, 12, 4, 5])


dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))

# L1
l1 = numpy.sqrt((vec1-vec2)*(vec1-vec2).T)
print(l1)

# L2
print(dist)

# cos
print(cos_sim(vec1, vec2))

