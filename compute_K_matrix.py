import sys
from kernel_util import *

corpus = np.load(sys.argv[0])
kernels = compute_ngram_kernels(corpus, 3)
K = dyad_library(kernels)
np.save(K, sys.argv[1])

