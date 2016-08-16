import sys
import random
from kernel_util import *

corpus = []
for n in range(1, len(sys.argv)-1):
    raw_corpus = sys.argv[n]
    if raw_corpus[-5:] == '.json':
        part_corpus = parse_json(raw_corpus)
        corpus += part_corpus
    else:
        print "File type not supported."
        exit(0)

corpus = random.sample(corpus, 3000)
p = 10
kernels = compute_ngram_kernels(corpus, 1)
K = dyad_library(kernels, p)
np.save(sys.argv[-1], K)

