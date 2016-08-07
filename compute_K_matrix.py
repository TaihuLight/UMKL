import sys
from kernel_util import *

raw_corpus = sys.argv[1]
if raw_corpus[-5:] == '.json':
    corpus = parse_json(raw_corpus)
else:
    print "File type not supported."
    exit(0)

corpus = corpus[:100]
kernels = compute_ngram_kernels(corpus, 3)
K = dyad_library(kernels)
np.save(sys.argv[2], K)

