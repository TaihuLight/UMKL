# Takes json data file, computes and saves kernel matrices 
# for unigram, bigram, and trigram represenations of review text.

# Command line arguments:
#   Argument 1: json data to parse
#   Argument 2: file name to save kernel matrices

from sklearn.feature_extraction.text import *
import numpy as np
import sys
import json

# Parse corpus
raw_corpus = sys.argv[1]
kernel_filename = sys.argv[2]
corpus = []
with open(raw_corpus, 'r') as f:
    for l in f:
        corpus.append(json.loads(l)['reviewText'])

# TODO: For now, we'll take a small corpus. Scalability will come later.
corpus = corpus[:100]

# Initialize vectorizers
unigram_vectorizer = CountVectorizer()
bigram_vectorizer = CountVectorizer(ngram_range=(2,2)) 
trigram_vectorizer = CountVectorizer(ngram_range=(3,3))

# Compute Document-term matrices and kernels for ngram representations
print 'Computing unigram kernel'
unigram_TM = unigram_vectorizer.fit_transform(corpus)
unigram_kernel = unigram_TM * unigram_TM.T

print 'Computing bigram kernel'
bigram_TM = bigram_vectorizer.fit_transform(corpus)
bigram_kernel = bigram_TM * bigram_TM.T

print 'Computing trigram kernel'
trigram_TM = trigram_vectorizer.fit_transform(corpus)
trigram_kernel = trigram_TM * trigram_TM.T

# Save kernel matrices to npy file
np.save(kernel_filename, [unigram_kernel, bigram_kernel, trigram_kernel]) 
