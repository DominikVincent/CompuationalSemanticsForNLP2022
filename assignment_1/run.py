#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment, generate_ngrams
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from word2vec import *
from sgd import *

# Check Python Version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
nngrams = len(dataset.ngram_hash)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
centerWordVectors = np.random.rand(nngrams, dimVectors) - 0.5
contextWordVectors =  np.zeros((nWords, dimVectors))
ngram_vectors, context_vectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingLossAndGradient),
    [centerWordVectors, contextWordVectors], 0.3, 20000, None, True, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took %d seconds" % (time.time() - startTime))

# concatenate the input and output word vectors


visualizeWords = ["sweet", "sweetly", "dance", "dancing" , "father", "fatherly", "eat", "eating", "meet", "meeting"
    ]

visualizeVecs = []
for word in visualizeWords:

    ngrams = generate_ngrams(word, 2,4)
    ngram_ids = [dataset.ngram_hash[ng] for ng in ngrams]
    wordVec = ngram_vectors[ngram_ids, :].sum(axis=0)
    visualizeVecs.append(wordVec)
visualizeVecs.append(visualizeVecs[7] - visualizeVecs[6])
visualizeVecs.append(visualizeVecs[9] - visualizeVecs[8])
visualizeVecs = np.array(visualizeVecs)
visualizeWords.extend(["eating-eat", "meeting-meet"])


temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeVecs) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')
