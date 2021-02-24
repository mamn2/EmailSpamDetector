# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math

def trainBigramClassifier(train_set, train_labels, laplace):

    bigrams = {}

    # first value is for num words in spam mails, second is for num words in non-spam
    numBigramsPerClass = [0, 0]

    for i in range(len(train_labels)):
        for j in range(len(train_set[i]) - 1):
            bigram = (train_set[i][j].lower(), train_set[i][j + 1].lower())
            numBigramsPerClass[train_labels[i]] += 1
            if bigram in bigrams:
                bigrams[bigram][train_labels[i]] += 1
            else:
                if train_labels[i] == 0:
                    bigrams[bigram] = [1, 0]
                else:
                    bigrams[bigram] = [0, 1]
          
    probBigramGivenClass = {}

    uniqueBigrams = len(bigrams)

    for bigram in bigrams:
        probBigramGivenClass[(bigram, 0)] = math.log((bigrams[bigram][0] + laplace) / (numBigramsPerClass[0] + uniqueBigrams * laplace))
        probBigramGivenClass[(bigram, 1)] = math.log((bigrams[bigram][1] + laplace) / (numBigramsPerClass[1] + uniqueBigrams * laplace))

    return (probBigramGivenClass, uniqueBigrams, numBigramsPerClass)
   
def trainUnigramClassifier(train_set, train_labels, laplace):

    words = {}

    # first value is for num words in spam mails, second is for num words in non-spam
    numWordsPerClass = [0, 0]

    for i in range(0, len(train_labels)):
        for word in train_set[i]:
            numWordsPerClass[train_labels[i]] += 1
            if word in words:
                words[word][train_labels[i]] += 1
            else:
                if train_labels[i] == 0:
                    words[word] = [1, 0]
                else:
                    words[word] = [0, 1]
          
    probWordGivenClass = {}

    uniqueWords = len(words)

    for word in words:
        probWordGivenClass[(word, 0)] = math.log((words[word][0] + laplace) / (numWordsPerClass[0] + uniqueWords * laplace))
        probWordGivenClass[(word, 1)] = math.log((words[word][1] + laplace) / (numWordsPerClass[1] + uniqueWords * laplace))

    return (probWordGivenClass, uniqueWords, numWordsPerClass)
    

def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set

    dev_labels = []

    probWordGivenClass, uniqueWords, numWordsPerClass = trainUnigramClassifier(train_set, train_labels, unigram_smoothing_parameter)
    probBigramGivenClass, uniqueBigrams, numBigramsPerClass = trainBigramClassifier(train_set, train_labels, bigram_smoothing_parameter)

    for doc in dev_set:
        probHamGivenWord = math.log(pos_prior)
        probSpamGivenWord = math.log(1 - pos_prior)
        probHamGivenBigram = math.log(pos_prior)
        probSpamGivenBigram = math.log(1 - pos_prior)
        for i in range(len(doc)):

            #unigram
            word = doc[i]
            if (word, 0) not in probWordGivenClass:
                continue
            if probWordGivenClass[(word, 0)] == 0:
                probSpamGivenWord += math.log(unigram_smoothing_parameter / (numWordsPerClass[0] + uniqueWords * unigram_smoothing_parameter))
            else:
                probSpamGivenWord += probWordGivenClass[(word, 0)]
            if probWordGivenClass[(word, 1)] == 0:
                probHamGivenWord += math.log(unigram_smoothing_parameter / (numWordsPerClass[1] + uniqueWords * unigram_smoothing_parameter))
            else:
                probHamGivenWord += probWordGivenClass[(word, 1)]

            #bigram
            if i == len(doc) - 1:
                continue
            bigram = (doc[i].lower(), doc[i + 1].lower())
            if (bigram, 0) not in probBigramGivenClass:
                continue
            if probBigramGivenClass[(bigram, 0)] == 0:
                probSpamGivenBigram += math.log(bigram_smoothing_parameter / (numBigramsPerClass[0] + uniqueBigrams * bigram_smoothing_parameter))
            else:
                probSpamGivenBigram += probBigramGivenClass[(bigram, 0)]
            if probBigramGivenClass[(bigram, 1)] == 0:
                probHamGivenBigram += math.log(bigram_smoothing_parameter / (numBigramsPerClass[1] + uniqueBigrams * bigram_smoothing_parameter))
            else:
                probHamGivenBigram += probBigramGivenClass[(bigram, 1)]

        probSpam = (((1 - bigram_lambda) * probSpamGivenWord) + (bigram_lambda * probSpamGivenBigram))
        probHam = (((1 - bigram_lambda) * probHamGivenWord) + (bigram_lambda * probHamGivenBigram))

        if probHam >= probSpam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels