# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

import math

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def trainClassifier(train_set, train_labels, laplace):

    words = {}

    # first value is for num words in spam mails, second is for num words in non-spam
    numWordsPerClass = [0, 0]

    for i in range(len(train_labels)):
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
    

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set

    probWordGivenClass, uniqueWords, numWordsPerClass = trainClassifier(train_set, train_labels, smoothing_parameter)
    
    dev_labels = []

    for doc in dev_set:
        probHamGivenWord = math.log(pos_prior)
        probSpamGivenWord = math.log(1 - pos_prior)
        for word in doc:
            if (word, 0) not in probWordGivenClass:
                continue
            if probWordGivenClass[(word, 0)] == 0:
                probSpamGivenWord += math.log(smoothing_parameter / (numWordsPerClass[0] + uniqueWords * smoothing_parameter))
            else:
                probSpamGivenWord += probWordGivenClass[(word, 0)]
            if probWordGivenClass[(word, 1)] == 0:
                probHamGivenWord += math.log(smoothing_parameter / (numWordsPerClass[1] + uniqueWords * smoothing_parameter))
            else:
                probHamGivenWord += probWordGivenClass[(word, 1)]

        if probHamGivenWord >= probSpamGivenWord:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels
    