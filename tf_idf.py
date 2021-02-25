# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used when your
code is evaluated, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator

def numDocsWithWord(train_set):

    words = {}
    for doc in train_set:
        uniqueWordsInDoc = set()
        for word in doc:
            if word in uniqueWordsInDoc:
                continue
            else:
                uniqueWordsInDoc.update([word])
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
    return words



def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each mail
    example: suppose I had two mails 'like this city' and 'get rich quick' in my training set
    Then train_set := [['like','this','city'], ['get','rich','quick']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two mails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each mail that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """

    numDocsWithToken = numDocsWithWord(train_set)
    toReturn = []

    for doc in train_set:
        numTimesInDoc = {}
        totalWordsDoc = 0
        highestTFIDF = ("", -1)
        for word in doc:
            totalWordsDoc += 1
            if word in numTimesInDoc:
                numTimesInDoc[word] += 1
            else:
                numTimesInDoc[word] = 1
        for word in numTimesInDoc:
            tf_idf = (numTimesInDoc[word] / totalWordsDoc) * math.log(len(train_set) / (1 + numDocsWithToken[word]))
            if tf_idf > highestTFIDF[1]:
                highestTFIDF = (word, tf_idf)
        toReturn.append(highestTFIDF[0])
        
    # return list of words (should return a list, not numpy array or similar)
    return toReturn