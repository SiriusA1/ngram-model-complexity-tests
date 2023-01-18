# Conor McGullam
from xml.etree.ElementTree import PI
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import seaborn as sns
import string
from operator import itemgetter
import itertools

def load_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    #print("No of sentences in Corpus: "+str(len(lines)))
    return lines

def tokenize_sentence(lines):
    lines = [i.strip("''").split(" ") for i in lines] 
    #print("No of sentences in Corpus: "+str(len(lines)))
    return lines

def prep_data(lines, n):
    for i in range(len(lines)):
        lines[i] = [''.join(c for c in s if c not in string.punctuation) for s in lines[i]] # remove punctuations
        lines[i] = [s for s in lines[i] if s]          # removes empty strings
        lines[i] = [word.lower() for word in lines[i]] # lower case
        lines[i] += ['</s>']*(n-1)                           # Append </s> at the end of each sentence in the corpus
        for j in range(n-1):
            lines[i].insert(0, '<s>')                      # Append <s> at the beginning of each sentence in the corpus
    #print("No of sentences in Corpus: "+str(len(lines)))
    return lines

def find_word_freq(lines):
    bag_of_words = list(itertools.chain.from_iterable(lines)) # change the nested list to one single list
    count = {}
    for word in bag_of_words:
        if word in count :
            count[word] += 1
        else:
            count[word] = 1
    return count

def compute_unigram_frequencies(lines):
    unigram_frequencies = dict() 
    for sentence in lines:
        for word in sentence:
            unigram_frequencies[word] = unigram_frequencies.get(word) + 1
    return unigram_frequencies

def compute_ngram_frequencies(lines, n):
    ngram_frequencies = dict() 
    for sentence in lines:
        prev_words = []
        for word in sentence:
            if len(prev_words) == n-1:
                ngram_frequencies[(tuple(prev_words + [word]))] = ngram_frequencies.get((tuple(prev_words + [word])),0) + 1
                prev_words = prev_words[1:] + [word]
            else:
                prev_words += [word]
    return ngram_frequencies

def compute_unigram_probabilities(unigram_frequencies):
    unigram_probabilities = dict() 
    for key in unigram_frequencies:
        numerator = unigram_frequencies.get(key)
        denominator =  len(unigram_frequencies.keys())
        if (numerator ==0 or denominator==0 or denominator == None):
            unigram_probabilities[key] = 0
        else:
            unigram_probabilities[key] = float(numerator)/float(denominator)
    return unigram_probabilities

def compute_ngram_probabilities(ngram_frequencies, n_minus_one_freqs):
    ngram_probabilities = dict() 
    for key in ngram_frequencies:
        numerator = ngram_frequencies.get(key)
        denominator = n_minus_one_freqs.get(key[:len(key)-1]) # n_minus_one_freqs.get(key[0]) will get the frequency of previous ngram in the corpus.
        if (numerator ==0 or denominator==0 or denominator == None):
            ngram_probabilities[key] = 0
        else:
            ngram_probabilities[key] = float(numerator)/float(denominator)
    return ngram_probabilities

def compute_prob_test_data(n_minus_one_freqs, ngram_frequencies, ngram_unique_word_count, ngram_probabilities, test, smoothing, n):
    test_sent_prob = 0
    
    if(smoothing == 0):
        prev_words = []
        for word in test:
            if len(prev_words) == n-1:
                if ngram_probabilities.get(tuple(prev_words + [word]))==0 or ngram_probabilities.get(tuple(prev_words + [word]))== None:
                    print(tuple(prev_words + [word]))
                    print("here")
                    return 0
                else:
                    test_sent_prob+=math.log((ngram_probabilities.get(tuple(prev_words + [word]),0)),10)
                prev_words = prev_words[1:] + [word]
            else:
                prev_words += [word]
            
    elif(smoothing ==1):
        prev_words = []
        for word in test:
            if len(prev_words) == n-1:
                ngram_freq = 0 if ngram_frequencies.get(tuple(prev_words + [word]))==None else ngram_frequencies.get(tuple(prev_words + [word]))
                n_minus_one_freq = 0 if n_minus_one_freqs.get((tuple(prev_words)))==None else n_minus_one_freqs.get((tuple(prev_words)))
                numerator = ngram_freq+1
                denominator = n_minus_one_freq+ngram_unique_word_count
                probability = 0 if numerator==0 or denominator ==0 else float(numerator)/float(denominator)
                if(probability==0):
                    return 0
                test_sent_prob +=math.log(probability,10)
                prev_words = prev_words[1:] + [word]
            else:
                prev_words += [word]
            
    return 10**test_sent_prob

def main():
    smoothing = 0
    n = 3
    filename = "ted.txt"
    data = load_file(filename)
    data = tokenize_sentence(data)
    data = prep_data(data, n)
    # unique_word_frequency is a dictionary {word:frequency}.
    word_freq = find_word_freq(data)
    ngram_frequencies = compute_ngram_frequencies(data, n)
    ngram_unique_word_count = len(word_freq)
    # second argument should be compute_ngram_frequencies(data, n-1) or compute_unigram_frequencies(data) if n-1=1
    ngram_probabilities = compute_ngram_probabilities(ngram_frequencies, compute_ngram_frequencies(data, n-1))
    
    test = load_file("test.ted.txt")
    test = tokenize_sentence(test)
    test = prep_data(test, n)
    # test_1d = []
    # for sentence in test:
    #     test_1d += sentence

    print("The probability of the test data under the trained model"+"\nsmoothing ="+str(smoothing))
    print(compute_prob_test_data(compute_ngram_frequencies(test, n-1), ngram_frequencies, ngram_unique_word_count, ngram_probabilities, test[50], smoothing, n))

if __name__ == "__main__":
    main()