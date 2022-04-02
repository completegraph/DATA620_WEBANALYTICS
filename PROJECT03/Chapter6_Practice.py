# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("\nChapter 6 Practice\n\n")

import pandas as pd
import nltk
from nltk.corpus import names

import random


def gender_features(word):
    return {'last_letter' : word[-1]}

print( gender_features('Shrek') , "\n" )

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
[(name, 'female') for name in names.words('female.txt')])

random.shuffle(labeled_names)

df_names = pd.DataFrame(labeled_names)

featuresets = [ ( gender_features(n), gender ) for (n, gender) in labeled_names ]

train_set, test_set = featuresets[500:] , featuresets[:500]

classifier = nltk.classify.NaiveBayesClassifier.train( train_set)

# Test on 2 names
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

print(nltk.classify.accuracy(classifier, test_set), "\n")

classifier.show_most_informative_features(5)


from nltk.classify import apply_features
train_set = apply_features(gender_features, labeled_names[500:])
test_set  = apply_features(gender_features, labeled_names[:500])


def gender_features_custom(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["second_letter"] = name[1].lower()
        
    features["last_letter"] = name[-1].lower()
    features["length"] = len(name)
    return features

train_set_custom = apply_features(gender_features_custom, labeled_names[500:])
test_set_custom  = apply_features(gender_features_custom, labeled_names[:500])

train_set_custom = apply_features(gender_features, labeled_names[500:550])
test_set_custom  = apply_features(gender_features, labeled_names[:500])

classifier_custom = nltk.classify.NaiveBayesClassifier.train( train_set_custom )

print(nltk.classify.accuracy(classifier_custom, test_set_custom), "\n")

classifier_custom.show_most_informative_features(20)


print("\n1.2 Choosing the Right Features\n")

# Example 1.2 Overfitting

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

gender_features2('John')



featuresets = [ ( gender_features2(n), gender) for (n, gender) in labeled_names ]
train_set, test_set = featuresets[500:] , featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train( train_set)

print(nltk.classify.accuracy(classifier, test_set), "\n")

# Training, Dev-Test and Test sets
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]


# Train a model using training set and run it on dev-test set
train_set = [ (gender_features(n)  , gender ) for ( n, gender) in train_names ]
devtest_set = [ (gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [( gender_features(n), gender) for (n, gender) in test_names]

classifier = nltk.NaiveBayesClassifier.train( train_set)

print("\nDevtest Accuracy\n")
print( nltk.classify.accuracy(classifier, devtest_set))

# Generate a list of errors
errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name) )
    if guess != tag:
        errors.append( ( tag, guess, name ))

# Then examine the individual error case where the model predicted the wrong label.
# Print out the errors
loop_count = 0
for (tag, guess, name) in sorted( errors):
    if loop_count < 10:
        print('correct={:8} guess={:<8s} name={:<30}'.format(tag, guess, name))
    loop_count += 1

# Look at 2-letter suffixes:  e.g. -yn is female even though -n is male.
def gender_features(word):
    return {'suffix1' : word[-1:] , 
            'suffix2' : word[-2:]}

train_set = [ (gender_features(n), gender ) for (n, gender) in train_names ]
devtest_set = [ ( gender_features(n), gender ) for (n, gender) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train( train_set)

print( nltk.classify.accuracy(classifier, devtest_set ))

print(" So the accuracy with 2 letter suffixes is improved.\n")


# Section 6.1.3

print("\n1.3 Document Classification")

from nltk.corpus import movie_reviews

documents = [( list(movie_reviews.words(fileid)), category )
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category) ]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000] 

def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

df957 = document_features(movie_reviews.words('pos/cv957_8737.txt'))

print({k: df957[k] for k in list( df957 )[:20] } )


# Compute accuracy on the test set
  	

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)

# Section 6.1.4

print("\n 1.4 Part-of-Speech-Tagging\n")

from nltk.corpus import brown
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
    
common_suffixes = [ suffix for (suffix, count) in suffix_fdist.most_common(100)]
print( common_suffixes)

print("Next, define a feature extractor function which checks a given word for these suffixes.")

def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix) ] = word.lower().endswith(suffix)
    return features

print("Train a new decision treee classifier\n")

tagged_words = brown.tagged_words(categories = 'news')
featuresets = [ (pos_features(n), g) for (n,g) in tagged_words ]

size = int(len(featuresets) * 0.05)
train_set, test_set = featuresets[size:15000] , featuresets[:size]    

classifiert = nltk.DecisionTreeClassifier.train( train_set)
nltk.classify.accuracy( classifiert, test_set)

classifiert.classify( pos_features('cats'))

# NLTK can print a decision tree's rules.
print(classifiert.pseudocode(depth=4))

# Section 6.1.5 Exploiting Content

def pos_features(sentence, i ):
    features = {"suffix(1)":  sentence[i][-1:],
                "suffix(2)":  sentence[i][-2:],
                "suffix(3)":  sentence[i][-3:] }
    
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features

# Demo of the sentence level position context
pos_features( brown.sents()[0], 8)              
                
tagged_sents = brown.tagged_sents( categories = 'news')
featuresets = []

for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i , (word, tag) in enumerate(tagged_sent) :
        featuresets.append( ( pos_features(untagged_sent, i), tag))

size = int( len(featuresets) * 0.1 )

train_set , test_set = featuresets[size:] , featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train( train_set )

nltk.classify.accuracy( classifier, test_set )

# Section 6.1.6

print("\n  1.6 Sequence Classification\n")

def pos_features(sentence, i , history):
    features = {"suffix(1)" : sentence[i][-1:] ,
                "suffix(2)" : sentence[i][-2:] ,
                "suffix(3)" : sentence[i][-3:]
                }
    if i == 0:
        features["prev-word"] = "<START>"
        features["prev-tag"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        features["prev-tag"] = history[i-1]
    return features


class ConsecutivePosTagger(nltk.TaggerI ):
    
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features( sentence, i, history) 
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

tagged_sents = brown.tagged_sents( categories = 'news')
size = int(len(tagged_sents) * 0.1 )
train_sents, test_sents = tagged_sents[size:] , tagged_sents[:size]

tagger = ConsecutivePosTagger(train_sents)

print(tagger.evaluate(test_sents))

# Section 6.1.7

print("\nOther Methods for Sequence Classification\n")

