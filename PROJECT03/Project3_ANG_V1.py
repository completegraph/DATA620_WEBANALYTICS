#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:05:41 2022

Version 1 of the Classification of Names by Gender: Project 3

@author: alexanderng
"""

print("\nVersion 1:  Project 3 - Classify Names by Gender\n\n")

import pandas as pd
from nltk.corpus import names

import pronouncing

import phonetics

import random

#
# Basic Feature Extraction

def gender_features(word):
    return {'last_letter' : word[-1]}


labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
[(name, 'female') for name in names.words('female.txt')])


# Set seed to ensure the simulation can be reproduced.

random.seed(a = 102)  
random.shuffle(labeled_names)




# 
#  Let's make sure the dictionary keys are globally unique by imposing
#  a namespace qualifier.
#  Use a prefix by feature-type.   
#  Each extractor function computes only 1 feature type.
#  So features in one extractor function 
# -------------------------------------------------------------------------

#   generates the dictionary key name:  
#     var_type = 1  =>  Alphabetical
#     var_type = 2  =>  Integer
#     var_type = 3  =>  Boolean
# -----------------------------------------------
def make_key_name(prefix, var_type , key ):
    if var_type == 1:        
        keyname = prefix + "_" + "A" + "_" + key
    elif var_type == 2:
        keyname = prefix + "_" + "N" + "_" + key
    else:
        keyname = prefix + "_" + "B" + "_" + key
    return keyname


def extract_response(gender):
    pre = "R"
    features = {}
    features[make_key_name(pre, 1, "gender")] = gender
    return features


def count_consonants(string):
   num_consonants = 0
   # to count the consonants
   for char in string:
      if char not in "aeiou":
         num_consonants += 1
   return num_consonants

def count_vowels(string):
   num_vowels = 0
   # to count the voweles
   for char in string:
      if char in "aeiou":
         num_vowels += 1
   return num_vowels

def extract_basic_features(name):
    pre = "B"
    features = {}
    
    features[make_key_name(pre, 1, "name")] = name
    features[make_key_name(pre, 1, "firstletter") ] = name[0].lower()
    features[make_key_name(pre, 1, "secondletter")] = name[1].lower()
    features[make_key_name(pre, 1, "lastletter")] = name[-1].lower()
    features[make_key_name(pre, 2, "length")] = len(name)
    
    
    # Count the consonants and vowels
    features[make_key_name(pre, 2, "numconsonants")] = count_consonants(name.lower())
    features[make_key_name(pre, 2, "numvowels")] = count_vowels(name.lower())
    
    return features




def extract_phonetic_features(name):
    pre = "P"
    features = {}
    
    dmeta  = phonetics.dmetaphone(name)  # Double Metaphone is usually defined for all names.

    features[make_key_name(pre, 1, "dmetacode")] = dmeta[0] if len(dmeta) > 0 else ""
    features[make_key_name(pre, 2, "dmetalen")]  = len(dmeta[0]) if len(dmeta) > 0 else 0    

    # Initial the phoneme split algorithm: 0
    for j in range(0, 12):
        features[make_key_name(pre, 1, "phx_" + str(j).zfill(2) )] = ""

    last_phoneme = ""

    stress_pos =  0  # If value is undefined

    plist = pronouncing.phones_for_word(name)

    num_syllables = 0
    
    # Final
    
    #
    # Strip out the accent.
    #
    if len(plist) > 0:
        

        
        num_syllables = pronouncing.syllable_count(plist[0])
        
        stress_string = pronouncing.stresses( plist[0])
        
        stress_pos = stress_string.find('1') + 1 # Finds the primary stress in the syllable.
        
        
        list_phonemes_stressed = str.split(plist[0])
        
        
        for j in range(0, len(list_phonemes_stressed)):
            
            s = list_phonemes_stressed[j]
                        
            s_no_ints = ''.join(x for x in s if not x.isdigit() )
            
            features[make_key_name(pre, 1, "phx_" + str(j).zfill(2) ) ] = s_no_ints
            
            if j == len(list_phonemes_stressed) - 1 :
                last_phoneme = s_no_ints
        
    else:
        num_syllables = 0
        
    features[make_key_name(pre, 1, "phcode")] = plist[0] if len(plist) > 0 else ""
    features[make_key_name(pre, 1, "phfirst")] = str.split(plist[0])[0] if len(plist) > 0 else ""
    features[make_key_name(pre, 2, "phlen")] = len( str.split(plist[0]))  if len(plist) > 0 else 0
    features[make_key_name(pre, 2, "phsyllables")] = num_syllables
    features[make_key_name(pre, 3, "phfound")] = True if len(plist) > 0 else False
    features[make_key_name(pre, 2, "phx_stress")] = stress_pos
    features[make_key_name(pre, 1, "phx_last")] = last_phoneme
    return features



def test_extractors():
    
    gf = extract_response("male")
    bf = extract_basic_features("John")
    pf = extract_phonetic_features("John")
    
    allf = { **bf, **pf , **gf}
    
    gf2 = extract_response("female")
    bf2 = extract_basic_features("Samantha")
    pf2 = extract_phonetic_features("Samantha")
    
    allf2 = { **bf2, **pf2, **gf2 }

    rows = []

    rows.append(allf)
    rows.append(allf2)    

    df = pd.DataFrame.from_dict(rows, orient = 'columns')

    return(df, rows )
    
test_extractors()


# Let's work on the pronouncing library

i = 0

total_isin = 0
total_notin = 0
total_female_isin = 0
total_female_notin = 0

total_male_isin = 0
total_male_notin = 0


# Construct the dataframe of outputs row-by-row

rows = []

for n, g in labeled_names:
    
    i += 1
    
    gfeatures = extract_response(g)
    bfeatures = extract_basic_features(n)
    pfeatures = extract_phonetic_features(n)
    
    all_features = { **bfeatures, **pfeatures, **gfeatures }
    
    rows.append(all_features)
    
    isin = all_features["P_B_phfound"]
    
    # Tabulate statistics on the available pronunciations.
    
    if isin:
        total_isin += 1
        if g == "female":
            total_female_isin += 1
        else:
            total_male_isin += 1
    else:
        total_notin += 1
        if g == "female":
            total_female_notin += 1
        else:
            total_male_notin += 1
    
    
    if i < 20:
        num_syllables = all_features["P_N_phsyllables"]
        pronunciation = all_features["P_A_phcode"]
            
        print(" Name: ", n , " isin: " , isin , "gender: ", g, 
                  "Phonetic: " , all_features["P_A_dmetacode"] ,
                  "Pronounciation: ", pronunciation, 
                  " syllables: ", num_syllables )


df_features = pd.DataFrame.from_dict( rows , orient = 'columns')



print("Total Isin: ", total_isin, " Total Not In: ", total_notin, " Sum: " , total_isin + total_notin)

print("Total Female ISIN:  ", total_female_isin , " Total Male ISIN:  ", total_male_isin )

print("Total Female NOTIN: ", total_female_notin, " Total Male Not In:", total_male_notin )



# We will export the training and devtest data jointly.  This allows
# k-fold cross validation to do its work for model evaluation.


df_train_large = df_features[500:]
df_test = df_features[:500]

df_features.to_csv("names_all.csv", index=False)
df_train_large.to_csv("names_train.csv", index = False )
df_test.to_csv( "names_test.csv", index = False )

print("\nGenerated names with features in dataframe format for test-train split\n")

