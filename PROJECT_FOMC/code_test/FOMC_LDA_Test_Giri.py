# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

######################################
### TOPIC MODELING OF FOMC MINUTES ###
######################################

### IMPORT LIBRARIES ###

import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim import models
import matplotlib.pyplot as plt
import spacy
from pprint import pprint
from wordcloud import WordCloud
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_lg")
nlp.max_length = 1500000 # In case max_length is set to lower than this (ensure sufficient memory)

### GRAB THE DOCUMENTS BY PARSING URLs ###

### SETTING UP THE CORPUS ###

FOMCMinutes = [] # A list of lists to form the corpus
FOMCWordCloud = [] # Single list version of the corpus for WordCloud
FOMCTopix = [] # List to store minutes ID (date) and weight of each para



# Define function to prepare corpus
def PrepareCorpus(fomc_statements_file ,  minparalength, numdocs):

    fomcstatements = []
    fomcwordcloud = []
    fomctopix = []
    
    df_fomc_statements_raw = pd.read_csv( fomc_statements_file )
    
    df_fomc_statements = df_fomc_statements_raw[ :numdocs]
    
    for statement in df_fomc_statements.text:
        
        print("\n Raw statement\n")
        print(statement[:150])
        
        # Clean text - stage 1
        statement = statement.strip()  # Remove white space at the beginning and end
        statement = statement.replace('\r', '') # Replace the \r with null
        statement = statement.replace(' ', ' ') # Replace " " with space. 
        statement = statement.replace(' ', ' ') # Replace " " with space.
        
        while '  ' in statement:
            statement = statement.replace('  ', ' ') # Remove extra spaces

        # Clean text - stage 2, using regex (as SpaCy incorrectly parses certain HTML tags)    
        statement = re.sub(r'(<[^>]*>)|' # Remove content within HTML tags
                         '([_]+)|' # Remove series of underscores
                         '(http[^\s]+)|' # Remove website addresses
                         '((a|p)\.m\.)', # Remove "a.m" and "p.m."
                         '', statement) # Replace with null

        
        #print("\n Stage 2\n")
        #print( statement)
        
        # Find length of minutes document for calculating paragraph weights
        statementParas = statement.split('\n') # List of paras in statement, where statement is split based on new line characters
        cum_paras = 0 # Set up variable for cumulative word-count in all paras for a given  transcript
        for para in statementParas:
            if len(para)>minparalength: # Only including paragraphs larger than 'minparalength'
                cum_paras += len(para)
        
        # Extract paragraphs
        for para in statementParas:
            if len(para)>minparalength: # Only extract paragraphs larger than 'minparalength'
                
                topixTmp = [] # Temporary list to store minutes date & para weight tuple
                topixTmp.append(statement) # First element of tuple (minutes date)
                topixTmp.append(len(para)/cum_paras) # Second element of tuple (para weight), NB. Calculating weights based on pre-SpaCy-parsed text
                            
                # Parse cleaned para with SpaCy
                statementPara = nlp(para)
                
                statementTmp = [] # Temporary list to store individual tokens
                
                # Further cleaning and selection of text characteristics
                for token in statementPara:
                    if token.is_stop == False and token.is_punct == False and (token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ =="VERB"): # Retain words that are not a stop word nor punctuation, and only if a Noun, Adjective or Verb
                        statementTmp.append(token.lemma_.lower()) # Convert to lower case and retain the lemmatized version of the word (this is a string object)
                        fomcwordcloud.append(token.lemma_.lower()) # Add word to WordCloud list
                fomcstatements.append(statementTmp) # Add para to corpus 'list of lists'
                fomctopix.append(topixTmp) # Add minutes date & para weight tuple to list for storing
            
    return fomcstatements, fomcwordcloud, fomctopix


# Prepare corpus
derived_data_dir = "../derived"

fomc_statements_file = derived_data_dir + "/" + "FOMC_statements.csv"

FOMCStatements, FOMCWordCloud, FOMCTopix = PrepareCorpus( fomc_statements_file , minparalength=10, numdocs = 1)


#
#   Create a Gensim dictionary from the documents embedded in the fomc statement dataframe
#
#
# ----------------------------------
df_fomc_statements_raw = pd.read_csv( fomc_statements_file )

fomc_statements = df_fomc_statements_raw.text

# Tokenize the sentences into words.
texts = [[ ]]