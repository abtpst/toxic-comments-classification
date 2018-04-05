'''
Created on Apr 4, 2018

@author: abhijit.tomar
'''
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
class TextAnalyzer(object):
    '''
    Class for analyzing certain properties of text
    '''
    my_stopwords=None

    def __init__(self, params):
        '''
        Constructor
        '''
        nltk.data.path.append('../../nltkData')
        self.my_stopwords = set(stopwords.words("english"))
    
    def get_metrics(self,intext,nasties=set()):
        sentence_list = nltk.tokenize.sent_tokenize(intext)
        total_words=0
        unique_words=set()
        total_letters=0
        total_punctuations=0
        total_upper_case_words=0
        total_title_words=0
        total_stopwords=0
        mean_length_of_words=0;
        num_nasties=0
        for sentence in sentence_list:
            words = nltk.word_tokenize(sentence)
            mean_length_of_words=mean_length_of_words+np.mean([len(w) for w in words])
            unique_words.update(words)
            total_words=total_words+len(words)
            for word in words:
                total_letters=total_letters+len(word)
                if word.isupper():
                    total_upper_case_words=total_upper_case_words+1
                if word.istitle():
                    total_title_words=total_title_words+1
                if word in string.punctuation:
                    total_punctuations=total_punctuations+1
                if word in self.my_stopwords:
                    total_stopwords=total_stopwords+1
                if nasties is not None and len(nasties)>0:
                    num_nasties=num_nasties+1
        
        return len(sentence_list),total_words,len(unique_words),total_letters,total_punctuations,total_upper_case_words,total_title_words,total_stopwords,mean_length_of_words,num_nasties