'''
Created on Apr 4, 2018

@author: abhijit.tomar
'''
import os
import pickle
import pandas as pd
import progressbar
progressbar.streams.flush()
from processor.TextProcessors import TextAnalyzer

class DirectFeatures(object):
    '''
    Class for extracting features from text
    '''
    
    _train=None
    _test=None
    _direct_features=None
    _direct_save_path=None
    nasties=set()
    
    def __init__(self, params):
    
        if 'train' in params and 'test' in params:
            self._train = pd.read_csv(params['train'])
            self._test = pd.read_csv(params['test'])
        
        self._direct_save_path='../../data/cleaned/direct_features.csv'
        if os.path.exists(self._direct_save_path):
            if 'purge' in params:
                if not params['purge']:
                    self._direct_features=pd.read_csv(self._direct_save_path)
                else:
                    os.remove(self._direct_save_path)
            else:
                self._direct_features=pd.read_csv(self._direct_save_path)
                
        if 'nasty_path' in params:
            with open(params['nasty_path'],'rb') as nasty:
                self.nasties = pickle.load(nasty)
        
        if 'nasties' in params:
            self.nasties=params['nasties']
            
    def generate_direct_features(self):  
        
        if self._direct_features is None:  
            self._direct_features = pd.concat([self._train.iloc[:,1:3],self._test.iloc[:,1:3]])
            self._direct_features=self._direct_features.reset_index(drop=True)
    
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
        ta=TextAnalyzer({})
        for index in bar(range(len(self._direct_features))):
            row=self._direct_features.loc[index]
            comment=row.comment_text
            num_sentences,num_words,num_unique_words,num_letters,num_punctuations,num_uppers,num_titles,num_stops,mean_length,num_nasties,num_emojis = ta.get_metrics(comment,self.nasties)
            self._direct_features.loc[index, 'number_of_sentences'] = num_sentences
            self._direct_features.loc[index, 'number_of_words'] = num_words
            self._direct_features.loc[index, 'number_of_unique_words'] = num_unique_words
            self._direct_features.loc[index, 'number_of_letters'] = num_letters
            self._direct_features.loc[index, 'number_of_punctuations'] = num_punctuations
            self._direct_features.loc[index, 'number_of_uppercase_words'] = num_uppers
            self._direct_features.loc[index, 'number_of_title_words'] = num_titles
            self._direct_features.loc[index, 'number_of_stop_words'] = num_stops
            self._direct_features.loc[index, 'mean_length_of_words'] = mean_length
            self._direct_features.loc[index, 'number_of_nasty_words'] = num_nasties
            self._direct_features.loc[index, 'number_of_emojis'] = num_emojis
            self._direct_features.loc[index, 'percentage_of_unique_words'] = (100*num_unique_words/num_words)
            self._direct_features.loc[index, 'percentage_of_punctuations'] = (100*num_punctuations/num_words)
            self._direct_features.loc[index, 'percentage_of_nasty_words'] = (100*num_nasties/num_words)
            self._direct_features.loc[index, 'percentage_of_emojis'] = (100*num_emojis/num_words)
        
        self._direct_features.to_csv(self._direct_save_path)
    
    def get_direct_features(self):
        return self._direct_features
    
    def get_train(self):
        return self._train
    
    def get_test(self):
        return self._test