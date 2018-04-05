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

class Features(object):
    
    _train=None
    _test=None
    _save_path=None
    _features_data_frame=None
    
    def __init__(self,params):
        
        if 'train' in params and 'test' in params:
            self._train = pd.read_csv(params['train'])
            self._test = pd.read_csv(params['test'])
            
class DirectFeatures(Features):
    '''
    Class for extracting features from text
    '''
  
    nasties=set()
    
    def __init__(self, params):
    
        super(DirectFeatures, self).__init__(params)
        
        self._save_path='../../data/cleaned/direct_features.csv'
        if os.path.exists(self._save_path):
            if 'purge' in params:
                if not params['purge']:
                    self._features_data_frame=pd.read_csv(self._save_path)
                else:
                    os.remove(self._save_path)
            else:
                self._features_data_frame=pd.read_csv(self._save_path)
                
        if 'nasty_path' in params:
            with open(params['nasty_path'],'rb') as nasty:
                self.nasties = pickle.load(nasty)
        
        if 'nasties' in params:
            self.nasties=params['nasties']
            
    def generate_features_data_frame(self):  
        
        if self._features_data_frame is None:  
            self._features_data_frame = pd.concat([self._train.iloc[:,1:3],self._test.iloc[:,1:3]])
            self._features_data_frame=self._features_data_frame.reset_index(drop=True)
    
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
        ta=TextAnalyzer({})
        for index in bar(range(len(self._features_data_frame))):
            row=self._features_data_frame.loc[index]
            comment=row.comment_text
            num_sentences,num_words,num_unique_words,num_letters,num_punctuations,num_uppers,num_titles,num_stops,mean_length,num_nasties,num_emojis = ta.get_metrics(comment,self.nasties)
            self._features_data_frame.loc[index, 'number_of_sentences'] = num_sentences
            self._features_data_frame.loc[index, 'number_of_words'] = num_words
            self._features_data_frame.loc[index, 'number_of_unique_words'] = num_unique_words
            self._features_data_frame.loc[index, 'number_of_letters'] = num_letters
            self._features_data_frame.loc[index, 'number_of_punctuations'] = num_punctuations
            self._features_data_frame.loc[index, 'number_of_uppercase_words'] = num_uppers
            self._features_data_frame.loc[index, 'number_of_title_words'] = num_titles
            self._features_data_frame.loc[index, 'number_of_stop_words'] = num_stops
            self._features_data_frame.loc[index, 'mean_length_of_words'] = mean_length
            self._features_data_frame.loc[index, 'number_of_nasty_words'] = num_nasties
            self._features_data_frame.loc[index, 'number_of_emojis'] = num_emojis
            self._features_data_frame.loc[index, 'percentage_of_unique_words'] = (100*num_unique_words/num_words)
            self._features_data_frame.loc[index, 'percentage_of_punctuations'] = (100*num_punctuations/num_words)
            self._features_data_frame.loc[index, 'percentage_of_nasty_words'] = (100*num_nasties/num_words)
            self._features_data_frame.loc[index, 'percentage_of_emojis'] = (100*num_emojis/num_words)
        
        self._features_data_frame.to_csv(self._direct_save_path)
    
    def get_features_data_frame(self):
        return self._features_data_frame
    
    def get_train(self):
        return self._train
    
    def get_test(self):
        return self._test
    
class DerivedFeatures(Features):
    '''
    Class for extracting derived features from text
    '''
   
    corpus=None
    
    def __init__(self, params):
    
        super(DerivedFeatures, self).__init__(params)
        
        self._save_path='../../data/cleaned/derived_features.csv'
        if os.path.exists(self._save_path):
            if 'purge' in params:
                if not params['purge']:
                    self._features_data_frame=pd.read_csv(self._save_path)
                else:
                    os.remove(self._save_path)
            else:
                self._features_data_frame=pd.read_csv(self._save_path)
            
            self.corpus=self._features_data_frame.comment_text
            
    def generate_features_data_frame(self):  
        
        if self._features_data_frame is None:  
            self._features_data_frame = pd.concat([self._train.iloc[:,1:3],self._test.iloc[:,1:3]])
            self._features_data_frame=self._features_data_frame.reset_index(drop=True)
            self.corpus=self._features_data_frame.comment_text
    
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
        ta=TextAnalyzer({})
        for index in bar(range(len(self.corpus))):
            row=self._features_data_frame.loc[index]
            comment=row.comment_text
            self.corpus.loc[index, 'sanitized_comment'] = ta.get_sanitized(comment)
            
        self.corpus.to_csv(self._direct_save_path)
    
    def get_features_data_frame(self):
        return self._features_data_frame
    
    def get_train(self):
        return self._train
    
    def get_test(self):
        return self._test