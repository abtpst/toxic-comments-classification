'''
Created on Apr 4, 2018

@author: abhijit.tomar
'''
import nltk
import string
import numpy as np
import emoji
import re
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

class TextAnalyzer(object):
    '''
    Class for analyzing certain properties of text
    '''
    my_stopwords=None
    my_apostrophes=None
    my_tokenizer=None
    my_lemmatizer=None
    
    def __init__(self, params):
        '''
        Constructor
        '''
        nltk.data.path.append('../../nltkData')
        self.my_tokenizer=TweetTokenizer()
        self.my_lemmatizer= WordNetLemmatizer()
        self.my_stopwords = set(stopwords.words("english"))
        self.my_apostrophes={
                            "aren't" : "are not",
                            "can't" : "cannot",
                            "couldn't" : "could not",
                            "didn't" : "did not",
                            "doesn't" : "does not",
                            "don't" : "do not",
                            "hadn't" : "had not",
                            "hasn't" : "has not",
                            "haven't" : "have not",
                            "he'd" : "he would",
                            "he'll" : "he will",
                            "he's" : "he is",
                            "i'd" : "I would",
                            "i'd" : "I had",
                            "i'll" : "I will",
                            "i'm" : "I am",
                            "isn't" : "is not",
                            "it's" : "it is",
                            "it'll":"it will",
                            "i've" : "I have",
                            "let's" : "let us",
                            "mightn't" : "might not",
                            "mustn't" : "must not",
                            "shan't" : "shall not",
                            "she'd" : "she would",
                            "she'll" : "she will",
                            "she's" : "she is",
                            "shouldn't" : "should not",
                            "that's" : "that is",
                            "there's" : "there is",
                            "they'd" : "they would",
                            "they'll" : "they will",
                            "they're" : "they are",
                            "they've" : "they have",
                            "we'd" : "we would",
                            "we're" : "we are",
                            "weren't" : "were not",
                            "we've" : "we have",
                            "what'll" : "what will",
                            "what're" : "what are",
                            "what's" : "what is",
                            "what've" : "what have",
                            "where's" : "where is",
                            "who'd" : "who would",
                            "who'll" : "who will",
                            "who're" : "who are",
                            "who's" : "who is",
                            "who've" : "who have",
                            "won't" : "will not",
                            "wouldn't" : "would not",
                            "you'd" : "you would",
                            "you'll" : "you will",
                            "you're" : "you are",
                            "you've" : "you have",
                            "'re": " are",
                            "wasn't": "was not",
                            "we'll":" will",
                            "didn't": "did not",
                            "tryin'":"trying"
                            }
    
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
        num_emojis=0
        for sentence in sentence_list:
            words = nltk.word_tokenize(sentence)
            mean_length_of_words=mean_length_of_words+np.mean([len(w) for w in words])
            unique_words.update(words)
            total_words=total_words+len(words)
            num_emojis=num_emojis+self.extract_emojis(sentence)
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
                    if word.lower() in nasties and word not in string.punctuation and word.lower() not in self.my_stopwords:
                        num_nasties=num_nasties+1
        
        return len(sentence_list),total_words,len(unique_words),total_letters,total_punctuations,total_upper_case_words,total_title_words,total_stopwords,mean_length_of_words,num_nasties,num_emojis
    
    def extract_emojis(self,sentence):
        emoji_list = []

        data = re.findall(r'[^a-zA-Z\d\s\n]', sentence)
        for word in data:
            if word not in string.punctuation and len(word)>0:
                emoji_list.append(word)
        if len(emoji_list)>0:
            print('Found emojis ',emoji_list)
        return len(emoji_list)
    
    def get_sanitized(self,input):
        
        comment=input.lower()
        #remove \n
        comment=re.sub("\\n"," ",comment)
        comment=re.sub("\\s+"," ",comment)
        comment=re.sub("\\t"," ",comment)
        # remove leaky elements like ip,user
        comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
        #removing usernames
        comment=re.sub("\[\[.*\]","",comment)
        
        words=self.my_tokenizer.tokenize(comment)
        
        words=[self.my_apostrophes[word] if word in self.my_apostrophes else word for word in words]
        words=[self.my_lemmatizer.lemmatize(word, "v") for word in words]
        words = [w for w in words if not w in self.my_stopwords]
        words = [w.lower() for w in words]
        
        clean_sent=" ".join(words)
        # remove any non alphanum,digit character
        clean_sent=re.sub("\W+"," ",clean_sent)
        clean_sent=re.sub("  "," ",clean_sent)
        return(clean_sent)