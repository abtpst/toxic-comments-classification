'''
Created on Apr 5, 2018

@author: abhijit.tomar
'''
from explore.WordCloudGenerator import Clouder
from feature_engineering.PlotFeatures import DirectFeaturesPlotter,DerivedFeaturesPlotter
from feature_engineering.FeaturesFromContent import DirectFeatures,DerivedFeatures

if __name__ == '__main__':
    '''
    cloud = Clouder({'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv'})
    cloud.generate_word_clouds()
    
    direct = DirectFeatures({'purge':True,'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv','nasty_path':'../../data/nasties.pkl'})
    direct.generate_features_data_frame()
    feat_plot = DirectFeaturesPlotter({'features':direct})
    feat_plot.generate_plots()
    '''
    derived = DerivedFeatures({'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv','nasty_path':'../../data/nasties.pkl'})
    derived.generate_corpus()
    derived.generate_features()
    '''
    feat_plot = DerivedFeaturesPlotter({'features':derived})
    feat_plot.generate_plots()
    '''