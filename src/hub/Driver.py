'''
Created on Apr 5, 2018

@author: abhijit.tomar
'''
from explore.WordCloudGenerator import Clouder
from feature_engineering.PlotFeatures import FeaturePlotter
from feature_engineering.FeaturesFromContent import DirectFeatures

if __name__ == '__main__':
    
    cloud = Clouder({'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv'})
    cloud.generate_word_clouds()
    direct = DirectFeatures({'purge':True,'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv','nasty_path':'../../data/nasties.pkl'})
    direct.generate_direct_features()
    feat_plot = FeaturePlotter({'direct':direct})
    feat_plot.generate_plots_for_direct_features()