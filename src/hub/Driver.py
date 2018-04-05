'''
Created on Apr 5, 2018

@author: abhijit.tomar
'''
from feature_engineering.FeaturesFromContent import DirectFeatures
from feature_engineering.PlotFeatures import FeaturePlotter
from explore.WordCloudGenerator import Clouder
if __name__ == '__main__':
    
    cloud = Clouder({'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv'})
    cloud.generate_word_clouds()
    direct = DirectFeatures({'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv','nasties':cloud.get_nasties()})
    direct.generate_direct_features()
    feat_plot = FeaturePlotter({'direct':direct})
    feat_plot.generate_plots_for_direct_features()