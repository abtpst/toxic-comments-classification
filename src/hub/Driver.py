'''
Created on Apr 5, 2018

@author: abhijit.tomar
'''
import logging
from hub import Constants
from logging.config import dictConfig
dictConfig(Constants.logging_config)
logger = logging.getLogger()
from explore.WordCloudGenerator import Clouder
from feature_engineering.PlotFeatures import DirectFeaturesPlotter,DerivedFeaturesPlotter
from feature_engineering.FeaturesFromContent import DirectFeatures,DerivedFeatures

if __name__ == '__main__':

    '''
    logger.info("Will generate wordclouds")
    cloud = Clouder({'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv'})
    cloud.generate_word_clouds()
    logger.info("Done with wordclouds")
    logger.info("Will create direct features")
    direct = DirectFeatures({'purge':True,'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv','nasty_path':'../../data/nasties.pkl'})
    direct.generate_features_data_frame()
    logger.info("Done with direct features")
    logger.info("Will plot direct features")
    feat_plot = DirectFeaturesPlotter({'features':direct})
    feat_plot.generate_plots()
    logger.info("Done with plotting direct features")
    '''
    logger.info("Will create derived features")
    derived = DerivedFeatures({'purge':True,'train':'../../data/cleaned/train.csv','test':'../../data/cleaned/test.csv','nasty_path':'../../data/nasties.pkl'})
    logger.info("Will try to create corpus")
    derived.generate_corpus()
    logger.info("Created corpus")
    logger.info("Will try to create feature vectors")
    derived.generate_features()
    logger.info("Done with feature vectors")
    '''
    logger.info("Will plot derived features")
    feat_plot = DerivedFeaturesPlotter({'features':derived})
    feat_plot.generate_plots()
    logger.info("Done with plotting derived features")
    '''
    