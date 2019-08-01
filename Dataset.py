'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
from scipy.sparse import dok_matrix
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + "/train.csv")
        self.testRatings = self.load_rating_file_as_matrix(path + "/test.csv")
        self.textualfeatures,self.imagefeatures = self.load_textual_image_features(path)
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        for index, row in df.iterrows():
            u, i = int(row['userID']), int(row['itemID'])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for index, row in df.iterrows():
            user, item ,rating = int(row['userID']), int(row['itemID']) ,1.0
            if (rating > 0):
                mat[user, item] = 1.0
        return mat

    def load_textual_image_features(self,data_path):
        import os, json
        from gensim.models.doc2vec import Doc2Vec
        asin_dict = json.load(open(os.path.join(data_path, 'asin_sample.json'), 'r'))
        # all_items = set(asin_dict.keys())
        # Prepare textual feture data.
        doc2vec_model = Doc2Vec.load(os.path.join(data_path, 'doc2vecFile'))
        vis_vec = np.load(os.path.join(data_path, 'image_feature.npy')).item()
        #print(type(vis_vec),vis_vec)
        text_vec = {}
        #visual_vec = {}
        for asin in asin_dict:
            text_vec[asin] = doc2vec_model.docvecs[asin]
            #visual_vec[asin] = vis_vec[asin]
        filename = data_path + '/train.csv'
        print(filename)

        df = pd.read_csv(filename, index_col=None, usecols=None)
        num_items = 0
        asin_i_dic = {}
        for index, row in df.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            asin_i_dic[i] = asin
            num_items = max(num_items, i)
        t_features = []
        v_features = []
        for i in range(num_items+1):
            t_features.append(text_vec[asin_i_dic[i]])
            v_features.append(vis_vec[asin_i_dic[i]])
        return np.asarray(t_features,dtype=np.float32), np.asarray(v_features,dtype=np.float32)
