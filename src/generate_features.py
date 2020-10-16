#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io, os, sys
import pandas as pd
import numpy as np
import csv
import tqdm


# In[2]:


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols


# In[8]:


from sklearn.model_selection import train_test_split
import sklearn.decomposition
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


#Generate Morgan fingerprint feature matrix for the ligands
def generate_morgan_matrix(smiles):
    morgan_matrix=np.zeros((1,2048))
    length=len(smiles)
    dindex=[]
    for i in range(length):
        try:
            compund=Chem.MolFromSmiles(smiles[i])
            fp=Chem.AllChem.GetMorganFingerprintAsBitVect(compund, 2, nBits=2048)
            fp=fp.ToBitString()
            matrix_row=np.array([int(x) for x in list(fp)])
            morgan_matrix=np.row_stack((morgan_matrix,matrix_row))
#            if i%1000==0:
#                percentage=np.round(100*(i/length),1)
#                print(f'{percentage}% done')
        except:
            print(f'problem index:{i}')
            dindex.append[i]
    morgan_matrix=np.delete(morgan_matrix,0,axis=0)
    print('\n')
    print(f'Morgan Matrix dimensions:{morgan_matrix.shape}')
    return morgan_matrix, dindex  


# In[10]:


def ligand_features(morgan_matrix_df,df_ba,test_size=0.2):
    print('Generating ligand features...')
    LX_train,LX_test,Ly_train,Ly_test=train_test_split(morgan_matrix_df,df_ba,test_size=0.2,random_state=42,shuffle=False)
    #Dimensionality reduction
    pca=PCA(.90)
    pca.fit(LX_train)
    print('Ligand features after PCA:',pca.n_components_)
    LX_train = pca.transform(LX_train)
    LX_test = pca.transform(LX_test)
    print('Ligand features complete!')
    return LX_train,LX_test,Ly_train,Ly_test


# In[ ]:


def protein_features(df_protein, df_ba, test_size = 0.2):
    print('Generating protein features...')   
    PX_train,PX_test,y_train,y_test = train_test_split(df_protein, df_ba, test_size = 0.2, random_state = 42,shuffle=False)

    # Create a Count Vectorizer to gather the unique elements in sequence
    vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))

    # Fit and Transform CountVectorizer
    vect.fit(PX_train)
    PX_train = vect.transform(PX_train)
    PX_test = vect.transform(PX_test)
    
    #Dimensionality reduction
    svd = sklearn.decomposition.TruncatedSVD(n_components=50)
    PX_train = svd.fit_transform(PX_train)
    PX_test = svd.transform(PX_test)

    #print('Protein features':vect.get_feature_names()[-20:])
    print('Protein features complete!')
    return PX_train,PX_test,y_train,y_test


# In[ ]:




