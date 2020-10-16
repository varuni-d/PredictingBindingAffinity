#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install import_ipynb')


# In[1]:


import io, os, sys, types
from IPython import get_ipython
from nbformat import read
import import_ipynb
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
import pandas as pd
import tqdm
import getdata
import generate_features
import model
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols


# In[3]:


#Do this the first time if needing to download db
data_path = './data/BindingDB'
data_path = getdata.download_BindingDB(data_path)


# In[4]:


#Preprocess dataset
df_3d = getdata.process_BindingDB(path = data_path, y = 'Kd')


# In[5]:


df_3d.info()


# In[6]:


df_3d.to_csv('df_ligand_protein.csv',index=False)


# In[7]:


df_3d=pd.read_csv('df_ligand_protein.csv',dtype=str)


# In[8]:


df_smiles=df_3d['ligand_smiles']
df_protein=df_3d['bindingdb_target_chain__sequence']
df_ba=df_3d['bindingaffinity']


# In[9]:


#run this for the first time to generate ligand features
morgan_matrix_generated=generate_features.generate_morgan_matrix(df_smiles)
morgan_matrix_df=pd.DataFrame(morgan_matrix_generated)
morgan_matrix_df.to_csv('morgan_matrix_generated_ligandprotein.csv',index=False)


# In[10]:


morgan_matrix_df=pd.read_csv('morgan_matrix_generated_ligandprotein.csv',dtype=str)


# In[11]:


morgan_matrix_df.head()


# In[12]:


#dindex


# In[13]:


dindex=[1460,4119,4120,4121,4122,4123,4124,4193,4194,4195]


# In[14]:


#Drop any values for which Morgan fingerprints were not generated

df_protein.drop(df_protein.index[dindex], axis=0, inplace=True)
df_ba.drop(df_ba.index[dindex], axis=0, inplace=True)


# <b>Feature Engineering</b>

# In[15]:


#Get feature matrices
test_size=0.2
LX_train,LX_test,Ly_train,Ly_test=generate_features.ligand_features(morgan_matrix_df,df_ba,test_size)
PX_train,PX_test,Py_train,Py_test=generate_features.protein_features(df_protein,df_ba,test_size)


# In[16]:


print(LX_train.shape),print(PX_train.shape)
print(LX_test.shape),print(PX_test.shape)


# In[17]:


PX_train_df = pd.DataFrame(PX_train)
LX_train_df = pd.DataFrame(LX_train)

PX_test_df = pd.DataFrame(PX_test)
LX_test_df = pd.DataFrame(LX_test)


# In[18]:


PX_train_df.columns = PX_train_df.columns.map(lambda x: str(x) + '_b')
PX_test_df.columns = PX_test_df.columns.map(lambda x: str(x) + '_b')


# In[19]:


df_total_train=PX_train_df.join(LX_train_df)
df_total_test=PX_test_df.join(LX_test_df)

y_train=Py_train
y_test=Py_test


# In[23]:


#lr_test_pred=model.linear_regression(df_total_train, y_train, df_total_test, y_test)
print('Linear Regression done')

#dt_test_pred=model.decisiontree(df_total_train, y_train, df_total_test, y_test)
print('DT done')

rf_test_pred=model.randomforest(df_total_train, y_train, df_total_test, y_test)
print('RF done')


# In[21]:


#gb_test_pred=model.gradientboosting(df_total_train, y_train, df_total_test, y_test)
#print('GB done')


# In[24]:


print(model.score_table)


# In[ ]:


# Load optimized random forest model

rf_model = pickle.load(open('randomforest_model.sav', 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

