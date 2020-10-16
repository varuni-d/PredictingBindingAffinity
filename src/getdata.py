#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import wget
from zipfile import ZipFile 
import json
import os 
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Download and save BindingDB dataset
def download_BindingDB(path = './BindingDB'):
    print('Downloading BindingDB...')

    if not os.path.exists(path):
        os.makedirs(path)

    url = 'https://www.bindingdb.org/bind/downloads/BindingDB_All_2020m2.tsv.zip'
    item_path = wget.download(url, path)
    with ZipFile(item_path, 'r') as zip: 
        zip.extractall(path = path) 
        print('Download complete:', path) 
    path = path + '/BindingDB_All.tsv'
    return path 


# In[3]:


#Compute log affinity
def get_log_affinity(y):

    #y = -np.log10(y*1e-9 + 1e-10)
    power=-9
    log_affinity= -(np.log10(y*10**power))
    
    return log_affinity


# In[4]:



def process_BindingDB(path = None, y = 'Kd'):

    if not os.path.exists(path):
        os.makedirs(path)
    print('Loading Dataset...')
    df = pd.read_csv(path, sep = '\t', error_bad_lines=False)
    
    print('Preprocessing data...')
    
    df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
    df = df[df['Ligand SMILES'].notnull()]

    if y == 'Kd':
        label = 'Kd (nM)'
    elif y == 'IC50':
        label = 'IC50 (nM)'
    elif y == 'Ki':
        label = 'Ki (nM)'
    elif y == 'EC50':
        label = 'EC50 (nM)'
    else:
        print('select Kd, Ki, IC50 or EC50')
        
    #Handle null values
    df = df[df[label].notnull()]
    
    df = df[['BindingDB Target Chain  Sequence','BindingDB Reactant_set_id', 'PDB ID(s) of Target Chain', 'Ligand SMILES',
             'PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain', label]]
    df.rename(columns={label: 'label_name'}, inplace=True)
    
    #Compute binding affinity 
    df['label_name'] = df.label_name.str.replace('>', '')
    df['label_name'] = df.label_name.str.replace('<', '')   
    df['label_name'] = df.label_name.astype(float)
    df = df[df['label_name'] <= 10000000.0]
#    df['bindingaffinity']=-(np.log10(df['label_name'].values*10**(-9)))
    df['bindingaffinity'] = get_log_affinity(df.label_name.values) 

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    #Filter columns and handle duplicates
    df_filtered = df.filter(['bindingdb_target_chain__sequence','ligand_smiles','bindingaffinity'], axis=1)
    df_filtered = df_filtered.drop_duplicates(keep='last')
    indexNames = df_filtered[df_filtered['bindingaffinity'] >100 ].index
    df_filtered.drop(indexNames , inplace=True)
        
    print(str(len(df_filtered))+' protein-ligand pairs ready')    
    return df_filtered
#    return df_filtered.bindingdb_target_chain__sequence.values, df_filtered.ligand_smiles.values, df_filtered.bindingaffinity.values


# In[ ]:




