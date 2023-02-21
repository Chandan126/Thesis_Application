from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from collections import Counter
import json
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/sources")
def read_sources():
    return ['R2','R5']

@app.get("/sources/{source}")
def read_book_level_scatter(source):
    print(source)
    if(source=='R2'):
        r2 = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R2\\r2_k_result.csv')
        result = r2.to_json(orient="records")
        parsed = json.loads(result)
        return parsed

    elif(source=='R5'):
        r5 = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R5\\r5_k_result.csv')
        result = r5.to_json(orient="records")
        parsed = json.loads(result)
        return parsed

@app.get("/labels/{source}")
def get_label_number(source):
    if(source=='R2'):
        return '2'
    elif(source=='R5'):
        return '5'

def get_famous_words(df,indices):
  unique_words = []
  word_counts = {}
  cluster_words = {}
  for i in indices:
    for j in set(df[df.Article_no==i].Events.values):
      unique_words.append(j) 
  for i,j in Counter(unique_words).most_common(15):
    word_counts[i] = j 
    cluster_words[i] = 1
  return word_counts,cluster_words

def get_cluster_freq(my_dict,clusters,word):
  cf = 0
  name = 'cluster_words'
  for i in range(clusters):
    cf += my_dict[name+str(i)].get(word,0)
  return cf

def get_global_explanations(clusters,result_df,input_path):
  clusterData = {}
  facets = ['Events','Whats','Whens','Wheres','Whos']
  cluster_freq_name = 'cluster_freq'
  cluster_words_name = 'cluster_words'
  files_list = ['all_events_k_x_means_labelled','all_whats_k_x_means_labelled','all_whens_k_x_means_labelled','all_wheres_k_x_means_labelled','all_whos_k_x_means_labelled']
  for i in range(clusters):
    clusterData['Cluster '+str(i)] = {}
  for i in range(5):
    input = input_path + files_list[i] + '.csv' 
    input_df = pd.read_csv(input)
    my_dict = {}
    for j in range(clusters):
      result_indexes = result_df[result_df.k_labels==j].index
      cluster_freq,cluster_words = get_famous_words(input_df,result_indexes)
      my_dict[cluster_freq_name+str(j)] = cluster_freq
      my_dict[cluster_words_name+str(j)] = cluster_words
    event_explanations = []
    for k in range(clusters):
      cf_idf = {}
      for l in my_dict[cluster_freq_name+str(k)]:
        cf_idf[l] = my_dict[cluster_freq_name+str(k)][l] * math.log10(clusters/(get_cluster_freq(my_dict,clusters,l)))
      sorted_dict = sorted(cf_idf, key=cf_idf.get, reverse=True)
      clusterData['Cluster '+str(k)][facets[i]] = sorted_dict[:5]
      event_explanations.append(sorted_dict[:5])
  return clusterData

@app.get("/global_explanations/{source}")
def fetch_global_explanations(source):
    if(source=='R2'):
        result_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R2\\r2_k_result.csv')
        input_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\R2\\'
        clusterData = get_global_explanations(2,result_df,input_path)
    elif(source=='R5'):
        result_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R5\\r5_k_result.csv')
        input_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\R5\\'
        clusterData = get_global_explanations(5,result_df,input_path)
    return clusterData
    
@app.get("/get_articles/{source}/{selectedClusterNumber}")
def fetch_global_explanations(source,selectedClusterNumber):
    clusterNum = int(selectedClusterNumber.split(" ")[1])
    if(source=='R2'):
        result_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R2\\r2_k_result.csv')
        articles = [f"Article {x}" for x in result_df[result_df.k_labels==clusterNum].index.values]
    elif(source=='R5'):
        result_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R5\\r5_k_result.csv')
        articles = [f"Article {x}" for x in result_df[result_df.k_labels==clusterNum].index.values]
    return articles

def get_important_words(article1,article2,input_path):
  files_list = ['all_events_k_x_means_labelled.csv','all_whats_k_x_means_labelled.csv','all_whens_k_x_means_labelled.csv','all_wheres_k_x_means_labelled.csv','all_whos_k_x_means_labelled.csv']
  result = []
  for i in range(5):
    input_filename = input_path + files_list[i]
    result_df = pd.read_csv(input_filename)
    table_a = result_df[result_df.Article_no==article1]
    table_b = result_df[result_df.Article_no==article2]
    common_k_labels = set(table_a['k_labels']).intersection(set(table_b['k_labels']))
    for k_label in common_k_labels:
        events_a = table_a[table_a['k_labels'] == k_label]['Parent_Words'].tolist()
        events_b = table_b[table_b['k_labels'] == k_label]['Parent_Words'].tolist()
        result.extend(list(set(events_a)|set(events_b)))
  return result

@app.get("/local_explanations/{source}/{article1}/{article2}")
def fetch_local_explanations(source,article1,article2):
    article1 = int(article1.split(" ")[1])
    article2 = int(article2.split(" ")[1])
    if(source=='R2'):
        r2 = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R2\\r2.csv')
        input_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\R2\\'
        article1_data = r2.iloc[article1].content
        article2_data = r2.iloc[article2].content
        important_words = get_important_words(article1,article2,input_path)
    elif(source=='R5'):
        r5 = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\R5\\r5.csv')
        input_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\R5\\'
        article1_data = r5.iloc[article1].content
        article2_data = r5.iloc[article2].content
        important_words = get_important_words(article1,article2,input_path)
    return {'article1_data':article1_data,'article2_data':article2_data,'important_words':important_words}