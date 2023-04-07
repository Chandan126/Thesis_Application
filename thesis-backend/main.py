import string
from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE
import numpy as np
from uuid import uuid4
import os
import ast
import shutil
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

R2_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\R2'
R5_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\R5'
parent_dir = 'C:\\Studies\\Thesis_Application\\thesis-backend'
sources = ['R2','R5']
@app.get("/session")
def create_session():
    session_id = str(uuid4())
    path = os.path.join(parent_dir,session_id)
    os.mkdir(path)
    src = [R2_path,R5_path]
    for i,source in enumerate(src):
       new_path = os.path.join(path,sources[i])
       os.mkdir(new_path)
       src_files = os.listdir(source)
       for file_name in src_files:
            full_file_name = os.path.join(source, file_name)
            if os.path.isfile(full_file_name):
              shutil.copy(full_file_name, new_path)
    return {"sessionId": session_id}

@app.get("/sources")
def read_sources():
    return ['R2','R5']

@app.get("/sources/{sessionId}/{source}")
def read_book_level_scatter(sessionId,source):
    path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\result.csv'
    result_df = pd.read_csv(path)
    result = result_df.to_json(orient="records")
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

def get_all_bigrams(cluster,result_df,data_path):
  data = pd.read_csv(data_path)
  filtered_result_df = data.iloc[result_df[result_df.k_labels==cluster].index]
  all_bigrams = []
  for r in filtered_result_df['bigrams']:
    all_bigrams.extend(ast.literal_eval(r))
  return all_bigrams

def get_common_bigram(query,all_bigram):
  result = []
  for i in range(len(all_bigram)):
      if query in all_bigram[i]:
          result.append(all_bigram[i])
  common = Counter(result).most_common(2)
  output = []
  for com,num in common:
      com_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in com]
      output.append(' '.join(com_words))
  return output

def get_global_explanations(clusters,result_df,input_path,data_path):
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
    for k in range(clusters):
      cf_idf = {}
      all_bigrams = get_all_bigrams(k,result_df,data_path)
      for l in my_dict[cluster_freq_name+str(k)]:
        cf_idf[l] = my_dict[cluster_freq_name+str(k)][l] * math.log10(clusters/(get_cluster_freq(my_dict,clusters,l)))
      sorted_dict = sorted(cf_idf, key=cf_idf.get, reverse=True)
      query = sorted_dict[:5]
      result_words = []
      for quer in query:
        result_words.extend(get_common_bigram(quer,all_bigrams))
      clusterData['Cluster '+str(k)][facets[i]] = result_words
  return clusterData

@app.get("/global_explanations/{sessionId}/{source}")
def fetch_global_explanations(sessionId,source):
    path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\result.csv'
    input_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\'
    data_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\data.csv' 
    result_df = pd.read_csv(path)
    if(source=='R2'):
        clusterData = get_global_explanations(2,result_df,input_path,data_path)
    elif(source=='R5'):
        clusterData = get_global_explanations(5,result_df,input_path,data_path)
    return clusterData
    
@app.get("/get_articles/{sessionId}/{source}/{selectedClusterNumber}")
def fetch_articles(sessionId,source,selectedClusterNumber):
    clusterNum = int(selectedClusterNumber.split(" ")[1])
    path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\result.csv'
    result_df = pd.read_csv(path)
    articles = [f"Article {x}" for x in result_df[result_df.k_labels==clusterNum].index.values]
    return articles

def get_important_words(article1,article2,input_path,data_path):
  files_list = ['all_events_k_x_means_labelled.csv','all_whats_k_x_means_labelled.csv','all_whens_k_x_means_labelled.csv','all_wheres_k_x_means_labelled.csv','all_whos_k_x_means_labelled.csv']
  article1_words = {}
  article2_words = {}
  facets = ['Events','Whats','Whens','Wheres','Whos']
  data_df = pd.read_csv(data_path)
  article1_bigram = ast.literal_eval(data_df.iloc[article1]['bigrams'])
  article2_bigram = ast.literal_eval(data_df.iloc[article2]['bigrams'])
  for i in range(5):
    input_filename = input_path + files_list[i]
    result_df = pd.read_csv(input_filename)
    table_a = result_df[result_df.Article_no==article1]
    table_b = result_df[result_df.Article_no==article2]
    common_k_labels = set(table_a['k_labels']).intersection(set(table_b['k_labels']))
    for k_label in common_k_labels:
        query = list(set(table_a[table_a['k_labels'] == k_label]['Parent_Words'].tolist()))
        result_words = []
        for quer in query:
          result_words.extend(get_common_bigram(quer.lower(),article1_bigram))
        #print(result_words)
        article1_words[facets[i]] = result_words
        result_words = []
        query = set(table_b[table_b['k_labels'] == k_label]['Parent_Words'].tolist())
        for quer in query:
          result_words.extend(get_common_bigram(quer.lower(),article2_bigram))
        article2_words[facets[i]] = result_words
  return article1_words,article2_words

@app.get("/local_explanations/{sessionId}/{source}/{article1}/{article2}")
def fetch_local_explanations(sessionId,source,article1,article2):
  article1 = int(article1.split(" ")[1])
  article2 = int(article2.split(" ")[1])
  data_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\data.csv' 
  #path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\data.csv'
  #data_df = pd.read_csv(path)
  input_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\'
  article1_words,article2_words = get_important_words(article1,article2,input_path,data_path)
  return {'article_1':article1_words,'article_2':article2_words}

@app.get("/get_article_content/{sessionId}/{source}/{article}")
def fetch_local_explanations(sessionId,source,article):
    path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\data.csv'
    data_df = pd.read_csv(path)
    article_data = data_df.iloc[int(article)].content
    return {'article_data':article_data}

@app.get("/get_article_div/{sessionId}/{source}")
def fetch_article_div(sessionId,source):
    feature_list = ['Events ','Whats ','Whens ','Wheres ','Whos ']
    files_list = ['all_events_feature_vector','all_whats_feature_vector','all_whens_feature_vector','all_wheres_feature_vector','all_whos_feature_vector']
    generated_feature_list = []
    for i in range(5):
       path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\' + files_list[i] + '_kmeans.csv'
       df_k = pd.read_csv(path)
       for j in range(len(df_k.columns)-1):
          generated_feature_list.append(feature_list[i] + str(j))
    return generated_feature_list

def find_closest_rows(arr, input_arr, k=5):
    """
    Given a 2D numpy array and an input array,
    returns an array of the k indices of the rows
    in the array that are closest to the input array
    using cosine distance.
    """
    distances = np.apply_along_axis(lambda x: cosine(x, input_arr), axis=1, arr=arr)
    try:
      indices = np.argpartition(distances, k)[:k]
    except: 
      indices = np.argpartition(distances,len(distances)-1)[:]
    indices = indices[np.argsort(distances[indices])]
    return indices

def read_ind_vectors(vec):
  vector=[]
  for j in vec.split(' '):
    k = j.split('[')
    try:
      vector.append(float(k[1]))
    except: 
      pass
    k = j.split(']')
    try:
      vector.append(float(k[0]))
    except: 
      pass
  vector = np.array(vector)
  return vector

def read_vectors(df):
  vectors = []
  for i in df.vectors:
    vector = read_ind_vectors(i)
    vectors.append(vector)
  vectors = np.array(vectors)
  vectors = np.vstack(vectors)
  return vectors

@app.get("/get_facet_explanation/{sessionId}/{source}/{facet}/{article_no}")
def get_facet_explanation(sessionId,source,facet,article_no):
  facet, label = facet.split(' ')
  if('Whats' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\all_whats_k_x_means_labelled.csv')
  elif ('Wheres' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_wheres_k_x_means_labelled.csv')
  elif ('Events' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_events_k_x_means_labelled.csv')
  elif ('Whens' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_whens_k_x_means_labelled.csv')
  elif ('Whos' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_whos_k_x_means_labelled.csv')
  mask = (words_df['Article_no'] == int(article_no)) & (words_df['k_labels'] == int(label))
  all_events = np.unique(words_df.loc[mask].Events.values)
  words_df = words_df[words_df['k_labels'] == int(label)]
  words_df_without_duplicates = words_df.drop_duplicates(subset=['Events'])
  words_df_without_duplicates = words_df_without_duplicates.reset_index()
  words_df_without_duplicates = words_df_without_duplicates[~words_df_without_duplicates.Events.isin(all_events)]
  all_words = []
  all_vecs = read_vectors(words_df_without_duplicates)
  for word in all_events:
    vector = read_ind_vectors(words_df[words_df.Events==word].vectors.values[0])
    if(int(20/len(all_events))<5):
      k = 5
    else:
      k = int(20/len(all_events))
    indices = find_closest_rows(all_vecs,vector,k)
    parent_names = words_df_without_duplicates.iloc[indices]['Parent_Words'].values
    all_words.append(word)
    all_words.extend(parent_names)
  data = []
  for word in all_words:
     data.append({'text':word})
  return {'facet_words' : data}


@app.get("/get_similar_words/{sessionId}/{source}/{facet}/{word}")
def get_most_similar_words(sessionId,source,facet,word):
  facet, label = facet.split(' ')
  all_words = []
  if('Whats' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\all_whats_k_x_means_labelled.csv')
  elif ('Wheres' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_wheres_k_x_means_labelled.csv')
  elif ('Events' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_events_k_x_means_labelled.csv')
  elif ('Whens' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_whens_k_x_means_labelled.csv')
  elif ('Whos' in facet):
    words_df = pd.read_csv('C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\all_whos_k_x_means_labelled.csv')
  vector = read_ind_vectors(words_df[words_df.Parent_Words==word].vectors.values[0])
  words_df = words_df[words_df['k_labels'] == int(label)]
  words_df_without_duplicates = words_df.drop_duplicates(subset=['Events'])
  words_df_without_duplicates = words_df_without_duplicates.reset_index()
  words_df_without_duplicates = words_df_without_duplicates[words_df_without_duplicates.Parent_Words!=word]
  all_vecs = read_vectors(words_df_without_duplicates)
  indices = find_closest_rows(all_vecs,vector,5)
  parent_names = words_df_without_duplicates.iloc[indices]['Parent_Words'].values
  all_words.extend(parent_names)
  return all_words

@app.get("/reassign_words/{sessionId}/{source}/{facet}/{word}/{new_cluster}")
def reassign_words(sessionId,source,facet,word,new_cluster):
  facet, label = facet.split(' ')
  new_cluster, new_label = new_cluster.split(' ')
  if('Whats' in facet):
    filepath = 'C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\'+source+'\\all_whats_k_x_means_labelled.csv'
  elif ('Wheres' in facet):
    filepath = 'C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\'+source+'\\all_wheres_k_x_means_labelled.csv'
  elif ('Events' in facet):
    filepath = 'C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\'+source+'\\all_events_k_x_means_labelled.csv'
  elif ('Whens' in facet):
    filepath = 'C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\'+source+'\\all_whens_k_x_means_labelled.csv'
  elif ('Whos' in facet):
    filepath = 'C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\'+source+'\\all_whos_k_x_means_labelled.csv'
  words_df = pd.read_csv(filepath)
  word_events = words_df.loc[words_df.Parent_Words==word, 'Events']
  words_df.loc[words_df.Events.isin(word_events), 'k_labels'] = int(new_label)
  words_df.to_csv(filepath)


def get_feature_vectors(data,label_feature,sessionId,source):
  source_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\' + 'result.csv'
  source_df = pd.read_csv(source_path)
  all_feature_vec=[]
  if(label_feature=='k_means'):
    k = len(set(data.k_labels.values))
  elif(label_feature=='x_means'):
    k = len(set(data.x_labels.values)) 
  for i in range(len(source_df)):
    event_features_ind_df = data[data.Article_no == i]
    if(label_feature=='k_means'):
      all_vals = Counter(event_features_ind_df.k_labels.values).most_common(k)
    elif(label_feature=='x_means'):
      all_vals = Counter(event_features_ind_df.x_labels.values).most_common(k)      
    feature_vec = np.zeros(k)
    for j in all_vals:
      feature_vec[j[0]] = j[1]/len(event_features_ind_df)
    all_feature_vec.append(feature_vec)
  return all_feature_vec

def get_all_feature_vectors(sessionId,source,feature_sizes_k):
  files_list = ['all_events_k_x_means_labelled','all_whats_k_x_means_labelled','all_whens_k_x_means_labelled','all_wheres_k_x_means_labelled','all_whos_k_x_means_labelled']
  output_list = ['all_events_feature_vector','all_whats_feature_vector','all_whens_feature_vector','all_wheres_feature_vector','all_whos_feature_vector']
  for i in range(5):
    file_input = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\' + files_list[i] + '.csv'
    df = pd.read_csv(file_input)
    k_vectors = get_feature_vectors(df,'k_means',sessionId,source)
    k_vector_df = pd.DataFrame(k_vectors)
    output_file = 'C:\\Studies\\Thesis_Application\\thesis-backend\\'+ sessionId + '\\' + source + '\\' + output_list[i] + '_kmeans.csv'
    k_vector_df.to_csv(output_file)

def get_elbow_k(vectors,config):
  model = KMeans()
  if(config==0):
    visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
  elif(config==1):
    visualizer = KElbowVisualizer(model,metric='silhouette', k=(2,30), timings= True)
  elif(config==2):
    visualizer = KElbowVisualizer(model,metric='calinski_harabasz', k=(2,30), timings= True)
  visualizer.fit(vectors) 
  return visualizer.elbow_value_

def get_kmeans_clusters(k,data):
  kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
  data_labels = kmeans.predict(data)
  return data_labels,kmeans

def get_tsne(data,perplexity_value):
  tsne = TSNE(n_components=2, learning_rate='auto',init='random',perplexity=perplexity_value)
  components = tsne.fit_transform(data)
  return components

def get_clustering_labels(data,k_config=0,elbow_config=0):
  if(k_config==0):
    k = get_elbow_k(data,elbow_config) 
  else:
    k=k_config
  k_means_data_labels,kmeans = get_kmeans_clusters(k,data)
  result = pd.DataFrame(k_means_data_labels,columns=['k_labels'])
  return result

def get_book_level_clustering(k_final_vectors,k):
  k_result = get_clustering_labels(k_final_vectors,k)
  return k_result

def calculate_weights(feature_sizes_k,global_weights,increase_local_weights=None,decrease_local_weights=None):
  features = ['Whats 0','Whens 0','Wheres 0','Whos 0']
  all_weight = np.ones(len(feature_sizes_k))
  n = 0
  for i,weight in enumerate(global_weights):
    weight = int(weight)
    try:
      final_indices = feature_sizes_k.index(features[i])
    except:
      final_indices = len(feature_sizes_k)
    if weight==1 or weight==0:
      all_weight[n:final_indices] = np.full(len(all_weight[n:final_indices]),weight)
    else:
      all_weight[n:final_indices] = np.full(len(all_weight[n:final_indices]),10 ** (-1*weight))
    n = final_indices
  if increase_local_weights!=None:
    increase_weights = [feature_sizes_k.index(local_weight) for local_weight in increase_local_weights]
    for i in increase_weights:
      all_weight[i] = all_weight[i] * (0.1)
  if decrease_local_weights!=None:
    decreased_weights = [feature_sizes_k.index(local_weight) for local_weight in decrease_local_weights]
    for i in decreased_weights:
      all_weight[i] = all_weight[i] * (10)
  return all_weight

def get_final_vectors(sessionId,feature_sizes_k,global_weights,source,increase_local_weights,decrease_local_weights):
    files_list = ['all_events_feature_vector', 'all_whats_feature_vector', 'all_whens_feature_vector', 'all_wheres_feature_vector', 'all_whos_feature_vector']
    base_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\{}_kmeans.csv'
    dfs = [pd.read_csv(base_path.format(f)).drop(['Unnamed: 0'], axis=1) for f in files_list]
    k_final_vectors = np.concatenate([df.values for df in dfs], axis=1)
    importance = calculate_weights(feature_sizes_k,global_weights,increase_local_weights,decrease_local_weights)
    if importance is None:
        importance = np.ones(k_final_vectors.shape[1])
    k_final_vectors *= importance
    return k_final_vectors

@app.get("/recluster_words/{sessionId}/{source}/{feature_sizes_k}/{global_weights}/{increase_local_weights}/{decrease_local_weights}")
def recluster(sessionId,source,feature_sizes_k,global_weights,increase_local_weights=None,decrease_local_weights=None):
  output_path = 'C:\\Studies\\Thesis_Application\\thesis-backend\\' + sessionId + '\\' + source + '\\result.csv'
  feature_sizes_k = json.loads(feature_sizes_k)
  global_weights = json.loads(global_weights)
  increase_local_weights = json.loads(increase_local_weights) if increase_local_weights else None
  decrease_local_weights = json.loads(decrease_local_weights) if decrease_local_weights else None
  get_all_feature_vectors(sessionId,source,feature_sizes_k)
  k_final_vectors = get_final_vectors(sessionId,feature_sizes_k,global_weights,source,increase_local_weights,decrease_local_weights)
  if(source=='R2'):
    labels_df = get_book_level_clustering(k_final_vectors,2)
  else:
    labels_df = get_book_level_clustering(k_final_vectors,5)
  tsne_df = pd.DataFrame(get_tsne(k_final_vectors,30),columns=['x_axis','y_axis'])
  result = pd.concat([pd.DataFrame(k_final_vectors),labels_df,tsne_df],axis=1)
  #result['article_no'] = result.index
  result.to_csv(output_path, index_label='article_no', header=True)
  return {'message': 'Success'}