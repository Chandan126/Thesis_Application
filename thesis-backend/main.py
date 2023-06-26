from operator import itemgetter
import string
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI
import logging
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import numpy as np
import random
import joblib
from uuid import uuid4
import os
import ast
import shutil
import json
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import io

app = FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the paths accordingly
backend_path = '/backend/dirs_path/'
R2_path = '/backend/dirs_path/R2'
R5_path = '/backend/dirs_path/R5'
Trec_path = '/backend/dirs_path/TREC'
parent_dir = '/backend/dirs_path'
sources = ['R2','R5','TREC']
@app.get("/session/{system}")
def create_session(system):
    session_id = str(uuid4())
    path = os.path.join(parent_dir,session_id)
    os.mkdir(path)

    global logger
    log_file_path = os.path.join(path, "logs.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Session created ")
    logger.info('System accessed is '+ str(system))

    src = [R2_path,R5_path,Trec_path]
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
    logger.info("Read all the sources")
    return ['R2','R5','TREC']

def expand_query_glove(original_query, embeddings_dict, num_expansion_terms=8, similarity_threshold=0.5):
    # Tokenize and preprocess the original query
    query_terms = original_query.lower().split()
    # Remove stop words or other irrelevant terms from the query_terms if needed
    # Remove punctuation
    query_terms = [term.translate(str.maketrans('', '', string.punctuation)) for term in query_terms]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    query_terms = [term for term in query_terms if term not in stop_words]
    # Convert each query term to its GloVe vector representation
    query_vectors = [embeddings_dict.get(term, np.zeros_like(next(iter(embeddings_dict.values())))) for term in query_terms]

    # Calculate the similarity between each query term and all other words in the vocabulary
    similarities = cosine_similarity(query_vectors, list(embeddings_dict.values()))

    # Select expansion terms based on similarity scores
    expansion_terms = []
    for i in range(len(query_terms)):
        similar_words = [word for word, similarity in zip(embeddings_dict.keys(), similarities[i]) if similarity > similarity_threshold]
        top_similar_words = sorted(similar_words, key=lambda word: similarities[i][list(embeddings_dict.keys()).index(word)], reverse=True)[:num_expansion_terms]
        expansion_terms.extend(top_similar_words)

        # Remove punctuation and stop words from expansion terms
    expansion_terms = [term.translate(str.maketrans('', '', string.punctuation)) for term in expansion_terms]
    expansion_terms = [term for term in expansion_terms if term not in stop_words]

    # Expand the original query by adding the expansion terms
    expanded_query = ' '.join(query_terms) + ' ' + ' '.join(expansion_terms)

    return expanded_query.lower()

@app.get("/expand_query/{sessionId}/{source}/{query}")
def expand_query(sessionId,source,query):
  logger.info('Original Query ' + str(query))
  glove_path = backend_path + sessionId + '/' + source + '/glove_vectors.pkl'
  embeddings_dict = joblib.load(glove_path)
  query = expand_query_glove(query,embeddings_dict)
  logger.info('Expanded Query ' + str(query))
  return query

def search(sessionId,source,query):
  vectorizer_path = backend_path + sessionId + '/' + source + '/vectorizer.pkl'
  tfidf_matrix_path = backend_path + sessionId + '/' + source + '/tfidf_matrix.pkl'
  vectorizer = joblib.load(vectorizer_path)
  tfidf_matrix = joblib.load(tfidf_matrix_path)
  query_tfidf = vectorizer.transform([query])
  cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
  similar_indices = cosine_similarities.argsort()[:-11:-1]
  return similar_indices



@app.get("/book_level_scatter/{sessionId}/{source}/{query}")
def read_book_level_scatter(sessionId,source,query='undefined'):
    logger_string = 'Reading the ' + str(source) + ' result file '
    logger.info(logger_string)
    path = backend_path + sessionId + '/' + source + '/result.parquet.gzip'
    #print(query)
    result_df = pd.read_parquet(path)
    if(query!='undefined'):
      #print(query)
      result_index = search(sessionId,source,query)
      result_df.loc[result_index, 'highlight'] = 1
    result_df.to_parquet(path,compression='gzip')
    result = result_df.to_json(orient="records")
    parsed = json.loads(result)
    return parsed

@app.get("/labels/{source}")
def get_label_number(source):
    logger.info('Read the ' + str(source) + ' label numbers ')
    if(source=='R2'):
        return '2'
    elif(source=='R5'):
        return '5'
    elif(source=='TREC'):
        return '11'

def get_famous_words(df,indices):
  unique_words = []
  word_counts = {}
  cluster_words = {}
  stopwords_list = set(stopwords.words('english'))
  additional_words = ['say', 'put', 'tell', 'hear', 'set', 'end', 'trend','ask','talk']
  stopwords_list.update(additional_words)
  for i in indices:
    for j in set(df[df.Article_no==i].Events.values):
        filtered_words = [word for word in j.split() if word.lower() not in stopwords_list]
        unique_words.extend(filtered_words)
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
  data = pd.read_parquet(data_path)
  filtered_result_df = data.iloc[result_df[result_df.k_labels==cluster].index]
  all_bigrams = []
  for r in filtered_result_df['bigrams']:
    all_bigrams.extend(ast.literal_eval(r))
  return all_bigrams

def get_common_bigram(query,all_bigram):
  result = []
  for bigram in all_bigram:
      if query in bigram[0] or query in bigram[1]:
          result.append(bigram)
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
    input = input_path + files_list[i] + '.parquet.gzip' 
    input_df = pd.read_parquet(input)
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
      clusterData['Cluster '+str(k)][facets[i]] = ', '.join(result_words)
  return clusterData

@app.get("/global_explanations/{sessionId}/{source}/{system}")
def fetch_global_explanations(sessionId,source,system):
    logger.info('Selected Global explanation type ' + str(system))
    if(system=='A'):
      path = backend_path + sessionId + '/' + source + '/result.parquet.gzip'
      input_path = backend_path + sessionId + '/' + source + '/'
      data_path = backend_path + sessionId + '/' + source + '/data.parquet.gzip' 
      result_df = pd.read_parquet(path)
      if(source=='R2'):
          clusterData = get_global_explanations(2,result_df,input_path,data_path)
      elif(source=='R5'):
          clusterData = get_global_explanations(5,result_df,input_path,data_path)
      elif(source=='TREC'):
          clusterData = get_global_explanations(11,result_df,input_path,data_path)
    else:
      path = backend_path + sessionId + '/' + source + '/explanations.json'
      with open(path, 'r') as f:
        clusterData = json.load(f)
    return clusterData
    
@app.get("/get_articles/{sessionId}/{source}/{selectedClusterNumber}")
def fetch_articles(sessionId,source,selectedClusterNumber):
    logger.info('Fetching Articles from ' + str(source) + ' and cluster ' + str(selectedClusterNumber))
    clusterNum = int(selectedClusterNumber.split(" ")[1])
    path = backend_path + sessionId + '/' + source + '/result.parquet.gzip'
    result_df = pd.read_parquet(path)
    articles = [f"Article {x}" for x in result_df[result_df.k_labels==clusterNum].index.values]
    return articles

def get_important_words(article1,article2,input_path,data_path):
  files_list = ['all_events_k_x_means_labelled.parquet.gzip','all_whats_k_x_means_labelled.parquet.gzip','all_whens_k_x_means_labelled.parquet.gzip','all_wheres_k_x_means_labelled.parquet.gzip','all_whos_k_x_means_labelled.parquet.gzip']
  article1_words = {}
  article2_words = {}
  facets = ['Events','Whats','Whens','Wheres','Whos']
  data_df = pd.read_parquet(data_path)
  article1_bigram = ast.literal_eval(data_df.iloc[article1]['bigrams'])
  article2_bigram = ast.literal_eval(data_df.iloc[article2]['bigrams'])
  for i in range(5):
    input_filename = input_path + files_list[i]
    result_df = pd.read_parquet(input_filename)
    table_a = result_df[result_df.Article_no==article1]
    table_b = result_df[result_df.Article_no==article2]
    common_k_labels = set(table_a['k_labels']).intersection(set(table_b['k_labels']))
    #print(common_k_labels)
    for k_label in common_k_labels:
        query = list(set(table_a[table_a['k_labels'] == k_label]['Parent_Words'].tolist()))
        result_words = []
        for quer in query:
          result_words.extend(get_common_bigram(quer.lower(),article1_bigram))
        #print(result_words)
        if(len(result_words)!=0):
          article1_words[facets[i]] = result_words
        result_words = []
        query = list(set(table_b[table_b['k_labels'] == k_label]['Parent_Words'].tolist()))
        for quer in query:
          result_words.extend(get_common_bigram(quer.lower(),article2_bigram))
        if(len(result_words)!=0):
          article2_words[facets[i]] = result_words
  return article1_words,article2_words

@app.get("/local_explanations/{sessionId}/{source}/{article1}/{article2}")
def fetch_local_explanations(sessionId,source,article1,article2):
  logger.info('Fetching local explanation for ' + str(article1) + ' and ' + str(article2))
  article1 = int(article1.split(" ")[1])
  article2 = int(article2.split(" ")[1])
  data_path = backend_path + sessionId + '/' + source + '/data.parquet.gzip' 
  input_path = backend_path + sessionId + '/' + source + '/'
  article1_words,article2_words = get_important_words(article1,article2,input_path,data_path)
  return {'article_1':article1_words,'article_2':article2_words}

@app.get("/get_article_content/{sessionId}/{source}/{article}")
def get_article_content(sessionId,source,article):
    logger.info('Fetching article content for ' + str(source) + ' and article number ' + str(article))
    path = backend_path + sessionId + '/' + source + '/data.parquet.gzip'
    data_df = pd.read_parquet(path)
    article_data = data_df.iloc[int(article)].content
    article_title = data_df.iloc[int(article)].title
    return {'article_data':article_data,'article_title':article_title}

@app.get("/get_article_div/{sessionId}/{source}")
def fetch_article_div(sessionId,source):
    logger.info('Fetching article division for ' + str(source))
    feature_list = ['Events ','Whats ','Whens ','Wheres ','Whos ']
    files_list = ['all_events_feature_vector','all_whats_feature_vector','all_whens_feature_vector','all_wheres_feature_vector','all_whos_feature_vector']
    generated_feature_list = []
    for i in range(5):
       path = backend_path + sessionId + '/' + source + '/' + files_list[i] + '_kmeans.parquet.gzip'
       df_k = pd.read_parquet(path)
       for j in range(len(df_k.columns)):
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

"""def generate_random_words(facet):
  what_list = ["question", "mystery", "answer", "fact", "information", "knowledge", "puzzle", "riddle", "enigma",
             "solution", "query", "concept", "phenomenon", "thing", "object", "idea", "task", "challenge",
             "occurrence", "incident", "happening", "event", "episode", "circumstance", "situation", "scenario",
             "condition", "state", "circumstance", "status", "position", "location", "place", "site", "venue",
             "destination", "spot", "area", "region", "country", "city", "town", "village", "locale", "neighborhood",
             "time", "moment", "date", "period", "duration", "schedule", "timeline"]

  who_list = ["person", "individual", "human", "man", "woman", "child", "adult", "elder", "citizen", "stranger",
            "guest", "resident", "character", "figure", "celebrity", "artist", "musician", "actor", "actress",
            "politician", "leader", "hero", "villain", "spectator", "audience", "participant", "player", "athlete",
            "team", "crew", "cast", "group", "organization", "company", "community", "society", "population",
            "public", "fan", "follower", "supporter", "friend", "family", "relative", "colleague"]

  when_list = ["time", "moment", "date", "period", "duration", "schedule", "deadline", "appointment", "event",
             "occasion", "year", "month", "week", "day", "hour", "minute", "second", "morning", "afternoon",
             "evening", "night", "dawn", "twilight", "sunset", "sunrise", "past", "present", "future", "now",
             "soon", "early", "late", "yesterday", "today", "tomorrow", "before", "after", "during", "while",
             "once", "twice", "always", "never", "frequently", "occasionally", "regularly", "rarely"]

  events_list = ["concert", "festival", "performance", "exhibition", "show", "party", "celebration", "ceremony",
               "wedding", "birthday", "anniversary", "conference", "seminar", "meeting", "gathering", "competition",
               "tournament", "match", "game", "race", "sport", "parade", "procession", "march", "protest", "demonstration",
               "campaign", "launch", "reception", "farewell", "farewell", "award", "prize", "presentation",
               "exposition", "fair", "market", "sale", "auction", "carnival", "fairground", "circus", "performance"]

  where_list = ["location", "place", "site", "venue", "destination", "spot", "area", "region", "country", "city",
              "town", "village", "locale", "neighborhood", "address", "geography", "terrain", "landscape", "map",
              "direction", "boundary", "district", "zone", "province", "state", "continent", "island", "ocean",
              "sea", "coast", "river", "lake", "mountain", "forest", "park", "building", "structure", "street",
              "road", "highway", "bridge", "airport", "station", "port", "hotel", "restaurant", "store"]
  
  words = []
  
  if facet == "Whats":
      words.extend(random.sample(what_list, 7))
      words.extend(random.sample(where_list, 2))
      words.extend(random.sample(when_list, 2))
      words.extend(random.sample(who_list, 2))
      words.extend(random.sample(events_list, 2))
  elif facet == "Wheres":
      words.extend(random.sample(where_list, 7))
      words.extend(random.sample(what_list, 2))
      words.extend(random.sample(when_list, 2))
      words.extend(random.sample(who_list, 2))
      words.extend(random.sample(events_list, 2))
  elif facet == "Whens":
      words.extend(random.sample(when_list, 7))
      words.extend(random.sample(what_list, 2))
      words.extend(random.sample(where_list, 2))
      words.extend(random.sample(who_list, 2))
      words.extend(random.sample(events_list, 2))
  elif facet == "Whos":
      words.extend(random.sample(who_list, 7))
      words.extend(random.sample(what_list, 2))
      words.extend(random.sample(where_list, 2))
      words.extend(random.sample(when_list, 2))
      words.extend(random.sample(events_list, 2))
  elif facet == "Events":
      words.extend(random.sample(events_list, 7))
      words.extend(random.sample(what_list, 2))
      words.extend(random.sample(where_list, 2))
      words.extend(random.sample(when_list, 2))
      words.extend(random.sample(who_list, 2))
    
  return words
  """


@app.get("/get_facet_explanation/{sessionId}/{selectedSystem}/{source}/{facet}/{article_no}")
def get_facet_explanation(sessionId,selectedSystem,source,facet,article_no):
  facet, label = facet.split(' ')
  logger.info('Fetching facet explanation for ' + str(selectedSystem) + ' and source' + str(source) + ' for facet ' + str(facet) + ' article num ' + str(article_no))
  #if((selectedSystem == 'System Red' and source == 'R5') or (selectedSystem == 'System Blue' and source == 'TREC')):
  logger.info('Getting correct facet explanation')
  if('Whats' in facet):
    words_df = pd.read_parquet(backend_path + sessionId + '/' + source + '/all_whats_k_x_means_labelled.parquet.gzip')
  elif ('Wheres' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_wheres_k_x_means_labelled.parquet.gzip')
  elif ('Events' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_events_k_x_means_labelled.parquet.gzip')
  elif ('Whens' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_whens_k_x_means_labelled.parquet.gzip')
  elif ('Whos' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_whos_k_x_means_labelled.parquet.gzip')
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
  #else:
    #logger.info('Getting random facet explanation')
    #all_words = generate_random_words(facet)
  data = []
  
  # check if the word cloud is empty
  if len(all_words) == 0:
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(['Empty Words']))
  else :
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(all_words))

  # Save the word cloud to a buffer
  buf = io.BytesIO()
  wordcloud.to_image().save(buf, format='PNG')

  # Encode the image as a base64 string
  img_str = base64.b64encode(buf.getvalue()).decode()

  for word in all_words:
     data.append({'text':word})
  
  api_response = {'word_cloud': img_str, 'facet_words': data}
  return api_response


@app.get("/get_similar_words/{sessionId}/{source}/{facet}/{word}")
def get_most_similar_words(sessionId,source,facet,word):
  logger.info('Fetching most similar words for ' + str(source) + ' for facet ' + str(facet) + ' word ' + str(word))
  facet, label = facet.split(' ')
  all_words = []
  if('Whats' in facet):
    words_df = pd.read_parquet(backend_path + sessionId + '/' + source + '/all_whats_k_x_means_labelled.parquet.gzip')
  elif ('Wheres' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_wheres_k_x_means_labelled.parquet.gzip')
  elif ('Events' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_events_k_x_means_labelled.parquet.gzip')
  elif ('Whens' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_whens_k_x_means_labelled.parquet.gzip')
  elif ('Whos' in facet):
    words_df = pd.read_parquet(backend_path+ sessionId + '/' + source + '/all_whos_k_x_means_labelled.parquet.gzip')
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
    filepath = backend_path+ sessionId + '/'+source+'/all_whats_k_x_means_labelled.parquet.gzip'
  elif ('Wheres' in facet):
    filepath = backend_path+ sessionId + '/'+source+'/all_wheres_k_x_means_labelled.parquet.gzip'
  elif ('Events' in facet):
    filepath = backend_path+ sessionId + '/'+source+'/all_events_k_x_means_labelled.parquet.gzip'
  elif ('Whens' in facet):
    filepath = backend_path+ sessionId + '/'+source+'/all_whens_k_x_means_labelled.parquet.gzip'
  elif ('Whos' in facet):
    filepath = backend_path+ sessionId + '/'+source+'/all_whos_k_x_means_labelled.parquet.gzip'
  words_df = pd.read_parquet(filepath)
  word_events = words_df.loc[words_df.Parent_Words==word, 'Events']
  words_df.loc[words_df.Events.isin(word_events), 'k_labels'] = int(new_label)
  words_df.to_parquet(filepath,compression='gzip')


def get_feature_vectors(data,label_feature,sessionId,source):
  source_path = backend_path + sessionId + '/' + source + '/' + 'result.parquet.gzip'
  source_df = pd.read_parquet(source_path)
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

def get_all_feature_vectors(sessionId,source):
  files_list = ['all_events_k_x_means_labelled','all_whats_k_x_means_labelled','all_whens_k_x_means_labelled','all_wheres_k_x_means_labelled','all_whos_k_x_means_labelled']
  output_list = ['all_events_feature_vector','all_whats_feature_vector','all_whens_feature_vector','all_wheres_feature_vector','all_whos_feature_vector']
  for i in range(5):
    file_input = backend_path + sessionId + '/' + source + '/' + files_list[i] + '.parquet.gzip'
    df = pd.read_parquet(file_input)
    k_vectors = get_feature_vectors(df,'k_means',sessionId,source)
    k_vector_df = pd.DataFrame(k_vectors)
    k_vector_df.columns = k_vector_df.columns.astype(str)
    output_file = backend_path+ sessionId + '/' + source + '/' + output_list[i] + '_kmeans.parquet.gzip'
    k_vector_df.to_parquet(output_file,compression='gzip')

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
  data_weighted = data
  kmeans = KMeans(n_clusters=k, random_state=0).fit(data_weighted)
  data_labels = kmeans.predict(data_weighted)
  #data_transformed = kmeans.transform(data)
  return data_labels

def get_tsne(data,perplexity_value):
  tsne = TSNE(n_components=2, learning_rate='auto',init='random',perplexity=perplexity_value)
  components = tsne.fit_transform(data)
  return components

def get_clustering_labels(data,k_config=0,elbow_config=0):
  if(k_config==0):
    k = get_elbow_k(data,elbow_config) 
  else:
    k=k_config
  k_means_data_labels = get_kmeans_clusters(k,data)
  result = pd.DataFrame(k_means_data_labels,columns=['k_labels'])
  return result

def get_book_level_clustering(k_final_vectors,k):
  k_result = get_clustering_labels(k_final_vectors,k)
  return k_result

def find_normalized_value(values):
  min_value = np.min(values)
  max_value = np.max(values)
  normalized_value = (values - min_value) / (max_value - min_value)
  return normalized_value

def run_logistic_regression(x_train, y_train):
  clf = LogisticRegression()
  clf.fit(x_train, y_train)
  values = clf.coef_[0]
  normalized_value = find_normalized_value(values)
  return normalized_value


def calculate_weights(selectedSystem,source,feature_sizes_k,x_train,y_train,global_weights,increase_local_weights=None,decrease_local_weights=None):
  features = ['Whats 0','Whens 0','Wheres 0','Whos 0']
  all_weight = np.ones(len(feature_sizes_k))
  #print(all_weight)
  if((selectedSystem == 'System Red' and source == 'R5') or (selectedSystem == 'System Blue' and source == 'TREC')):
    logger.info('Actual Weights')
    if(x_train is not None):
      similarity_weights = run_logistic_regression(x_train,y_train)
      resultant_vector = similarity_weights
    n = 0
    for i,weight in enumerate(global_weights):
      weight = int(weight)
      try:
        final_indices = feature_sizes_k.index(features[i])
      except:
        final_indices = len(feature_sizes_k)
      if weight==1:
        weight=0.5
      elif weight==2:
        weight=1
      elif weight==4:
        weight=5
      all_weight[n:final_indices] = np.full(len(all_weight[n:final_indices]),weight)
      n = final_indices
    #print(all_weight)
    all_weight_set = set(all_weight)
    global_value = all_weight
    if(len(all_weight_set)!=1):
      global_value = find_normalized_value(all_weight)
      resultant_vector = similarity_weights + global_value / 2
    if increase_local_weights!=None:
      increase_weights = [feature_sizes_k.index(local_weight) for local_weight in increase_local_weights]
      for i in increase_weights:
        resultant_vector[i] = resultant_vector[i] * (4)
    if decrease_local_weights!=None:
      decreased_weights = [feature_sizes_k.index(local_weight) for local_weight in decrease_local_weights]
      for i in decreased_weights:
        resultant_vector[i] = resultant_vector[i] * (0.5)
    #print(resultant_vector)
    return resultant_vector
  elif(selectedSystem == 'System Red' and source == 'TREC'):
    logger.info('Random Weights for Red and TREC')
    #random_nums = np.random.uniform(-5000, 5000, size=(6000, 1))
    random_array = np.random.uniform(0, 500, size=len(all_weight))
    all_weight = np.where(all_weight == 1, random_array, all_weight)
    return all_weight
  elif(selectedSystem == 'System Blue' and source == 'R5'):
    logger.info('Random Weights for Blue and R5')
    #random_nums = np.random.uniform(-5000, 5000, size=(6807, 1))
    random_array = np.random.uniform(0, 500, size=len(all_weight))
    all_weight = np.where(all_weight == 1, random_array, all_weight)
    return all_weight

def generate_training_data(data,relevance,not_relevance):
  x_train = data[np.concatenate((relevance, not_relevance))]
  y_train = np.concatenate((np.ones(len(relevance)), np.zeros(len(not_relevance))))
  return x_train, y_train

def get_final_vectors(sessionId,selectedSystem,feature_sizes_k,relevant_docs,not_relevant_docs,global_weights,source,increase_local_weights,decrease_local_weights):
    files_list = ['all_events_feature_vector', 'all_whats_feature_vector', 'all_whens_feature_vector', 'all_wheres_feature_vector', 'all_whos_feature_vector']
    base_path = backend_path + sessionId + '/' + source + '/{}_kmeans.parquet.gzip'
    dfs = [pd.read_parquet(base_path.format(f)) for f in files_list]
    k_final_vectors = np.concatenate([df.values for df in dfs], axis=1)
    x_train = None
    y_train = None
    if(len(relevant_docs)>0):
      x_train, y_train = generate_training_data(k_final_vectors,relevant_docs,not_relevant_docs)
    importance = calculate_weights(selectedSystem,source,feature_sizes_k,x_train,y_train,global_weights,increase_local_weights,decrease_local_weights)
    logger.info('The importance vectors are ' + str(importance))
    if importance is None:
        importance = np.ones(k_final_vectors.shape[1])
    k_final_vectors = k_final_vectors * importance
    return k_final_vectors


def get_nearest_neighbours(relevant_docs,res_df):
  relevant_docs_list = list(relevant_docs)
  k = 7  # Number of nearest neighbors to find
  nn = NearestNeighbors(n_neighbors=k)
  all_points = res_df[['x_axis', 'y_axis']].values
  nn.fit(all_points)
  nearest_neighbors = []
  # Find the nearest neighbors for each relevant point
  for doc_index in relevant_docs_list:
      # Get the coordinates of the relevant point
      relevant_point = res_df.loc[doc_index, ['x_axis', 'y_axis']].values.reshape(1, -1)

      # Find the indices of the nearest neighbors for the relevant point
      _, nearest_indices = nn.kneighbors(relevant_point)

      # Retrieve the rows corresponding to the nearest neighbor indices
      nearest_neighbors.append(res_df.iloc[nearest_indices.flatten()].index.tolist())
  return nearest_neighbors


@app.get("/recluster_words/{sessionId}/{selectedSystem}/{source}/{feature_sizes_k}/{relevant_docs}/{not_relevant_docs}/{global_weights}/{increase_local_weights}/{decrease_local_weights}")
def recluster(sessionId,selectedSystem,source,feature_sizes_k,relevant_docs,not_relevant_docs,global_weights,increase_local_weights=None,decrease_local_weights=None,):
  logger.info('Reclustering for ' + str(selectedSystem) +' for source ' + str(source) +' for relevant docs ' + str(relevant_docs)
               + ' for non-relevant docs ' + str(not_relevant_docs) + ' for global_weights ' + str(global_weights) + ' for increase_local_weights ' + str(increase_local_weights) + ' for decrease_local_weights ' + str(decrease_local_weights)) 
  output_path = backend_path + sessionId + '/' + source + '/result.parquet.gzip'
  previous_output_path = backend_path + sessionId + '/' + source + '/prev_result.parquet.gzip'
  feature_sizes_k = json.loads(feature_sizes_k)
  global_weights = json.loads(global_weights)
  relevant_docs = json.loads(relevant_docs)
  not_relevant_docs = json.loads(not_relevant_docs)
  increase_local_weights = json.loads(increase_local_weights) if increase_local_weights else None
  decrease_local_weights = json.loads(decrease_local_weights) if decrease_local_weights else None
  get_all_feature_vectors(sessionId,source)
  if(len(relevant_docs)>0):
    result_df = pd.read_parquet(output_path)
    old_relevance = result_df[(result_df['highlight'] == 1.0) & (result_df['relevance'] == 1.0)]
    #set1 = set(old_relevance.article_no.to_list())
    #set2 = set(relevant_docs)
    #intersection = set1.intersection(set2)
    #print(old_relevance)
    logger.info('Old Relevance documents are ' + str(old_relevance.article_no.to_list()))
    all_docs = result_df[result_df.highlight==1].index
    if(len(not_relevant_docs)==0):
      not_relevant_docs = [doc for doc in all_docs if doc not in relevant_docs and doc in result_df[result_df.highlight==1].index]
    percentage = (len(relevant_docs) / (len(relevant_docs) + len(not_relevant_docs))) * 100
    logger.info(f"Percentage of list2 elements present in list1: {percentage:.2f}%")
    logger.info('Relevant documents are ' + str(relevant_docs))
    #print(relevant_docs)
    logger.info('All documents are ' + str(all_docs))
    #print(all_docs)
    logger.info('Not Relevant documents are ' + str(not_relevant_docs))
    #print(not_relevant_docs)
  k_final_vectors = get_final_vectors(sessionId,selectedSystem,feature_sizes_k,relevant_docs,not_relevant_docs,global_weights,source,increase_local_weights,decrease_local_weights)
  if(source=='R2'):
    labels_df = get_book_level_clustering(k_final_vectors,2)
  elif(source=='R5'):
    labels_df = get_book_level_clustering(k_final_vectors,5)
  elif(source=='TREC'):
    labels_df = get_book_level_clustering(k_final_vectors,11)
  tsne_df = pd.DataFrame(get_tsne(k_final_vectors,20),columns=['x_axis','y_axis'])
  result = pd.concat([pd.DataFrame(k_final_vectors),labels_df,tsne_df],axis=1)
  nearest_neighbors = get_nearest_neighbours(relevant_docs,result)
  logger.info('Nearest Neighbour ' + str(nearest_neighbors))
  #print(nearest_neighbors)
  if(len(relevant_docs)>0):
    result['relevance'] = np.zeros(len(result))
    result['highlight'] = np.zeros(len(result))
    result.loc[relevant_docs, 'highlight'] = 1
    result.loc[relevant_docs, 'relevance'] = 1
    for indices in nearest_neighbors:
      result.loc[indices,'highlight'] = 1
      result.loc[indices,'relevance'] = 1
    result.loc[not_relevant_docs, 'relevance'] = 0
  else:
    result['relevance'] = np.ones(len(result))
    result['highlight'] = np.zeros(len(result))
  result['article_no'] = result.index
  result.columns = result.columns.astype(str)
  result.to_parquet(output_path,compression='gzip')
  old_relevance.to_parquet(previous_output_path,compression='gzip')
  return {'message': 'Success'}

@app.get("/logout/{sessionId}")
def logout(sessionId):
  # Get a list of all files in the folder
  session_path = os.path.join(parent_dir, sessionId)

  subfolders = [f for f in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, f))]
  # Iterate over the subfolders
  for subfolder in subfolders:
      subfolder_path = os.path.join(session_path, subfolder)
      
      # Get a list of all files in the subfolder
      files = os.listdir(subfolder_path)
      
      # Iterate over the files
      for file_name in files:
          file_path = os.path.join(subfolder_path, file_name)
          
          # Check if the file is not "result.parquet.gzip" or "prev_result.parquet.gzip"
          if file_name != "result.parquet.gzip" and file_name != "prev_result.parquet.gzip":
              # Delete the file
              os.remove(file_path)