# dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
nlp = spacy.load("en_core_web_sm")
# STALE: from spacy.lang.en.stop_words import STOP_WORDS as stop
import nltk
from nltk.tokenize.casual import TweetTokenizer
# initialize tweet tokenizer
tt = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True, match_phone_numbers=False)
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()
import re

# intialize stop_words with pronouns commented out
stop_words = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 #'he',
 #'her',
 'here',
 #'hers',
 #'herself',
 #'him',
 #'himself',
 #'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 #'she',
 #"she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'}

# specify data filepath based on env (GLC or Colab)
def filepath():
  """
  Identifies whether the user is running notebooks from GLC or Colab and initializes filepath.
  """
  from os import environ
  env_var = environ.keys()
  if 'CLUSTER_NAME' in env_var:
    return '/home/datallah-jaymefis-gibsonce/'
  elif 'COLAB_JUPYTER_IP' in env_var:
    from google.colab import drive
    drive.mount('/content/drive')
    return '/content/drive/MyDrive/RtGender/'

# specify a file or list of files to load and process
def extract_dfs(filepath, filenames):
  """
  Accepts a string or list of filenames to read as DataFrames.

  Parameters:
  - filepath: path to shared drive where data files reside.
  - filenames: a string or list of strings corresponding to RtGender files.

  Returns:
  - df_dict: a dictionary of DataFrames where each key, value pair correspond to an RtGender file.
  """
  if type(filenames) == str: filenames = [filenames]
  df_dict = {}
  # iterate and load dict with dfs
  for file in filenames:
    file_temp = file[:-4] if file[-4:0] == '.csv' else file
    df_temp = pd.read_csv(filepath + file_temp + '.csv').reset_index()
    # response_text can occassionally be null, we want to remove these records
    df_temp.dropna(subset = ['response_text', 'op_gender'], inplace = True)
    df_temp['sourceID'] = df_temp['source'] + df_temp['index'].astype(str)
    df_dict[file] = df_temp
    return df_dict

# split dfs and save to drive
def load_df(filepath, df, name = None,
  train_size = 0.6, val_size = 0.2, test_size = 0.2):
  """
  Take extracted DataFrame and write it to a shared location.

  Parameters:
  - filepath: path to shared drive where data files reside.
  - df: a DataFrame to be written to the shared location.
  - train_size: size of train set proportional to overall df.
  - val_size: size of validation set proportional to overall df.
  - test_size: size of test set proportional to overall df.
  - name: optional string prefix for the filename.

  Returns:
  - load_dict: dictionary of load paths
  """
  if train_size + val_size + test_size != 1.0:
    raise ValueError("Train, test, and validation splits must sum to 1.")
  X = df.loc[:, df.columns != 'op_gender']
  y = df['op_gender']
  X_train_temp, X_test, y_train_temp, y_test = train_test_split(
      X, y, test_size = test_size, random_state = 42, 
      stratify = X['source'])
  X_train, X_val, y_train, y_val = train_test_split(
      X_train_temp, y_train_temp, 
      test_size = val_size/(train_size + val_size),
      random_state = 42)
  name = name + '_' if name is not None else ''
  load_dict = {}
  load_dict['X_train'] = filepath + 'trns/' + name + 'X_train.csv'
  X_train.to_csv(load_dict['X_train'])
  load_dict['X_test'] = filepath + 'trns/' + name + 'X_test.csv'
  X_test.to_csv(load_dict['X_test'])
  load_dict['X_val'] = filepath + 'trns/' + name + 'X_val.csv'
  X_val.to_csv(load_dict['X_val'])
  load_dict['y_train'] = filepath + 'trns/' + name + 'y_train.csv'
  y_train.to_csv(load_dict['y_train'])
  load_dict['y_test'] = filepath + 'trns/' + name + 'y_test.csv'
  y_test.to_csv(load_dict['y_test'])
  load_dict['y_val'] = filepath + 'trns/' + name + 'y_val.csv'
  y_val.to_csv(load_dict['y_val'])
  return load_dict

# create function to remove nouns using POS tagging
def subj_removal(doc_s):
  """
  Take a string document and apply subject removal. 

  Parameters:
  - doc: a string document or a series of string documents.

  Returns:
  - A string document, or list of string documents, where subjects are removed.
  """
  # if dealing with one document / iterating through documents
  if type(doc_s) == str:
    tag_doc = nlp(doc_s)
    filtered_doc = [tok.text for tok in tag_doc if tok.dep_ != 'nsubj']
    return ' '.join(filtered_doc)
  # for speed in processing many docs
  elif len(list(doc_s)) > 1:
    docs = list(nlp.pipe(doc_s))
    docs_lst = []
    for doc in docs:
      processed_doc = ' '.join([tok.text for tok in doc if tok.dep_ != 'nsubj'])
      docs_lst.append(processed_doc)
    return docs_lst

# flexible preprocessing 
def preprocess(filepath, align_case = True, subj_rm = False, 
  rm_stopwords = True, lemmatize = True, name = None):
  """
  Take split datasets from shared location and apply preprocessing.

  Parameters:
  - filepath: full path to data files to be preprocessed.
  - align_case: if true, response_text field will be aligned to lower case.
  - subj_rm: if true, all tagged subjects will be removed from response_text.
  - rm_stopwords: if true, stopwords will be removed from response_text.
  - lemmatize: if true, response_text will be lemmatized.
  - name: optional string prefix for the filename.

  Returns:
  - df: DataFrame with preprocessed response text column
  """
  # begin preprocessing...start by loading df and creating working column
  df = pd.read_csv(filepath).iloc[:, 1:]
  df['processed_response'] = df.response_text
  # if align_case is true lowercase the whole response column
  if align_case:
    df['processed_response'] = df.processed_response.str.lower()
  # if subj_rm is true, use POS tagging to remove subjects from each doc
  if subj_rm:
    df['processed_response'] = subj_removal(df.processed_response)
  # tokenize column using tweet tokenizer for future steps
  df['processed_response'] = df['processed_response'].apply(
    lambda x: [wrd for wrd in tt.tokenize(x)])
  # STALE: tokenize column using regex for any future steps
  # df['processed_response'] = df['processed_response'].apply(
  #   lambda x: [wrd for wrd in re.findall(r'(?u)\b\w\w+\b', x)])
  # if rm_stopwords is true remove stopwords from the whole response column
  if rm_stopwords:
    df['processed_response'] = df.processed_response.apply(
      lambda x: [word for word in x if word.lower() not in (stop)])
  # if lemmatize is true then lemmatize the whole response column
  if lemmatize:
    df['processed_response'] = df.processed_response.apply(
      lambda x: [wnl.lemmatize(word) for word in x])
  # load preprocessed df for later use
  parent_path = filepath[:filepath.index('trns')] + 'trns/' if 'trns' in filepath else filepath
  name = name if '.csv' in name else name + '.csv'
  df.to_csv(parent_path + name)
  return df