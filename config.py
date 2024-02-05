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
    import pandas as pd
    file_temp = file[:-4] if file[-4:0] == '.csv' else file
    df_temp = pd.read_csv(filepath + file_temp + '.csv').reset_index()
    # response_text can occassionally be null, we want to remove these records
    df_temp.dropna(subset = ['response_text', 'op_gender'], inplace = True)
    df_temp['sourceID'] = df_temp['source'] + df_temp['index'].astype(str)
    df_dict[file] = df_temp
    return df_dict

# split dfs and save to drive
def load_df(filepath, df, name = None):
  """
  Take extracted DataFrame and write it to a shared location.

  Parameters:
  - filepath: path to shared drive where data files reside.
  - df: a DataFrame to be written to the shared location.
  - name: optional string prefix for the filename.

  Returns:
  - load_dict: dictionary of load paths
  """
  X = df.loc[:, df.columns != 'op_gender']
  y = df['op_gender']
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size = 0.2, random_state = 42)
  name = name + '_' if name is not None else ''
  load_dict = {}
  load_dict['X_train'] = filepath + 'trns/' + name + 'X_train.csv'
  X_train.to_csv(load_dict['X_train'])
  load_dict['X_test'] = filepath + 'trns/' + name + 'X_test.csv'
  X_test.to_csv(load_dict['X_test'])
  load_dict['y_train'] = filepath + 'trns/' + name + 'y_train.csv'
  y_train.to_csv(load_dict['y_train'])
  load_dict['y_test'] = filepath + 'trns/' + name + 'y_test.csv'
  y_test.to_csv(load_dict['y_test'])
  return load_dict


# flexible preprocessing 
def preprocess(filepath, align_case = True, rm_stopwords = True,
  lemmatize = True, name = None):
  """
  Take split datasets from shared location and apply preprocessing.

  Parameters:
  - filepath: full path to data files to be preprocessed.
  - align_case: if true, response_text field will be aligned to lower case.
  - rm_stopwords: if true, stopwords will be removed from response_text.
  - lemmatize: if true, response_text will be lemmatized.
  - name: optional string prefix for the filename.

  Returns:
  - df: DataFrame with preprocessed response text column
  """
  import pandas as pd
  df = pd.read_csv(filepath).iloc[:, 1:]
  df['processed_response'] = df.response_text
  # if align_case is true lowercase the whole response column
  if align_case == True:
    df['processed_response'] = df.processed_response.str.lower()
  # if rm_stopwords is true remove stopwords from the whole response column
  if rm_stopwords == True:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS as stop
    df['processed_response'] = df.processed_response.apply(
      lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  # if lemmatize is true then lemmatize the whole response column
  if lemmatize == True:
    import nltk
    nltk.download('wordnet')
    from nltk.stem.wordnet import WordNetLemmatizer
    import re
    token_ser = df['processed_response'].apply(
      lambda x: [wrd for wrd in re.findall(r'(?u)\b\w\w+\b', x)])
    wnl = WordNetLemmatizer()
    df['processed_response'] = token_ser.apply(lambda x: [wnl.lemmatize(words) for words in x])
  if lemmatize == False:
    import re
    df['processed_response'] = df['processed_response'].apply(
      lambda x: [wrd for wrd in re.findall(r'(?u)\b\w\w+\b', x)])
  # load preprocessed df for later use
  parent_path = filepath[:filepath.index('trns')] + 'trns/' if 'trns' in filepath else filepath
  name = name if '.csv' in name else name + '.csv'
  df.to_csv(parent_path + name)
  return df