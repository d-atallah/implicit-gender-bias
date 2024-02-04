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
    df_temp['sourceID'] = df_temp['source'] + df_temp['index'].astype(str)
    df_dict[file] = df_temp
    return df_dict

# split dfs and save to drive
def load_dfs(filepath, df_dict):
  """
  Take preprocessed DataFrames and write them to a shared location.

  Parameters:
  - filepath: path to shared drive where data files reside.
  - df_dict: a dictionary of DataFrames to be written to the shared location.
  """
  for key in df_dict:
    df_temp = df_dict[key]
    X = df_temp.loc[:, df_temp.columns != 'op_gender']
    y = df_temp['op_gender']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42)
    X_train.to_csv(filepath + 'trns/' + key + '_X_train.csv')
    X_test.to_csv(filepath + 'trns/' + key + '_X_test.csv')
    y_train.to_csv(filepath + 'trns/' + key + '_y_train.csv')
    y_test.to_csv(filepath + 'trns/' + key + '_y_test.csv')