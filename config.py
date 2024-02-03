# specify data filepath based on env
def filepath():
  from os import environ
  env_var = os.environ.keys()
  if 'CLUSTER_NAME' in env_var:
    return '/home/datallah-jaymefis-gibsonce/'
  elif 'COLAB_JUPYTER_IP' in env_var:
    from google.colab import drive
    drive.mount('/content/drive')
    return '/content/drive/MyDrive/RtGender/'