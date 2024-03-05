import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import multiprocessing

### sklearn dependencies
import sklearn
from sklearn.mixture import BayesianGaussianMixture

### dependency vars
random_state = 42
tfidf_m = np.load('/home/datallah/datallah-jaymefis-gibsonce/bgmm/tfidf_trunc.npy')
tfidf_m_val = np.load('/home/datallah/datallah-jaymefis-gibsonce/bgmm/tfidf_trunc_val.npy')

# init variables
max_components = list(np.arange(1, 50 + 1, 2))
# lst.append(1)
# max_components = sorted(lst)
num_threads = 4

# init empty results df
doc_path = '/home/datallah/datallah-jaymefis-gibsonce/bgmm/large_step_sensitivity.csv'
if os.path.exists(doc_path) == False:
    df = pd.DataFrame(columns = ['n_components', 'max_log_likelihood', 'train_time'])
    df.to_csv(doc_path, sep = ',', index = False)
    del df

# create function to loop over
def fitter(n_components, train_m = tfidf_m, val_m = tfidf_m_val, random_state = random_state):
    # read csv
    temp_df = pd.read_csv(doc_path, sep = ',')
    if temp_df[temp_df['n_components'] == n_components].shape[0] == 0:
        # get time
        time_start = datetime.now()
        # init model
        bgmm = BayesianGaussianMixture(n_components = n_components,
                                       random_state = random_state, 
                                       max_iter = 1000,
                                       warm_start = True,
                                       verbose = 1)
        bgmm.fit(train_m)
        # calculate log likelihood of the current model
        log_likelihood = bgmm.score(tfidf_m_val)
        # get time
        time_end = datetime.now()
        train_time = (time_end - time_start).total_seconds()
        # write to csv
        temp_df = pd.read_csv(doc_path, sep = ',')
        cols = list(temp_df.columns)
        append_df = pd.DataFrame([n_components, log_likelihood, train_time], index = cols).T
        temp_df = pd.concat([temp_df, append_df], ignore_index = True)
        temp_df.to_csv(doc_path, sep = ',', index = False)
        # print message
        print(f"Number of components: {n_components}, Max Log likelihood: {log_likelihood}")
    else:
        print(f'Already trained for {n_components} components.')

# Create a ThreadPoolExecutor with the desired number of threads
with multiprocessing.Pool(num_threads) as executor:
    # Map the function to the list of items, running them in parallel
    results = list(executor.map(fitter, max_components))