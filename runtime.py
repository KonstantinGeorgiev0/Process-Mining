import pandas as pd
import time
import datetime
import splitter
import constants
# from decision_tree import *
from naive_time_estimation import *
from preprocessing import *
from naive_event_estimation import *

def main():
    # runtime preprocessing
    # merge the data sets and preprocess
    # train2 = pd.concat([train, test], axis="rows", ignore_index=True)
    # start_time_pre = time.time()
    # train2 = preprocessing(train)
    # end_time_pre = time.time()
    # preprocessing_runtime = end_time_pre - start_time_pre

    # runtime naive event estimation on train data
    start_time_naive_event = time.time()
    naive_est(case_ids)
    end_time_naive_event = time.time()
    naive_event_train_runtime = end_time_naive_event - start_time_naive_event

    # runtime naive event estimation on test data
    start_time_naive_event = time.time()
    naive_est(case_ids_test)
    end_time_naive_event = time.time()
    naive_event_test_runtime = end_time_naive_event - start_time_naive_event

    # runtime naive time estimation on train data
    start_time_naive_time = time.time()
    naive_time_estimator(train, test)
    end_time_naive_time = time.time()
    naive_time_train_runtime = end_time_naive_time - start_time_naive_time

    # runtime naive time estimation on test data
    start_time_naive_time = time.time()
    naive_time_estimator(train, test)
    end_time_naive_time = time.time()
    naive_time_test_runtime = end_time_naive_time - start_time_naive_time

    # print(f"Preprocessing runtime: {preprocessing_runtime} seconds")
    print(f"Naive event estimation on train data runtime: {naive_event_train_runtime} seconds")
    print(f"Naive event estimation on test data runtime: {naive_event_test_runtime} seconds")
    print(f"Naive time estimation on train data runtime: {naive_time_train_runtime} seconds")
    print(f"Naive time estimation on test data runtime: {naive_time_test_runtime} seconds")
    
    # start_time = time.time()
    # train_data, test_data = splitter.split_data(constants.DATASET_PATH, constants.RANDOM_SEED)

    # preprocess_dataset(train_data, test_data)

    # timer = splitter.Timer()

    # compare_all_models(train_data, test_data, timer)

    # naive_time_estimator(train_data, test_data)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {str(datetime.timedelta(seconds=elapsed_time))}")


if __name__ == '__main__':
    main()
