import numpy as np
import csv
import sys
import math
 
from validate import validate
  
def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter = ',', dtype = np.float64, skip_header = 1)
    return test_X
 
 
def compute_ln_norm_distance(vector1, vector2, n):
    diff = np.abs(vector1-vector2)
    diff_power_n = np.power(diff,n)
    sum_diff = np.sum(diff_power_n)
    ans = np.power(sum_diff,1/n)
    return ans
 
def find_k_nearest_neighbours(train_X,test_example, k, n):
    result_arr = []
    index = 0
    for var in train_X:
        dist = compute_ln_norm_distance(var,test_example,n)
        result_arr.append([index,dist])
        index += 1
    result_arr.sort(key = lambda x: x[1])
    nearest_k_pairs = result_arr[:k]
    nearest_k_pairs_indices = [var[0] for var in nearest_k_pairs]
    nearest_k_pairs_indices = np.array(nearest_k_pairs_indices)
    return nearest_k_pairs_indices
 
 
def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_element_x in test_X:
        k_nn_indices = find_k_nearest_neighbours(train_X,test_element_x, k, n)
        k_nn_classes = []
        for i in k_nn_indices:
            k_nn_classes.append(train_Y[i])
        all_diff_classes = list(set(k_nn_classes))
        max_count = 0
        most_freq_class = -1
        for a_class in all_diff_classes:
            count = k_nn_classes.count(a_class)
            if count > max_count:
                max_count = count
                most_freq_class = a_class
        test_Y.append(most_freq_class)
    test_Y = np.array(test_Y)
    return test_Y
         
     
     
def calculate_accuracy(predicted_Y, actual_Y):
    tot_matched = 0
    for i in range(len(predicted_Y)):
        if(predicted_Y[i] == actual_Y[i]):
            tot_matched += 1
         
    accuracy = (tot_matched) / len(predicted_Y)
    return accuracy
 
 
def get_best_k_using_validation_set(train_X, train_Y, validation_split_percentage, n):
    tot_observations = len(train_X)
    train_len = math.floor( float(100-validation_split_percentage)/100 *tot_observations )
    new_train_X = train_X[0:train_len]
    new_train_Y = train_Y[0:train_len]
    validation_X = train_X[train_len:]
    validation_Y = train_Y[train_len:]
     
    best_k =- 1
    best_accuracy = 0
    for k in range(1,train_len):
        predicted_Y = classify_points_using_knn(new_train_X, new_train_Y, validation_X, n, k)
        accuracy = calculate_accuracy(predicted_Y,validation_Y)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    return best_k
 
 
 
def predict_target_values(test_X):
    train_X = np.genfromtxt("train_X_knn.csv", delimiter = ',', dtype = np.float64, skip_header = 1)
    train_Y = np.genfromtxt("train_Y_knn.csv", delimiter = ',', dtype = np.float64)
    k = get_best_k_using_validation_set(train_X, train_Y, 30, n = 2)
    pred_Y = classify_points_using_knn(train_X, train_Y, test_X, k, n = 2)
    return pred_Y
     
     
 
def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()
 
 
def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")
 
 
if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv")