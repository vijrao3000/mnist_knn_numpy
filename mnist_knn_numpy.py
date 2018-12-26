
import numpy as np
from sklearn import datasets

# Load Minst data set
def load_mnist_data():
  mnist = datasets.load_digits()
  return(list(zip(mnist.data, mnist.target)))

# Split data into taining and test sets (no validation set)
def get_split_data(data,training_data_portion,test_data_portion):
  num = len(data)
  num_trn, num_tst = num*training_data_portion, num*test_data_portion

  # split test into training, validation and test
  trn,tst = data[:int(num_trn)], data[int(num_trn):int(num_trn)+int(num_tst)]
  print("We have {} items in the training set & {} items for test".format(int(len(trn)),int(len(tst))))
  return trn,tst

# Compute and return sum of square distance
def compute_vec_distance(tr,tt):
  diff, td = 0, (tr[0]-tt[0])**2
  for ii in range(len(td)):
    diff = diff+td[ii]
  return diff

# Get the label for the least difference by taking a poll of k-neighbors
def get_label_from_k_closest_neighbors(tr, kcn, dfk, k):
  labels = []
  for ii in kcn:
    labels.append(tr[ii][1])
  counts = np.bincount(labels)
  return np.argmax(counts)

# Get the predicted value
def get_predicted_label(training_data, diff_values, k):
  diff_values = np.reshape(diff_values, (1,-1))
  diff_values = diff_values[0]
  idx = np.argpartition(diff_values, k)

  k_closest_neighbours = idx[:k]
  diff_for_k_closest = diff_values[idx[:k]]

  # get the labels for the k closest neighbors
  label = get_label_from_k_closest_neighbors(training_data, k_closest_neighbours, diff_for_k_closest, k)
  return label

# Calculate and check the minimum distance between test image and training images.
# Returns 1 if the inference was correct, 0 otherwise
def check_predicted_value_against_ground_truth(tr,tt,k):
  diff = np.zeros((len(tr),1))
  for ii in range(len(tr)):
    diff[ii] = compute_vec_distance(tr[ii],tt)
    
  # get prediction
  if tt[1] == get_predicted_label(training_data=tr,diff_values=diff,k=k):
    return 1
  return 0

# Run each of the test images against all training images and check the prediction
def run_test_set_against_training_data(trn,tst,k):
  correct_num = 0
  # pick an image from test data and compare against each image from the training set
  for ii in range(len(tst)):
    ret = check_predicted_value_against_ground_truth(trn,tst[ii],k)
    correct_num = correct_num + ret
  return correct_num

# main
if __name__ == "__main__":
  # load the data
  data = load_mnist_data()
  # split the data (75% training set and 25% test)
  trn,tst = get_split_data(data,training_data_portion=0.75, test_data_portion=0.25)

  k_vals = [1, 3, 5, 7, 9, 11, 17, 23, 37, 49]
  for ii, k in enumerate(k_vals):
    print("Iteration {:02d} With k = {:02d}".format(ii, k), end="  ")
    # run the test
    correct_num = run_test_set_against_training_data(trn,tst,k)
    # report
    print("==> Result: Accuracy {0:.2f}% ".format((correct_num/len(tst))*100),end="")
    print("({} correct predictions out of {}) ".format(correct_num,len(tst)))
