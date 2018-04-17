import scipy.io
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
import time

mat = scipy.io.loadmat('./data_hyper.mat')
node_data = {}
headnode = tuple(range(14))



def build_dict(arr):
	temp = {}
	for i in range(len(arr)):
		temp[i] = arr[i][0]
	return temp

def get_means(arr, lt=False):
	means = {}
	for key in arr.keys():
		means[key] = np.mean(arr[key], axis=0)
	return means


def get_means_list(train_data, current_node):
	means = []
	for i in current_node:
		means.append(np.mean(train_data[i], axis=0))
	return means

def construct_node_data(node_data, current_node, N):
	
	print node_data.keys()

	if len(current_node) <= 1:
		return

	node_data[current_node] = {}

	# Build Training Data
	means_list = get_means_list(train_data, current_node)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(means_list)
	means_classes = kmeans.predict(means_list)
	training_feats = []
	H_train = []

	for i,key in enumerate(current_node):
		training_feats.extend(train_data[key])
		l = train_data[key].shape[0]
		print l
		temp = [0,0]
		temp[means_classes[i]] = 1
		H_train.extend([temp]*l)

	left_classes = []
	right_classes = []

	for i in range(len(current_node)):
		if means_classes[i] == 0:
			left_classes.append(current_node[i])
		else:
			right_classes.append(current_node[i])

	node_data[current_node]["H_train"] = np.transpose(np.array(H_train))
	node_data[current_node]["training_feats"] = np.transpose(np.array(training_feats))
	node_data[current_node][0] = tuple(left_classes)
	node_data[current_node][1] = tuple(right_classes)

	construct_node_data(node_data, node_data[current_node][0], N)
	construct_node_data(node_data, node_data[current_node][1], N)

def construct_svm_all_nodes(node_data, kernel="poly"):
	for current in node_data.keys():
		data = node_data[current]
		train = np.transpose(data["H_train"])
		training_feats = np.transpose(data["training_feats"])
		H_train = []
		for temp in train:
			if temp[0] == 1:
				H_train.append(0)
			else:
				H_train.append(1)
		H_train = np.array(H_train)
		clf = svm.SVC(kernel = kernel)
		clf.fit(training_feats, H_train)
		node_data[current]["classifier"] = clf

def classify_datapoint(datapoint, node_data = node_data, headnode = headnode):
	current_node = headnode

	while len(current_node) > 1:
		clf = node_data[current_node]["classifier"]
		prediction = clf.predict(datapoint)[0]
		print "Prediction:",prediction
		current_node = node_data[current_node][prediction]
	return current_node[0]

def predict(datapoints):
	result = []
	for datapoint in datapoints:
		result.append(classify_datapoint(datapoint))
	return result

def get_accuracy(test_data):
	total = 0
	correct = 0
	for key in test_data.keys():
		class_data = test_data[key]
		for datapoint in class_data:
			total += 1
			if classify_datapoint(datapoint) == key:
				correct += 1
	accuracy = float(correct)/float(total)
	print "****"*20,"\n",correct,"samples correctly classified out of total",total,"samples","\n","****"*20
	return accuracy

def construct_test_data(test_data):
	H_test = []
	testing_feats = []
	for key in test_data.keys():
		testing_feats.extend(test_data[key])
		l = test_data[key].shape[0]
		print l
		temp = [0,0]
		H_test.extend([temp]*l)
	H_test = np.transpose(np.array(H_test))
	testing_feats = np.transpose(np.array(testing_feats))

def set_prediction_data_lcksvd():
	for key_tuple in node_data.keys():
		filename = "prediction_"+str(key_tuple)[1:-1]+".mat"
		prediction = ((scipy.io.loadmat(filename)['prediction2']) - 1)
		node_data[key_tuple]["prediction"] = prediction

def get_prediction_lcksvd(testing_feats, node_data, headnode):
	testing_data = np.transpose(testing_feats)
	point_index = 0
	correct = 0
	for key in test_data.keys():
		for point in test_data[key]:
			current_node = headnode
			while len(current_node) > 1:
				print node_data[current_node]["prediction"],point_index
				pred = node_data[current_node]["prediction"][0][point_index]
				print "Prediction:",pred
				current_node = node_data[current_node][pred]
			if current_node[0] == key:
				correct += 1
			point_index += 1
	accuracy = float(correct)/float(point_index)
	print "****"*20,"\n",correct,"samples correctly classified out of total",point_index,"samples","\n","****"*20
	print "Accuracy:",accuracy
	return accuracy


def matlab_vaali_backchodi(sleep_time = 150):
	for key_tuple in node_data.keys():
		data = node_data[key_tuple]
		H_train = data["H_train"]
		training_feats = data["training_feats"]
		
		dest = "c:/tmp/arrdata.mat"
		scipy.io.savemat(dest, mdict={'H_train': H_train, 'training_feats': training_feats, 'H_test': H_test, 'testing_feats': testing_feats})

		# Call Matlab Here
		#######################################

		time.sleep_time(sleep_time)
		
		# Save the contents
		temp = scipy.io.loadmat("./trainingdata/predictions.mat")
		filename = "prediction_"+str(key_tuple)[1:-1]+".mat"
		scipy.io.savemat(filename,temp)


train_data = build_dict(mat['TR1'])
test_data = build_dict(mat['TS1'])


construct_node_data(node_data, headnode, 14)

test_svm = False

if test_svm:
	construct_svm_all_nodes(node_data)

	accuracy = get_accuracy(test_data)

	print "Accuracy:",accuracy


# Build test data
H_test = []
testing_feats = []
for key in test_data.keys():
	testing_feats.extend(test_data[key])
	l = test_data[key].shape[0]
	print l
	if key == 0 or key == 1 or key == 6:
		temp = [0,1]
	else:
		temp = [1,0]
	H_test.extend([temp]*l)
H_test = np.transpose(np.array(H_test))
testing_feats = np.transpose(np.array(testing_feats))



set_prediction_data_lcksvd()
get_prediction_lcksvd(testing_feats, node_data, headnode)