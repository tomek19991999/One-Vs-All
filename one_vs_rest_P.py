# one vs all algorithm 
from random import seed
from random import randrange
from csv import reader
import random
import matplotlib.pyplot as plt
import copy

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# make a prediction according to provided weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	if activation >= 0.0:
		return 1.0
	else: 
		return 0.0

# estimation of perceptron weights by SGD
def train_weights(train, learning_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + learning_rate * error # adjust bias	
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + learning_rate * error * row[i] # adjust weights
	return weights

# the actual perceptron code
def perceptron(train, test, learning_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, learning_rate, n_epoch) # train the perceptron
	# then use it on the test data
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# run perceptron and report results
def run_algorithm(train_set, test_set, learning_rate, n_epoch, plot_nr, flower_name):
	predicted = perceptron(train_set, test_set, learning_rate, n_epoch)
	data_visualisation(test_set, learning_rate, n_epoch, predicted, plot_nr, flower_name)

# data visulaisation and performance metrics
def data_visualisation(test_dataset, learning_rate, n_epoch, predictions, plot_nr, flower_name):
	
	# confusion matrix
	matrix=list()
	matrix=[[0 for i in range(2)] for j in range(2)]
	i = 0
	for row in test_dataset:
        #constructing the actual matrix
		if row[-1] == 0 and predictions[i] == 0.0:
			matrix[0][0] += 1 #tp
		elif row[-1] != 0 and predictions[i] == 0.0:
			matrix[0][1] += 1 #fp
		elif row[-1] == 0 and predictions[i] != 0.0:
			matrix[1][0] += 1 #fn
		elif row[-1] != 0 and predictions[i] != 0.0:
			matrix[1][1] += 1 #tn
		i = i + 1

	tp = matrix[0][0]
	fp = matrix[0][1]
	fn = matrix[1][0]
	tn = matrix[1][1]

	# confusion matrix table print
	axs[0, plot_nr].set_axis_off() 
	axs[0, plot_nr].set_title("performance metrics for\n" + flower_name + " flowers", fontweight ="bold")
	conf_matrix = axs[0, plot_nr].table( 
    	cellText = [[str(matrix[0][0]),str(matrix[1][1])], [str(matrix[0][1]),str(matrix[1][0])]],  
    	rowLabels = ['true', 'false'],  
    	colLabels = ['positive', 'negative'] , 
		colWidths = [0.4, 0.4],
    	rowColours =["lightblue"] * 2,  
    	colColours =["lightblue"] * 2, 
    	cellLoc ='center',  
    	loc ='center') 
	conf_matrix.scale(1,1.5) 

	# performance metrics
	accuracy = 1
	precision = 1
	recall = 1
	specificity = 1
	f_score = 1

	metrics = [	[str(round(accuracy,2)) + " %"],
				[str(round(precision,2)) + " %"],
				[str(round(recall,2)) + " %"],
				[str(round(specificity,2)) + " %"],
				[str(round(f_score,1))]]

	# performance metrics table print
	axs[1, plot_nr].set_axis_off() 
	perf_metrics = axs[1, plot_nr].table( 
    	cellText = metrics,  
    	rowLabels = ['accuracy', 'precision', 'recall', 'specificity', 'f-score'],   
		colWidths = [0.4],
    	rowColours =["lightblue"] * 5,   
    	cellLoc ='center',  
    	loc ='center') 
	perf_metrics.scale(1,1.5)

	# center the metrics table to the confusion matrix
	box = axs[1, plot_nr].get_position()
	box.x0 = box.x0 + 0.05
	axs[1, plot_nr].set_position(box)

def loading_txt (file):
    data=list()
    with open(file, 'r') as file:
        csvreader = reader(file)
        for row in csvreader:
            if not row:
                continue
            data.append(row)

    #make float data (before, we had string data)
    for row in data:
        for i in range(len(data[0])-1):
            row[i]=float(row[i])
            #print(row[i])
    return data


# preparing training and validation datasets in a one-vs-rest fashion
def prepare_subdataset(source_dataset, class_name, p, reversed = False):
	target_dataset = copy.deepcopy(source_dataset)
	for row in target_dataset:
		if row[-1] == class_name:
			row[-1] = 1
		else:
			row[-1] = 0
	output_dataset = list()
	dataset_range = range(round(len(target_dataset)*p))
	if (reversed == True):
		dataset_range = range(round(len(target_dataset)*p),0,-1)
	for row in dataset_range:
		output_dataset.append(target_dataset[row])
	return output_dataset


def take_one_vs_all_sets(training_data_all):
    data_setosa=copy.deepcopy(training_data_all) #Setosa_vs_all
    data_versicolor=copy.deepcopy(training_data_all)#versicolor_vs_all
    data_virginica=copy.deepcopy(training_data_all)#Virginica_vs_all
    length=len(training_data_all)
    for i in range(len(data_setosa)):
        if i < len(data_setosa) / 3:
            data_setosa[i][4] = 1
            data_versicolor[i][4]=0
            data_virginica[i][4]=0
        elif i < len(data_setosa) * 2 / 3:
            data_setosa[i][4] = 0
            data_versicolor[i][4]=1
            data_virginica[i][4]=0
        else:
            data_setosa[i][4] = 0
            data_versicolor[i][4]=0
            data_virginica[i][4]=1
    return data_setosa,data_versicolor,data_virginica

###########################
# TESTING
###########################

# initial parameters
learning_rate = 0.01
n_epoch = 10

# prepare plot
fig, (axs) = plt.subplots(nrows = 2, ncols = 3,figsize=(12,4))


training_data_all=list()
training_data_all=loading_txt("iris_training_set.txt")
test_data_all=list()
test_data_all=loading_txt("iris_test_set.txt")

#make 3xtraining set (because 3 options to choose)
training_data_setosa=list()
training_data_versicolor=list()
training_data_virginica=list()

# We have A,B,C. Make A=1 not-A=0; etc.
training_data_setosa,training_data_versicolor,training_data_virginica=take_one_vs_all_sets(training_data_all)

#delete names in last column
for row in test_data_all:
	row[4]=0
#random.shuffle(test_data_all)


matrix_setosa=list()
matrix_versicolor=list()
matrix_virginica=list()
weights_setosa=list()
weights_versicolor=list()
weights_virginica=list()

plt.show()


# load and prepare data
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)

random.shuffle(dataset)

dataset_setosa_training = prepare_subdataset(dataset, "Iris-setosa", 0.8)
dataset_versicolor_training = prepare_subdataset(dataset, "Iris-versicolor", 0.8)
dataset_virginica_training = prepare_subdataset(dataset, "Iris-virginica", 0.8)

dataset_setosa_validation = prepare_subdataset(dataset, "Iris-setosa", 0.2, True)
dataset_versicolor_validation = prepare_subdataset(dataset, "Iris-versicolor", 0.2, True)
dataset_virginica_validation = prepare_subdataset(dataset, "Iris-virginica", 0.2, True)



run_algorithm(training_data_versicolor, test_data_all, learning_rate, n_epoch, 0, "versicolor")


plt.show()