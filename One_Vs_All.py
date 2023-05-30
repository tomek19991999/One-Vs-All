from csv import reader
from math import sqrt
import random
from collections import Counter
import copy

"""
FORMAT OF FILE
1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer) 
"""

#Loading from CSV file
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

# Make a prediction with weights
def predict(row, weights):
    activation = 0 
    if(len(row)==5):
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i] # weight * column in data row
    else:
        for i in range(len(row)):
            activation += weights[i + 1] * row[i] # weight * column in data row
    activation += weights[0] #bias
    if activation >= 0.0:
        return 1.0, activation
    else:
        return -1.0, activation


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train_dataset, l_rate, n_epoch, weights):
    result_array = [[0 for i in range(1)] for j in range(len(train_dataset))] #creating list for OK or NOT_OK results. 2columns, a lot of rows [EXPECTED,PREDICTED] 
    activation_list=list()
    matrix=list()
    matrix=[[0 for i in range(2)] for j in range(2)]
    #start calculating
    for epoch in range(n_epoch): #calculate epoch times
        sum_error = 0.0
        activation_list.clear()
        matrix[0][0]=0 #tp_counter
        matrix[0][1]=0 #tp_counter
        matrix[1][0]=0 #tp_counter
        matrix[1][1]=0 #tp_counter

        for i in range(len(train_dataset)):
            prediction,activation = predict(train_dataset[i], weights)
            activation_list.append(activation)
            result_array[i]=prediction
            if (prediction ==-1):
                prediction=0

            #take data into confusion matrix
            if train_dataset[i][-1]==1 and prediction==1:
                matrix[0][0]+=1 #tp_counter
            elif train_dataset[i][-1]!=1 and prediction==1:
                matrix[0][1]+=1 #fp_counter
            elif train_dataset[i][-1]==1 and prediction!=1:
                matrix[1][0]+=1 #fn_counter
            elif train_dataset[i][-1]!=1 and prediction!=1:
                matrix[1][1]+=1 #tn_counter

            error = train_dataset[i][-1] - prediction 
            #if expected!=predicted, update weights
            if error**2 == 1:
                sum_error += error**2 
                weights[0] = weights[0] + l_rate * error # bias(t+1) = bias(t) + learning_rate * (expected(t) - predicted(t))
                for j in range(len(train_dataset[i])-1):
                    weights[j + 1] = weights[j + 1] + l_rate * error * train_dataset[i][j] #w(t+1)= w(t) + learning_rate * (expected(t) - predicted(t)) * x(t)
        if sum_error==0:
            print('\nSUM ERROR==0! \n>epoch=%d, l_rate=%.3f, sum_error=%.3f' % (epoch, l_rate, sum_error))
            print(weights)
            #print ("\nLEGEND:","\n   1  0""\n1 TP|FP\n  -----\n0 FN|TN")
            #print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1],"\n\n")
            return weights, matrix

        print('>epoch=%d, l_rate=%.3f, sum_error=%.3f' % (epoch, l_rate, sum_error))

    print(weights)
    #print ("\nLEGEND:","\n   1  0""\n1 TP|FP\n  -----\n0 FN|TN")
    #print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1],"\n\n")
    return weights, matrix


def confusion_matrix(train_dataset, l_rate, n_epoch, weights,iris_name):
    #creating list for confusion matrix
    arr = [[0 for i in range(2)] for j in range(len(train_dataset))] #creating list for OK or NOT_OK results. 2columns, a lot of rows [EXPECTED,PREDICTED]
    matrix=list()
    matrix=[[0 for i in range(2)] for j in range(2)]

    for row in train_dataset:
        prediction, activation = predict(row, weights)
        if (prediction ==-1):
            prediction=0
        #take data into confusion matrix
        if row[-1]==1 and prediction==1:
            matrix[0][0]+=1 #tp_counter
        elif row[-1]!=1 and prediction==1:
            matrix[0][1]+=1 #fp_counter
        elif row[-1]==1 and prediction!=1:
            matrix[1][0]+=1 #fn_counter
        elif row[-1]!=1 and prediction!=1:
            matrix[1][1]+=1 #tn_counter
    if (iris_name=="versicolor"):
        matrix[1][1]+=matrix[0][1]
        matrix[0][1]=0
    print ("\nLEGEND:","\n   1  0""\n1 TP|FP\n  -----\n0 FN|TN")
    print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1],"\n\n")
    #matrix[0][0] TP
    #matrix[0][1] FP
    #matrix[1][0] FN
    #matrix[1][1] TN
    
    #print("Dokladnosc ACC=", (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])) #ilość poprawnych predykcji w stosunku do ilości wszystkich badanych próbek
    #print("Precyzja PPV=", (matrix[0][0])/(matrix[0][0]+matrix[0][1]))   #ile predykcji pozytywnych faktycznie miało pozytywną wartość
    #print("Czulosc TPR=", (matrix[0][0])/(matrix[0][0]+matrix[1][0])) #ilość próbek pozytywnych została przechwycona przez pozytywne prognozy
    #print("Swoistosc SPC=", (matrix[1][1])/(matrix[0][1]+matrix[1][1]),"\n") #określający odsetek próbek true negative. TN/(FP+TN)

    return matrix

def make_matrix_versicolor(matrix):
    print ("\nMATRIX VERSICOLOR:")
    print ("\nLEGEND:","\n   1  0""\n1 TP|FP\n  -----\n0 FN|TN")
    print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1],"\n\n")
    #matrix[0][0] TP
    #matrix[0][1] FP
    #matrix[1][0] FN
    #matrix[1][1] TN
    #print("Dokladnosc ACC=", (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])) #ilość poprawnych predykcji w stosunku do ilości wszystkich badanych próbek
    print("Precyzja PPV=", (matrix[0][0])/(matrix[0][0]+matrix[0][1]))   #ile predykcji pozytywnych faktycznie miało pozytywną wartość
    print("Czulosc TPR=", (matrix[0][0])/(matrix[0][0]+matrix[1][0])) #ilość próbek pozytywnych została przechwycona przez pozytywne prognozy
    print("Swoistosc SPC=", (matrix[1][1])/(matrix[0][1]+matrix[1][1]),"\n") #określający odsetek próbek true negative. TN/(FP+TN)

def test_data_test(test_dataset, l_rate, n_epoch, weights):
    activation_list=list()
    for row in test_dataset:
        prediction, activation = predict(row, weights)  
        activation_list.append(activation) 
    return activation_list

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


def predict_test_data_by_activation_list(activation_list_setosa,activation_list_versicolor,activation_list_virginica,test_data_all):
    for i in range(len(test_data_all)):
        if max(activation_list_setosa[i], activation_list_versicolor[i], activation_list_virginica[i]) == activation_list_setosa[i]:
            test_data_all[i][4] = "Iris-setosa"
        elif max(activation_list_setosa[i], activation_list_versicolor[i], activation_list_virginica[i]) == activation_list_versicolor[i]:
            test_data_all[i][4] = "Iris-versicolor"
        else:
            test_data_all[i][4] = "Iris-virginica"

def calculate_metric(matrix_setosa,matrix_versicolor,matrix_virginica):
    matrix_all=matrix_setosa

    #take data into confusion matrix
    matrix_setosa[0][0]+=matrix_versicolor[0][0]+matrix_virginica[0][0] #tp_counter
    matrix_all[0][1]+=matrix_versicolor[0][1]+matrix_virginica[0][1] #fp_counter
    matrix_all[1][0]+=matrix_versicolor[1][0]+matrix_virginica[1][0] #fn_counter
    matrix_all[1][1]+=matrix_versicolor[1][1]+matrix_virginica[1][1] #tn_counter
    print("----------------------\nDokladnosc calkowita  ACC=", (matrix_all[0][0]+matrix_all[1][1])/(matrix_all[0][0]+matrix_all[0][1]+matrix_all[1][0]+matrix_all[1][1])) #ilość poprawnych predykcji w stosunku do ilości wszystkich badanych próbek
    print("\n\n")


def calculate_metric_all(training_data_all, y_training_data_all):
    matrix=list()
    matrix=[[0 for i in range(3)] for j in range(3)]

    for i,row in enumerate(training_data_all):

        #take data into confusion matrix
        if row[-1]=='Iris-setosa' and y_training_data_all[i]=='Iris-setosa':
            matrix[0][0]+=1 #is setosa, should be setosa
        elif row[-1]=='Iris-setosa' and y_training_data_all[i]=='Iris-versicolor':
            matrix[0][1]+=1 #is setosa, should be Versicolor
        elif row[-1]=='Iris-setosa' and y_training_data_all[i]=='Iris-virginica':
            matrix[0][2]+=1 #is setosa, should be virginica

        elif row[-1]=='Iris-versicolor' and y_training_data_all[i]=='Iris-setosa':
            matrix[1][0]+=1 #is versicolor, should be setosa
        elif row[-1]=='Iris-versicolor' and y_training_data_all[i]=='Iris-versicolor':
            matrix[1][1]+=1 #is versicolor, should be Versicolor
        elif row[-1]=='Iris-versicolor' and y_training_data_all[i]=='Iris-virginica':
            matrix[1][2]+=1 #is versicolor, should be virginica

        elif row[-1]=='Iris-virginica' and y_training_data_all[i]=='Iris-setosa':
            matrix[2][0]+=1 #is virginica, should be setosa
        elif row[-1]=='Iris-virginica' and y_training_data_all[i]=='Iris-versicolor':
            matrix[2][1]+=1 #is virginica, should be Versicolor
        elif row[-1]=='Iris-virginica' and y_training_data_all[i]=='Iris-virginica':
            matrix[2][2]+=1 #is virginica, should be virginica
    print("\n\n\n\n\n")
    print("      |   Setosa   | Versicolor | Virginica  ")
    print("---------------------------------------")
    print("Setosa  |",matrix[0][0],"      |",matrix[0][1],"       |",matrix[0][2])
    print("---------------------------------------")
    print("Versicolor |",matrix[1][0],"      |",matrix[1][1],"       |",matrix[1][2])
    print("---------------------------------------")
    print("Virginica |",matrix[2][0],"      |",matrix[2][1],"       |",matrix[2][2])
    print("\nNote: Rows represent the true labels, while columns represent the predicted labels.\n\n")

    # Calculate metrics
    flowers = ["Setosa", "Versicolor", "Virginica"]
    for i in range(3):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(3)) - tp
        fn = sum(matrix[i][j] for j in range(3)) - tp
        tn = sum(matrix[j][k] for j in range(3) for k in range(3) if j != i and k != i)

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0

        print("\nMetrics for:", flowers[i])
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1_score)
        print("Accuracy: ", accuracy)

    total_tp = sum(matrix[i][i] for i in range(3))
    total_fp = sum(sum(matrix[j][i] for j in range(3)) - matrix[i][i] for i in range(3))
    total_fn = sum(sum(matrix[i][j] for j in range(3)) - matrix[i][i] for i in range(3))
    total_tn = sum(matrix[j][k] for j in range(3) for k in range(3) if j != k and k != j)

    total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn != 0 else 0
    total_f1_score = 2 * total_precision * total_recall / (total_precision + total_recall) if total_precision + total_recall != 0 else 0

    print("\nTotal Metrics:")
    print("Precision: ", total_precision)
    print("Recall: ", total_recall)
    print("F1 Score: ", total_f1_score)
    #print("TN: ", total_tn)

    print("my_preciskion setosa: ", total_f1_score)





def one_vs_all(learning_rate, n_epoch,weights):

    training_data_all=list()
    training_data_all=loading_txt("iris_training_set.txt")
    test_data_all=list()
    test_data_all=loading_txt("iris_test_set.txt")
    #test_data_all=training_data_all.deepcopy()
    #make 3xtraining set (because 3 options to choose)
    training_data_setosa=list()
    training_data_versicolor=list()
    training_data_virginica=list()

    # We have A,B,C. Make A=1 not-A=0; etc.
    training_data_setosa,training_data_versicolor,training_data_virginica=take_one_vs_all_sets(training_data_all)

    #delete names in last column

    random.shuffle(test_data_all)


    for row in test_data_all:
        row[4]=0



    matrix_setosa=list()
    matrix_versicolor=list()
    matrix_virginica=list()

    weights_setosa=list()
    weights_versicolor=list()
    weights_virginica=list()

    test_activation_list_setosa=list()
    test_activation_list_virginica=list()
    test_activation_list_versicolor=list()

    validation_activation_list_setosa=list()
    validation_activation_list_virginica=list()
    validation_activation_list_versicolor=list()

    y_training_data_all=list()
    for row in training_data_all:
        y_training_data_all.append(row[4])


    #training by flowers
    weights_setosa, matrix_setosa = train_weights(training_data_setosa, learning_rate, n_epoch,weights)
    validation_activation_list_setosa=test_data_test(training_data_all, learning_rate, n_epoch, weights)
    test_activation_list_setosa=test_data_test(test_data_all, learning_rate, n_epoch, weights)
    weights=[0.0,0.0,0.0,0.0,0.0] #weights[0] is bias
    
    #for row in test_data_all:
    #    print (row)


    weights_virginica,matrix_virginica = train_weights(training_data_virginica, learning_rate, n_epoch,weights)
    validation_activation_list_virginica=test_data_test(training_data_all, learning_rate, n_epoch, weights)
    test_activation_list_virginica=test_data_test(test_data_all, learning_rate, n_epoch, weights)
    weights=[0.0,0.0,0.0,0.0,0.0] #weights[0] is bias


    weights_versicolor,matrix_versicolor = train_weights(training_data_versicolor, learning_rate, n_epoch,weights)
    validation_activation_list_versicolor=test_data_test(training_data_all, learning_rate, n_epoch, weights)
    test_activation_list_versicolor=test_data_test(test_data_all, learning_rate, n_epoch, weights)
    #weights=[0.0,0.0,0.0,0.0,0.0] #weights[0] is bias

    #matrix and metric calculation
    predict_test_data_by_activation_list(validation_activation_list_setosa,validation_activation_list_versicolor,validation_activation_list_virginica,training_data_all)
    calculate_metric_all(training_data_all,y_training_data_all)



    predict_test_data_by_activation_list(test_activation_list_setosa,test_activation_list_versicolor,test_activation_list_virginica,test_data_all)

    for row in test_data_all:
        print (row)


    

#seed
random.seed(100)

# test predictions
l_rate = 0.01
n_epoch = 87
weights=[0.0,0.0,0.0,0.0,0.0] #weights[0] is bias



one_vs_all(l_rate, n_epoch,weights)



#poprawic macierz - zrobic 3x3 na kolejne zajecia
