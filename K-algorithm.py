from csv import reader
from math import sqrt
import random
from collections import Counter

#Loading from CSV file
def loading_csv (file):
    data=list()
    with open(file, 'r') as file:
        csvreader = reader(file)
        for row in csvreader:
            if not row:
                continue
            data.append(row)

    #make flaot data (before, we had string data)
    for row in data:
        for i in range(len(data[0])-1):
            row[i]=float(row[i])
            #print(row[i])
    return data


#Data visualization
def data_visualisation():
    plt.figure(figsize=(8,7))
    color = ['red', 'green', 'blue']
    labels= ['setosa', 'versicolor', 'virginica']
    for row in range(len(data)): #put data to scattler plot
        if row<50:
            plt.scatter(data[row][2],data[row][3],c=color[0], label='setosa')
        elif row<100:
            plt.scatter(data[row][2],data[row][3],c=color[1], label='versicolor')
        else:
            plt.scatter(data[row][2],data[row][3],c=color[2], label='virginica')

    plt.legend()
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Scatter plot of IRIS flowers')
    plt.show()

#converting list to values from 1 to 0
def convert_to_values_from_one_to_zero(data):
    max_list=list()
    min_list=list()
    #taking maximum and minimum values
    for i in range (len(data[0])-1):
        a=max(l[i] for l in data) #take maxiumum from column "i"
        max_list.append(a)
        a=min(l[i] for l in data)
        min_list.append(a)
    #print("max:",max_list)
    #print("min:",min_list)
    
    #rescale our values from data
    for row in data:
        for i in range(len(row)-1):
            row[i] = (row[i]-min_list[i])/(max_list[i]-min_list[i])
    #print (data[0])
    return data

#calculate Euclidean distance
def euclidean_distance(row1,row2):
    distance=0.0
    for i in range (len(row1)-1):
        distance += (row1[i] - row2[i])**2 #euclidean distance = sqrt[(a1-a2)^2 + (b1-b2)^2 + (c1-c2)^2 + (d1-d2)^2 + .....]
    return sqrt(distance)

#get list of neighbors
def get_list_of_neighbors(data,test_sample,k):
    #calculating distance from all neighbors and make list with it (150 rows, 1 column)
    distance_list=list()
    for row in range (len(data)):
        distance_list.append(euclidean_distance(data[row],test_sample)) #make list with distances
    #print (distance_list)
    new_big_distance_list = [a + [b] for a, b in zip(data, distance_list)] #merge data and distance lists, format of big list: [a,b,c,d,group_name,distance] (with all data:150samples)
    new_big_distance_list.sort(key=lambda my_list:my_list[5])
    small_distance_list=new_big_distance_list[0:k] #take k elements to new, small list 
    #print(small_distance_list)
    return small_distance_list

#choose group for our test sample
def choose_final_group_for_test_sample(nearest_neighbors):
    #making list with only names of neighbors
    list_of_groups=list()
    for row in nearest_neighbors:
        list_of_groups.append(row[4])
    #searching for most string with biggest frequency
    word_count = Counter(list_of_groups) #make object with counted groups
    most_common_word = word_count.most_common(1)[0][0] #take world with biggest frequency
    #print (most_common_word)
    return most_common_word

#take from data: 60% training samples, 20% walidation samples, 20% test samples
def take_training_data(data):
    training_data=list()
    #calculate how many samples take
    how_many_training_data=0.6*len(data) #output 90
    #print(how_many_samples_take)

    #taking data
    for i in range (int(how_many_training_data/3)):
        training_data.append(data[i])    #0-49
        training_data.append(data[i+50]) #50-99
        training_data.append(data[i+100])#100-149
    #print(len(training_data))
    #print(training_data)
    return training_data

def take_validation_data(data):
    validation_data=list()
    #calculate how many samples take
    how_many_validation_data=0.2*len(data) #output 30
    #print(how_many_samples_take)

    #taking data
    for i in range (int(how_many_validation_data/3)):
        validation_data.append(data[i+30])
        validation_data.append(data[i+80])
        validation_data.append(data[i+130])
    return validation_data

def take_test_data(data):
    test_data=list()
    #calculate how many samples take
    how_many_data=0.2*len(data) #output 30
    #print(how_many_samples_take)

    #taking data
    for i in range (int(how_many_data/3)):
        test_data.append(data[i+40])
        test_data.append(data[i+90])
        test_data.append(data[i+140])

    #delete iris names column
    for j in test_data:
        del j[4]
    #print(test_data)
    return test_data

def validation_data_test (training_data,validation_data,k):
    nearest_neighbors=list()
    assign_list=list()
    assign_list=[[0 for i in range(3)] for j in range(len(validation_data))] #make 3 columns and X rows (in this data - 20) - [[IS_ASSIGNED_OK?][REAL_GROUP][CHOSED_GROUP]];
    #print(assign_list)
    for row in range (len(validation_data)):
        nearest_neighbors=get_list_of_neighbors(training_data,validation_data[row],k)
        final_group=choose_final_group_for_test_sample(nearest_neighbors)

        #take data to assing_list [[IS_ASSIGNED_OK?][REAL_GROUP][CHOSED_GROUP]];[[][][]];...
        if final_group==validation_data[row][4]:
            assign_list[row][0] = 1
        else:
            assign_list[row][0] = 0

        assign_list[row][1]=validation_data[row][4]
        assign_list[row][2]=final_group
            #print("Poprawnie przydzielono probke nr:",row)
    #print (assign_list)
    #print(sum(row[0] for row in assign_list))
    return assign_list

def test_data_test (training_data,test_data,k):
    nearest_neighbors=list()
    #print(test_data)
    for row in range (len(test_data)):
        nearest_neighbors=get_list_of_neighbors(training_data,test_data[row],k)
        final_group=choose_final_group_for_test_sample(nearest_neighbors)
        print("final group for test sample nb.",row+1,", is:",final_group)

def true_positive_false_positive_confusion_matrix(result,iris_name):
    matrix=list()
    matrix=[[0 for i in range(2)] for j in range(2)]
    print ("\nFor: ",iris_name, "\nTP|FP\n-----\nFN|TN")
    #print ("TP|FP\n---\nFN|TN")
    for row in range(len(result)): #[[IS_ASSIGNED_OK?][REAL_GROUP][CHOSED_GROUP]];[[][][]];...
        if result[row][1]==iris_name and result[row][2]==iris_name:
            matrix[0][0]+=1 #tp_counter
        elif result[row][1]!=iris_name and result[row][2]==iris_name:
            matrix[0][1]+=1 #fp_counter
        elif result[row][1]==iris_name and result[row][2]!=iris_name:
            matrix[1][0]+=1 #fn_counter
        elif result[row][1]!=iris_name and result[row][2]!=iris_name:
            matrix[1][1]+=1 #tn_counter
    print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1])



def k_nearest_neighbors(file_name,k,auto_find_k,randomize_data):

    data=loading_csv(file_name)
    data = convert_to_values_from_one_to_zero(data)

    # If randomize bit is ON
    if randomize_data==1:
        random.shuffle(data)


    training_data=list()
    validation_data=list()
    test_data=list()

    training_data=take_training_data(data)
    validation_data=take_validation_data(data)
    test_data=take_test_data(data)
    #training_data.append(validation_data[0]) #test
    
    # If K autofind is ON (1)
    if auto_find_k==1:
        max_result=0 #max output validation data test funcion (how many samples is assigned correctly)
        result=0
        k_result=0 #best k
        
        for i in range (1,len(training_data)):
            result=validation_data_test(training_data,validation_data,i) #return number of correct validations
            if max_result<(sum(row[0] for row in result)):
                max_result=sum(row[0] for row in result)
                k_result=i
        k=k_result
        result=(validation_data_test(training_data,validation_data,k))
        true_positive_false_positive_confusion_matrix(result,"Iris-setosa")
        true_positive_false_positive_confusion_matrix(result,"Iris-versicolor")
        true_positive_false_positive_confusion_matrix(result,"Iris-virginica")
        print("\nnumber of correct validations=",sum(row[0] for row in result), ", with k=",k,"\n")

    # If K autofind is OFF (0)
    else:
        result=(validation_data_test(training_data,validation_data,k))
        true_positive_false_positive_confusion_matrix(result,"Iris-setosa")
        true_positive_false_positive_confusion_matrix(result,"Iris-versicolor")
        true_positive_false_positive_confusion_matrix(result,"Iris-virginica")
        print("\nnumbers of correct validations=",sum(row[0] for row in result), "\n")

    #test data
    result=test_data_test(training_data,test_data,k)



####initialising variables
k=5
file_name="iris.csv"
##### flags
auto_find_k=0 # AUTOFIND BEST K - 0-OFF, 1-ON
randomize_data=0 # SHUFFLE DATA - 0-OFF, 1-ON

k_nearest_neighbors(file_name,k,auto_find_k,randomize_data)


