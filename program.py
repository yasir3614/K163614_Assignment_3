import xlrd 
import numpy
#pip install xlrd in command line 
from statistics import mean 
import math as m


########## Function To Classify Test Data ###############3
def feature_probabilty(xi,variance,mean_):
    PI = m.pi
    e = m.e
    e_power =  (-(xi-mean_)**2) / (2*variance) 
    answer = (e ** (e_power) ) / m.sqrt(2*PI*variance) 
    return answer
########## Function to scale the features in the range [ 0 - 1 ] #############################3
def feature_scaling(features):
    print("Feature Scaling The Given List Of Column")
    total_features = len(features) -1
    scaled_features = []
    
    for feature in features:
        temp = []
        max_ = max(feature)
        min_ = min(feature)
        #print(max_)
        #print(min_)
        for i in feature:
            new_value_i = (i-min_) / (max_ - min_)
            temp.append(new_value_i)
        scaled_features.append(temp)

    return scaled_features  
    
def main():

    ########## Reading EXCEL File ###############
    training_file_path = ("missnida.xlsx")
    work_book = xlrd.open_workbook(training_file_path)
    training_data = work_book.sheet_by_index(0)

    total_rows = training_data.nrows
    total_columns = training_data.ncols 
    
    

    features = []
    for i in range(0,total_columns-1):
        feature = []
        for j in range(0,total_rows):
            feature.append(training_data.cell_value(j,i))
        features.append(feature)

   
    ####  STEP 01  ####### Scaling from range [0 - 1] ###################

    SCALED_FEATURES = feature_scaling(features)
    final_features = []
    j=0

    print("FINAL FITAE\n")
    for i in range(0,7):
        temp = []
        for feature in SCALED_FEATURES:
            temp.append(feature[j])
        j+=1
        final_features.append(temp)
    
    # print(final_features)
    # Uncomment the line below to print the scaled features
    # print(SCALED_FEATURES)
    

    ### STEP 02    ####### Prior Probabilites of Class 0 and Class 1 #### 

    classes = []

    count_class_0 = 0
    count_class_1 = 0

    for i in range(0,total_rows):
        classes.append(training_data.cell_value(i,total_columns-1))

    for class_value in classes:
        if class_value == 1:
            count_class_0 +=1
        else:
            count_class_1 +=1

    prior_probability_class_0 = count_class_0 / total_rows
    prior_probability_class_1 = count_class_1 / total_rows

    print("\n----------------- PRIOR PROBABIBILITIES ------------------\n")
    print("Class 0 : " , prior_probability_class_0)
    print("Class 1 : " , prior_probability_class_1)

    
    # print(count_class_0)
    # print(count_class_1)
    # print(total_rows)
    # print(prior_probability_class_0)
    # print(prior_probability_class_1)


    ### STEP 03  GROUP ROWS BY THEIR RESPECTIVE CLASS #############################

    class_0_rows = []
    class_1_rows = []
    i=0

    for class_value in classes:
        if class_value == 1.0:
            class_1_rows.append(final_features[i])
        else:
            class_0_rows.append(final_features[i])
        i+=1

    ### STEP 04 CALCULATE MEAN AND VARIANCE #########################################

    class_0_rows_final = numpy.transpose(class_0_rows)
    # print(class_0_rows_final)

    class_1_rows_final = numpy.transpose(class_1_rows)
    # print(class_1_rows_final)

    #Mean List
    mean_class_0 = []
    mean_class_1 = []

    #Variance List
    variance_class_0 = []
    variance_class_1 = []

    for row in class_0_rows_final:
        mean_class_0.append(float(mean(row)))
        variance_class_0.append(numpy.var(row))

    for row in class_1_rows_final:
        mean_class_1.append(float(mean(row)))
        variance_class_1.append(numpy.var(row))

    for val in mean_class_0:
        print(val)
    # print(mean_class_0)
    # print(mean_class_1)

    # print(variance_class_0)


    ### STEP 05 Training Data Set ##############

    testing_file_path = ("missnidatesting.xlsx")
    work_book_testing = xlrd.open_workbook(testing_file_path)
    testing_data = work_book_testing.sheet_by_index(0)

    total_rows_testing = testing_data.nrows
    total_columns_testing = testing_data.ncols 
    
    

    features_Testing = []
    for i in range(0,total_columns-1):
        feature = []
        for j in range(0,total_rows):
            feature.append(testing_data.cell_value(j,i))
        features_Testing.append(feature)
    # print(features_Testing)
    SCALED_TESTING_DATA = feature_scaling(features_Testing)
    FINAL_SCALED_TESTING_DATA = numpy.transpose(SCALED_TESTING_DATA)

    print(FINAL_SCALED_TESTING_DATA)
    
    print("-------------")

    fop_0 = []
    fop_1 = []

    print("LENGTH :  " , len(FINAL_SCALED_TESTING_DATA))

    for i in range(0,len(FINAL_SCALED_TESTING_DATA)):
        temp = []
        for x in range(0,len(mean_class_0)):
            xi = FINAL_SCALED_TESTING_DATA[i][x]
            mean_ = mean_class_0[x]
            variance = variance_class_0[x]
            temp.append(feature_probabilty(xi,variance,mean_))
            # print(feature_probabilty(xi,variance,mean_))
        fop_0.append(temp)
    print("FOP_0")
    print(fop_0)

    print("\n")

    for i in range(0,len(FINAL_SCALED_TESTING_DATA)):
        temp = []
        for x in range(0,len(mean_class_1)):
            xi = FINAL_SCALED_TESTING_DATA[i][x]
            mean_ = mean_class_1[x]
            variance = variance_class_1[x]
            temp.append(feature_probabilty(xi,variance,mean_))
        fop_1.append(temp)
   
    print("FOP _ 1")
    print(len(fop_1))

    row_probabiblity_0 = []
    row_probabiblity_1 = []
    product_row = 1

    for row_class_0 in fop_0:
        for value_0 in row_class_0:
            print(value_0  , " " )
            product_row = product_row * value_0
        print("\n" ,product_row*prior_probability_class_0, " \n")
        row_probabiblity_0.append(product_row*prior_probability_class_0)
        product_row = 1

    product_row = 1
    for row_class_1 in fop_1:
        for value_1 in row_class_1:
            print(value_1  , " " )
            product_row = product_row * value_1
        print("\n" ,product_row*prior_probability_class_1, " \n")
        row_probabiblity_1.append(product_row*prior_probability_class_1)
        product_row = 1

    
    print(row_probabiblity_0)
    print(row_probabiblity_1)

    # final_class = list(map(max,zip(row_probabiblity_0,row_probabiblity_1)))
    # print("Final Classification")
    final_class = []

    final_class = [getVal(i,j) for i,j in zip(row_probabiblity_0, row_probabiblity_1)]

    print(numpy.transpose(features_Testing))
    print(final_class)
    

   
def getVal(i, j):
    if (i>j):
        return 0
    else:
        return 1    
if __name__ == "__main__":
   
    main()