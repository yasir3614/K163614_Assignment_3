import xlrd
import numpy
from statistics import mean
import math as m

def main():

    training_data = read_excel_file_and_extract_data("missnida.xlsx")
    total_rows = training_data.nrows
    total_columns = training_data.ncols
    features = populate_feature_by_column_values(total_columns, total_rows, training_data)

    print("####### Features #######")
    print(features)
    print("\nTotal Rows: " , total_rows , "\nTotal Columns: ",total_columns)

    ####  STEP 01  ####### Scaling from range [0 - 1] ###################

    print("\n####### Step 01 :: Scaling Training Data Between [0-1] #######")

    SCALED_FEATURES = get_scaled(features)
    features_after_scaling = populate_features_after_scaling_by_row(SCALED_FEATURES, total_rows)

    print(features_after_scaling)

    ### STEP 02    ####### Prior Probabilites of Class 0 and Class 1 ####

    print("\n####### Step 02 :: Calculating Prior Probibilities of classes 0 and 1 #######")

    classes = []
    prior_probability_class_0, prior_probability_class_1 = retrieve_prior_probabilities_of_binary_classes(classes,
                                                                                                          total_columns,
                                                                                                          total_rows,
                                                                                                          training_data)
    print("\nPrior Probability of class 0 : " , prior_probability_class_0, "\nPrior Probability of class 1 : " , prior_probability_class_1)

    ### STEP 03  GROUP ROWS BY THEIR RESPECTIVE CLASS #############################
    print("\n####### Step 03 :: Grouping Rows By Respective Class #######")

    class_0_rows = [features_after_scaling[index] for index, value in enumerate(classes) if classes[index] == 0]
    class_1_rows = [features_after_scaling[index] for index, value in enumerate(classes) if classes[index] == 1.0]
    print("\nClass 0 Rows: \n",class_0_rows)
    print("\n\nClass 1 Rows: \n",class_1_rows)

    ### STEP 04 CALCULATE MEAN AND VARIANCE #########################################
    print("\n####### Step 04 :: Calculating Mean And Variance Of Testing Data Scaled #######")

    class_0_transposed = numpy.transpose(class_0_rows)
    class_1_transposed = numpy.transpose(class_1_rows)

    mean_class_0, variance_class_0 = calculate_mean_and_variance_for(class_0_transposed)
    mean_class_1, variance_class_1 = calculate_mean_and_variance_for(class_1_transposed)

    print("\n\nMean Values Of Class 0: ")
    print("\n",mean_class_0)
    print("\n\nVariance Values Of Class 0: ")
    print("\n", variance_class_0)
    print("---------------------------------------------")
    print("\n\nMean Values Of Class 1: ")
    print("\n", mean_class_1)
    print("\n\nVariance Values Of Class 1: ")
    print("\n", variance_class_1)

    ### STEP 05 Training Data Set ##############
    print("\n####### Step 05 :: Training Process Started #######")

    testing_data = read_excel_file_and_extract_data("missnidatesting.xlsx")
    total_rows_testing = testing_data.nrows
    total_columns_testing = testing_data.ncols
    features_Testing = populate_feature_by_column_values(total_columns_testing, total_rows_testing, testing_data)

    SCALED_TESTING_DATA = get_scaled(features_Testing)
    FINAL_SCALED_TESTING_DATA = numpy.transpose(SCALED_TESTING_DATA)

    print("\n####### Step 05(a) :: Scaled Training Dataset #######")

    feature_probability_class0 = get_class_feature_probability_for(FINAL_SCALED_TESTING_DATA, mean_class_0,
                                                                   variance_class_0)
    feature_probability_class1 = get_class_feature_probability_for(FINAL_SCALED_TESTING_DATA, mean_class_1,
                                                                   variance_class_1)

    print("\nCalculated Feature Probabibilies")
    print("\n\nFeature Probability of Class 0: ")
    print(feature_probability_class0)
    print("=--------------------------------------------------")
    print("\n\nFeature Probability of Class 1: ")
    print(feature_probability_class1)

    row_probabiblity_0 = extract_row_probabilities_for(feature_probability_class0, prior_probability_class_0)
    row_probabiblity_1 = extract_row_probabilities_for(feature_probability_class1, prior_probability_class_1)

    print("\nCalculated Row Probabibilies With Respect To Class 0")
    print("\n\nRow Probability Of Class 0: ")
    print(row_probabiblity_0)
    print("=--------------------------------------------------")
    print("\n\nCalculated Row Probabibilities With Respect To Class 1: ")
    print("\n\nRow Probability Of Class 1:  ")
    print(row_probabiblity_1)

    final_class = [getVal(i, j) for i, j in zip(row_probabiblity_0, row_probabiblity_1)]
    original_class = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                      0, 0, 0, 1, 1, 1]

    accuracy = get_accuracy_of(final_class, original_class)
    print("Accuracy: ", accuracy)


def get_accuracy_of(final_class, original_class):
    accuracy = 0
    for x, y in zip(final_class, original_class):
        if x == y:
            accuracy += 1
    accuracy = accuracy * 100 / len(final_class)
    return accuracy


def read_excel_file_and_extract_data(file_name):
    testing_file_path = (file_name)
    work_book_testing = xlrd.open_workbook(testing_file_path)
    testing_data = work_book_testing.sheet_by_index(0)
    return testing_data


def extract_row_probabilities_for(feature_probability_class, prior_probability_class):
    row_probabiblity = []
    for row_class in feature_probability_class:
        product_row = 1
        row_probabiblity.append(multiply_with(product_row, row_class) * prior_probability_class)
    return row_probabiblity


def multiply_with(product_row, row_class_0):
    for value_0 in row_class_0:
        product_row = product_row * value_0
    return product_row


def get_class_feature_probability_for(FINAL_SCALED_TESTING_DATA, mean_class,
                                      variance_class):
    feature_probability_class = []
    for i in range(0, len(FINAL_SCALED_TESTING_DATA)):
        feature_probability_class.append(
            [feature_probabilty(FINAL_SCALED_TESTING_DATA[i][x], variance_class[x], mean_class[x]) for x in
             range(0, len(mean_class))])
    return feature_probability_class


def calculate_mean_and_variance_for(class_row):
    mean_class = []
    variance_class = []
    for row in class_row:
        mean_class.append(float(mean(row)))
        variance_class.append(numpy.var(row))
    return mean_class, variance_class


def retrieve_prior_probabilities_of_binary_classes(classes, total_columns, total_rows, training_data):
    count_class_0 = 0
    count_class_1 = 0
    for i in range(0, total_rows):
        classes.append(training_data.cell_value(i, total_columns - 1))
    for class_value in classes:
        if class_value == 1:
            count_class_0 += 1
        else:
            count_class_1 += 1
    prior_probability_class_0 = count_class_0 / total_rows
    prior_probability_class_1 = count_class_1 / total_rows
    return prior_probability_class_0, prior_probability_class_1


def populate_features_after_scaling_by_row(SCALED_FEATURES, total_rows):
    return [[feature[i] for feature in SCALED_FEATURES] for i in range(0, total_rows)]


def populate_feature_by_column_values(total_columns, total_rows, training_data):
    features = []
    for i in range(0, total_columns - 1):
        populate_features(features, i, total_rows, training_data)
    return features


def populate_features(features, i, total_rows, training_data):
    features.append([(training_data.cell_value(j, i)) for j in range(0, total_rows)])


def getVal(i, j):
    if (i > j):
        return 0
    else:
        return 1

def feature_probabilty(xi, variance, mean_):
    e = m.e
    e_power = (-(xi - mean_) ** 2) / (2 * variance)
    answer = (e ** e_power) / m.sqrt(2 * m.pi * variance)
    return answer


def get_scaled(features):
    return [[normalize(value, max(feature), min(feature)) for value in feature] for feature in
            features]


def normalize(value, max_, min_):
    return (value - min_) / (max_ - min_)

if __name__ == "__main__":
    main()
