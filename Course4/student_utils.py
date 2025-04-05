import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    # Clean reference data - keep first occurrence of each NDC_Code
    ndc_code_df_unique = ndc_df.drop_duplicates(subset='NDC_Code', keep='first')

    # Mapping code
    df['generic_drug_name'] = df['ndc_code'].map(ndc_code_df_unique.set_index('NDC_Code')['Non-proprietary Name'])

    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    # Select first encounter for each patient
    first_encounter_df = df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first')

    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    # Get unique patients
    unique_patients = df[patient_key].unique()

    # Split patients
    # Initially split into 60% train and 40% test
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.4, random_state=42)

    # Now split test into 50% validation and 50% test, which will result in a 60%/20%/20% split
    val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=42)

    # Split data
    train = df[df[patient_key].isin(train_patients)]
    validation = df[df[patient_key].isin(val_patients)]
    test = df[df[patient_key].isin(test_patients)]

    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(c, vocab_file_path)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)

        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = tf.feature_column.numeric_column(col, normalizer_fn=lambda x: normalize_numeric_with_zscore(x, MEAN, STD), default_value=default_value)

    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    # Convert to numpy array
    student_binary_prediction = df[col].values

    # Convert to binary
    student_binary_prediction = np.where(student_binary_prediction > 5, 1, 0)

    return student_binary_prediction
