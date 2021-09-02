# Define functions for loading multi-phase CECT images and clinical data
# Python 3.6, tensorflow-gpu 1.12.0, keras 2.2.4
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical

# Directory structure for training and test data:
# Data folder
# └───ClinicalData
# │   │   clinicalTrain.csv
# │   │   clinicalTest.csv
# │   
# └───CECT
# │   └───Train
# │   |   │   patientID1.png
# │   |   │   patientID2.png
# │   |   │   ...
# │   | 
# │   └───Test
# │   |   │   patientID1.png
# │   |   │   patientID2.png
# │   |   │   ...

data_path = '' # Path for training and test data

def df_dummy_encode(df_encode):
    """
    Dummy variables encoding.
    Arguments
        df_encode: DataFrame of discrete clinical features,
                including age, gender, platelet (PLT), total bilirubin (TBIL), 
                alpha fetoprotein (AFP), carbohydrate antigen 19-9 (CA19-9), 
                carcinoembryonic antigen (CEA), carbohydrate antigen 125 (CA125), 
                and hepatitis B surface antigen (HBsAg)
    Returns
        DataFrame of clinical features after dummy variable encoding.
    """
    dummy_columns_nonull = ["Age","Gender"]
    dummy_columns_null= ["PIL","TBIL","AFP","CA199","CEA","CA125","HbsAg"]

    df_dummy_interm = pd.get_dummies(df_encode,columns = dummy_columns_nonull,drop_first=True)
    df_dummy = pd.get_dummies(df_dummy_interm,columns = dummy_columns_null)

    return df_dummy


def load_data(mode="train",type_list=["HCC","ICC","Meta"],resize=224):
    """
    Loading multi-phase CECT images and clinical data.
    Arguments
        mode: string, "train" or "test", choose to load training or test data.
        type_list: list, choose to load data of "HCC", "ICC" or "Meta".
        resize: input image pixels size.
    Returns
        X: input images resized by interlinear interpolation.
        Z: clinical features after dummy variables encoding.
        Y: labels of corresponding data
    """
    img_list = []
    clinical_list = []
    label_list = []
    if mode=="train":
        clinical_path = os.path.join(data_path,"clinicalTrain.csv")
        img_path = os.path.join(data_path,"CECT","Train")
    elif mode=="test":
        clinical_path = os.path.join(data_path,"clinicalTest.csv")
        img_path = os.path.join(data_path,"CECT","Test")
    else:
        print("mode ERROR")
    
    clinical_df = pd.read_csv(clinical_path, index_col=0)
    clinical_dummy_df = df_dummy_encode(clinical_df)
    clinical_dummy_df["PatientID"] = clinical_dummy_df["PatientID"].astype("str")

    for patient in os.listdir(img_path):
        patient_id = patient.split('.')[0]
        patient_df = clinical_dummy_df.loc[clinical_dummy_df.PatientID==patient_id]
        patient_type = np.array(patient_df)[0][2]
        if patient_type in type_list:
            img = np.asarray(Image.open(os.path.join(img_path,patient)).convert("RGB"))
            img_resize = cv2.resize(img, (resize,resize))
            img_list.append(np.array(img_resize)/255.)
            clinical_data = np.array(patient_df)[0][3:]
            clinical_list.append(clinical_data)
            label_list.append(patient_type) 

    X = np.array(img_list)
    Z = np.array(clinical_list)
    class_le = LabelEncoder()
    label_encoded = class_le.fit_transform(label_list)
    Y = to_categorical(label_encoded, len(type_list))

    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    Z = Z[s]
    Y = Y[s]

    return X, Z, Y

