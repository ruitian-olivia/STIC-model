# Predicting scores of three types of malignant liver tumors using STIC model.
# Python 3.6, tensorflow 1.12.0, keras 2.2.4
# Usage:
# 1.single mode command line: python STIC_predicting.py single ../data test -predictionID=1
# 2.multiple model command line: python STIC_predicting.py multiple ../data test_multiple
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import logging
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

# set up log
logger = logging.getLogger('STIC_predicting')
logger.setLevel(level=logging.INFO)

# set up handler
handler = logging.FileHandler('STIC_predicting.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# add STIC model arg parser
parser = argparse.ArgumentParser(description="Arguments for STIC predicting.")

parser.add_argument(
    "mode",
    type=str,
    default="single",
    choices=["single", "multiple"],
    help="Mode of STIC model predicting, single and multiple, default is single",
)

parser.add_argument(
    "input_path",
    type=str,
    help="Folder path where the data to be predicted is stored",
)

parser.add_argument(
    "output_name", 
    type=str, 
    help="Name of file storing predicting scores(filename extension is not required)"
)

parser.add_argument(
    "-predictionID",
    type=str,
    help="Patient ID to be predicted (for single mode)",
)

args = parser.parse_args()

try:
    mode = args.mode
    in_path = args.input_path
    out_name = args.output_name
    predictionID = args.predictionID
except:
    logger.error("error in parsing args", exc_info=sys.exc_info())
    sys.exit()

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
    dummy_columns_nonull = ["Age","Gender",]
    dummy_columns_null= ["PLT","TBIL","AFP","CA199","CEA","CA125","HBsAg"]

    df_dummy_interm = pd.get_dummies(df_encode,columns = dummy_columns_nonull,drop_first=True)
    df_dummy = pd.get_dummies(df_dummy_interm,columns = dummy_columns_null)

    return df_dummy

def fill_ID(x):
    """
    Filling ID to string with specific length 2 (Add 0 to the left end of the ID string).
    Arguments
        x: ID.
    Returns
        ID string with specific length 2. 
    """
    return(str(x).zfill(2))

# single mode: predicting the score of one sample.
if mode == "single":
    try:
        predictionID = fill_ID(predictionID)
        clinical_df = pd.read_csv(os.path.join(in_path,"clinical/mapped.csv"), index_col=None, dtype={"testID": "str"})
        clinical_df = clinical_df[["testID","Age","Gender","PLT","TBIL","AFP","CA199","CEA","CA125","HBsAg"]]
        clinical_df["testID"] = clinical_df["testID"].apply(fill_ID)
        clinical_df['Age'] = pd.Categorical(clinical_df['Age'], 
            categories=["age1", "age2", "age3", "age4", "age5"], ordered=True)
        clinical_df['Gender'] = pd.Categorical(clinical_df['Gender'], 
            categories=["female", "male"], ordered=True)
        clinical_df['PLT'] = pd.Categorical(clinical_df['PLT'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['TBIL'] = pd.Categorical(clinical_df['TBIL'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['AFP'] = pd.Categorical(clinical_df['AFP'], 
            categories=["abnormal1","abnormal2", "normal"], ordered=True)
        clinical_df['CA199'] = pd.Categorical(clinical_df['CA199'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['CEA'] = pd.Categorical(clinical_df['CEA'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['CA125'] = pd.Categorical(clinical_df['CA125'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['HBsAg'] = pd.Categorical(clinical_df['HBsAg'], 
            categories=["negative", "positive"], ordered=True)
        prediction_df = clinical_df.loc[clinical_df.testID==predictionID]
        prediction_dummy_df = df_dummy_encode(prediction_df)
        clinical_data = np.array(prediction_dummy_df)[0][1:]
        clinical_data = np.expand_dims(clinical_data,0)

        img = np.asarray(Image.open(os.path.join(in_path,"CECT",predictionID,"RGB",predictionID+".png")).convert("RGB"))
        img_resize = cv2.resize(img, (224,224))
        multi_CECT = np.array(img_resize)/255.
        multi_CECT = np.expand_dims(multi_CECT,0)
    except:
        logger.error("error in loarding data of the patientID.",
                     exc_info=sys.exc_info)
    
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        STIC_model = load_model('../model/HIM_stic_model.h5')
        STIC_score = STIC_model.predict([multi_CECT,clinical_data],batch_size = 1)
        STIC_score = np.around(STIC_score, 3)

        print("Scores predicted by the STIC model:\n")
        print("HCC: {:.3f}\n".format(STIC_score[0][0]))
        print("ICC: {:.3f}\n".format(STIC_score[0][1]))
        print("Metastasis: {:.3f}\n".format(STIC_score[0][2]))

        score_df = pd.DataFrame({"predictionID":predictionID,
            "HCC_score":STIC_score[:,0],
            "ICC_score":STIC_score[:,1],
            "Meta_score":STIC_score[:,2]})
        score_df.to_csv(out_name+".csv",index = None)
    except:
        logger.error("error in STIC model predicting.", exc_info=sys.exc_info())
        sys.exit()

# multiple mode: predicting the scores of many samples in one directory.
if mode == "multiple":
    if predictionID is not None:
        logger.warning("predictionID is not needed for multiple mode")
    try:
        id_list = []
        img_list = []
        clinical_list = []

        clinical_df = pd.read_csv(os.path.join(in_path,"clinical/mapped.csv"), index_col=None, dtype={"testID": "str"})
        clinical_df = clinical_df[["testID","Age","Gender","PLT","TBIL","AFP","CA199","CEA","CA125","HBsAg"]]
        clinical_df["testID"] = clinical_df["testID"].apply(fill_ID)
        clinical_df['Age'] = pd.Categorical(clinical_df['Age'], 
            categories=["age1", "age2", "age3", "age4", "age5"], ordered=True)
        clinical_df['Gender'] = pd.Categorical(clinical_df['Gender'], 
            categories=["female", "male"], ordered=True)
        clinical_df['PLT'] = pd.Categorical(clinical_df['PLT'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['TBIL'] = pd.Categorical(clinical_df['TBIL'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['AFP'] = pd.Categorical(clinical_df['AFP'], 
            categories=["abnormal1","abnormal2", "normal"], ordered=True)
        clinical_df['CA199'] = pd.Categorical(clinical_df['CA199'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['CEA'] = pd.Categorical(clinical_df['CEA'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['CA125'] = pd.Categorical(clinical_df['CA125'], 
            categories=["abnormal", "normal"], ordered=True)
        clinical_df['HBsAg'] = pd.Categorical(clinical_df['HBsAg'], 
            categories=["negative", "positive"], ordered=True)
        clinical_dummy_df = df_dummy_encode(clinical_df)

        img_path = os.path.join(in_path,"CECT")

        for patient_id in os.listdir(img_path):
            id_list.append(patient_id)

            img = np.asarray(Image.open(os.path.join(img_path,patient_id,"RGB",patient_id+".png")).convert("RGB"))
            img_resize = cv2.resize(img, (224,224))
            img_list.append(np.array(img_resize)/255.)

            patient_df = clinical_dummy_df.loc[clinical_dummy_df.testID==patient_id]
            clinical_data = np.array(patient_df)[0][1:]
            clinical_list.append(clinical_data)

        img_array = np.array(img_list)
        clinical_array = np.array(clinical_list)
    except:
        logger.error("error in loarding data of the input_path.",
                     exc_info=sys.exc_info)
    
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        STIC_model = load_model('../model/HIM_stic_model.h5')
        STIC_score = STIC_model.predict([img_array,clinical_array],batch_size = 1)
        STIC_score = np.around(STIC_score, 3)

        score_df = pd.DataFrame({"predictionID":id_list,
            "HCC_score":STIC_score[:,0],
            "ICC_score":STIC_score[:,1],
            "Meta_score":STIC_score[:,2]})
        score_df.to_csv(out_name+".csv",index = None)
        print("The prediction of the STIC model has be done.")
        print("The predicted scores are save in {}".format(out_name+".csv"))
    except:
        logger.error("error in STIC model predicting.", exc_info=sys.exc_info())
        sys.exit()
