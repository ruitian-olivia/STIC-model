# Mapping clinical data to categorical variables 
import numpy as np
import pandas as pd

def encode_PLT(x):
    """
    Mapping PLT to categorical variables according to whether normal or missing.
    Arguments
        x: PLT value (10^9g/L).
    Returns
        Categorical variables. 
        Normal: x>100, Abnormal: x≤100, N/A: Absence.
    """
    if float(x)<=100:
        return("abnormal")
    elif float(x)>100:
        return("normal")
    else:
        return(np.nan)

def encode_TBIL(x):
    """
    Mapping TBIL to categorical variables according to whether normal or missing.
    Arguments
        x: TBIL value (umol/L).
    Returns
        Categorical variables. 
        Normal: x<17.1, Abnormal: x≥17.1, N/A: Absence.
    """
    if float(x)>=17.1:
        return("abnormal")
    elif float(x)<17.1:
        return("normal")
    else:
        return(np.nan)

def encode_AFP(x):
    """
    Mapping AFP to categorical variables according to whether normal or missing.
    Arguments
        x: AFP value (umol/L).
    Returns
        Categorical variables. 
        Normal: x<7, Abnormal: 7≤x<400, Extreme abnormal: x≥400, N/A: Absence.
    """
    if isinstance(x,str) and x.startswith(">"):
        return("abnormal1")
    elif isinstance(x,str) and x.startswith("<"):
        return("normal")
    elif float(x) >= 400:
        return("abnormal1")
    elif float(x) >= 7 and float(x) < 400:
        return("abnormal2")
    elif float(x) < 7:
        return("normal")
    else:
        return(np.nan)

def encode_CA199(x):
    """
    Mapping CA19-9 to categorical variables according to whether normal or missing.
    Arguments
        x: CA19-9 value (U/ml).
    Returns
        Categorical variables. 
        Normal: x<39, Abnormal: x≥39, N/A: Absence.
    """
    if isinstance(x,str) and x.startswith(">"):
        return("abnormal")
    elif isinstance(x,str) and x.startswith("<"):
        return("normal")
    elif float(x) >= 39:
        return("abnormal")
    elif float(x) < 39:
        return("normal")
    else:
        return(np.nan)    

def encode_CEA(x):
    """
    Mapping CEA to categorical variables according to whether normal or missing.
    Arguments
        x: CEA value (ng/ml).
    Returns
        Categorical variables. 
        Normal: x<10, Abnormal: x≥10, N/A: Absence.
    """
    if float(x) >= 10:
        return("abnormal")
    elif float(x) < 10:
        return("normal")
    else:
        return(np.nan)

def encode_CA125(x):
    """
    Mapping CA125 to categorical variables according to whether normal or missing.
    Arguments
        x: CA125 value (U/ml).
    Returns
        Categorical variables. 
        Normal: x<35, Abnormal: x≥35, N/A: Absence.
    """
    if isinstance(x,str) and x.startswith(">"):
        return("abnormal")
    elif isinstance(x,str) and x.startswith("<"):
        return("normal")
    elif float(x) >= 35:
        return("abnormal")
    elif float(x) < 35:
        return("normal")
    else:
        return(np.nan)

def encode_HBsAg(x):
    """
    Mapping HBsAg to categorical variables according to whether positive or missing.
    Arguments
        x: string, HBsAg indicator.
    Returns
        Categorical variables. 
        Positive: (+), Negative: (-), N/A: Absence.
    """
    if "+" in str(x):
        return("positive")
    elif "-" in str(x):
        return("negative")
    else:
        return(np.nan)

def encode_age(x):
    """
    Mapping Age to five intervals.
    Arguments
        x: int, Age(year)
    Returns
        Categorical variables. 
        Interval 1: x<40, Interval 2: 40≤x<50, Interval 3: 50≤x<60,
        Interval 4: 60≤x<70 , Interval 5: x≥70.
    """
    if int(x)<40:
        return("age1")
    elif int(x)>=40 and int(x)<50:
        return("age2")
    elif int(x)>=50 and int(x)<60:
        return("age3")
    elif int(x)>=60 and int(x)<70:
        return("age4")
    elif int(x)>=70:
        return("age5") 
    else:
        return(np.nan)

clinical_data_path = "../data/clinical/raw.xlsx"
clinical_mapped_path = "../data/clinical/mapped.csv"
clinical_df = pd.read_excel(clinical_data_path, index_col=0, dtype={"testID": "str"})
clinical_df["testID"] = clinical_df["testID"].zfill(2) 
clinical_df["Age"] = clinical_df["Age"].apply(encode_age)
clinical_df["PLT"] = clinical_df["PLT"].apply(encode_PLT)
clinical_df["TBIL"] = clinical_df["TBIL"].apply(encode_TBIL)
clinical_df["AFP"] = clinical_df["AFP"].apply(encode_AFP)
clinical_df["CA199"] = clinical_df["CA199"].apply(encode_CA199)
clinical_df["CEA"] = clinical_df["CEA"].apply(encode_CEA)
clinical_df["CA125"] = clinical_df["CA125"].apply(encode_CA125)
clinical_df["HBsAg"] = clinical_df["HBsAg"].apply(encode_HBsAg)
clinical_df.to_csv(clinical_mapped_path)
