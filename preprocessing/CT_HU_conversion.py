# CT HU value conversion 
# Python 3.6, matplotlib 3.3.1, Pydicom 2.0.0
import os
import shutil
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt

def HU_conversion(img_data, WW = 400, WL = 40):
    """
    CT HU value conversion.
    Arguments
        img_data: Two-dimensional array of CT value pixels. (-1024HU - 3071HU)
        WW: int, set Window Width as 400 to focus on the liver part in abdominal CT scans.
        WL: int, set Window Level as 40 to focus on the liver part in abdominal CT scans.
    Returns
        Two-dimensional array of CT pixels after HU conversion.(0-255 intensity)
    """
    img_temp = img_data
    min = (2 * WL - WW) / 2.0 + 0.5
    max = (2 * WL + WW) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = ((img_temp-min)*dFactor).astype(int)
    
    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp

# Directory structure for saving multi-phase CECT DICOM files:
# Data folder
# └───patient1
# │   └───Dicom
# │       │   NC.dcm
# │       │   ART.dcm
# │       │   PV.dcm
# │   
# └───patient2
# │   └───Dicom
# │       │   NC.dcm
# │       │   ART.dcm
# │       │   PV.dcm
# │   ...

CT_data_path = "../data/CECT"
for patient in os.listdir(CT_data_path):
    if os.listdir(os.path.join(CT_data_path,patient)):
        dicom_path = os.path.join(CT_data_path,patient,"Dicom")
        png_path = os.path.join(CT_data_path,patient,"PNG")
        if os.path.exists(png_path):
            shutil.rmtree(png_path)
        os.mkdir(png_path)
        for img in os.listdir(dicom_path):
            img_dicom = os.path.join(dicom_path,img.strip())
            img_ds = dicom.read_file(img_dicom)
            rescaleIntercept = np.float(img_ds.RescaleIntercept)
            rescaleSlope =  np.float(img_ds.RescaleSlope)
            img_pixel = img_ds.pixel_array
            img_ct = img_pixel*rescaleSlope + rescaleIntercept
            img_png_array = HU_conversion(img_ct,400,40)

            img_id = img.split('.')[0]+".png"
            img_png_path = os.path.join(png_path,img_id)
            plt.imsave(img_png_path,img_png_array, format='png', cmap='gray')

# Directory structure after CT HU conversion:
# Data folder
# └───patient1
# │   └───Dicom
# │   |   │   NC.dcm
# │   |   │   ART.dcm
# │   |   │   PV.dcm
# │   | 
# │   └───PNG
# │       │   NC.png
# │       │   ART.png
# │       │   PV.png
# │   
# └───patient2
# │   ...