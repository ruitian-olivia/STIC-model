# Merge three phase CECT images into one RGB file
# Python 3.6, matplotlib 3.3.1, pillow 7.2.0
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Directory structure before merging multi-phase CT images:
# Data folder
# └───patient1
# │   └───Dicom
# │   |   │   NC.dcm
# │   |   │   ART.dcm
# │   |   │   PV.dcm
# │   | 
# │   └───PNG
# │   |   │   NC.png
# │   |   │   ART.png
# │   |   │   PV.png
# │   | 
# │   └───Registration
# │       │   NC.png
# │       │   ART.png
# │       │   PV.png
# │   
# └───patient2
# │   ...

CT_data_path = "../data/CECT"
for patient in os.listdir(CT_data_path):
    Airlab_path = os.path.join(CT_data_path,patient,"Registration")
    rgb_path = os.path.join(CT_data_path,patient,"RGB")
    if not os.path.exists(rgb_path):
        os.mkdir(rgb_path)
    for img in os.listdir(Airlab_path):
        phase = img.split('.')[0]
        if phase == 'NC':
            nonPath=os.path.join(Airlab_path,img)
            nonGray = np.asarray(Image.open(nonPath).convert("L"))
        elif phase == 'ART':
            arteryPath=os.path.join(Airlab_path,img)
            arteryGray = np.asarray(Image.open(arteryPath).convert("L"))
        elif phase == 'PV':
            venousPath=os.path.join(Airlab_path,img)
            venousGray = np.asarray(Image.open(venousPath).convert("L"))
    multi = np.array([nonGray,arteryGray,venousGray]).swapaxes(1,0).swapaxes(2,1)
    multi_img_name = str(patient) + ".png"
    plt.imsave(os.path.join(rgb_path,multi_img_name), multi, format='png')

# Directory structure after merging multi-phase CT images:
# Data folder
# └───patient1
# │   └───Dicom
# │   |   │   NC.dcm
# │   |   │   ART.dcm
# │   |   │   PV.dcm
# │   | 
# │   └───PNG
# │   |   │   NC.png
# │   |   │   ART.png
# │   |   │   PV.png
# │   | 
# │   └───Registration
# │   |   │   NC.png
# │   |   │   ART.png
# │   |   │   PV.png
# │   | 
# │   └───RGB
# │       │   patient1.png
# │   
# └───patient2
# │   ...
