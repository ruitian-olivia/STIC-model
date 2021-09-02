# Multi-phase CT affine registration
# Python 3.6, matplotlib 3.3.1, torch 1.7.1, simpleitk 2.0.2, Airlab (https://github.com/airlab-unibas/airlab)
import os,shutil
import time
import matplotlib.pyplot as plt
import torch as th

import airlab as al

def register_ct(fixed_path,registered_path,result_path,iter_num,dtype,device):
    """
    It is a function to register the multi-phase CT images.
    Choose the NC phase CT as a reference.
    Register the other phase CT images using the affine registration algorithm.
    Arguments
        fixed_path: the file path of NC phase CT.
        registered_path: the file path of the other phase CT images.
        result_path: the file path saving CT images after affine registration.
        iter_num: int, the number of iterations for affine registration algorithm.
        dtype: tensor type.
        device: device for optimization of affine registration algorithm.
    """
    #load image data
    fixed_image = al.read_image_as_tensor(fixed_path,dtype=dtype,device=device)
    registered_image = al.read_image_as_tensor(registered_path,dtype=dtype,device=device)
    #normalize to [0,1]
    fixed_image, registered_image = al.utils.normalize_images(fixed_image,registered_image)
    #convert intensities
    fixed_image.image = 1 - fixed_image.image
    registered_image.image = 1 - registered_image.image
    #create pairwise registration object
    registration = al.PairwiseRegistration()
    #choose the affine transformation model
    transformation = al.transformation.pairwise.SimilarityTransformation(registered_image, opt_cm=True)
    #initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)
    registration.set_transformation(transformation)
    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image, registered_image)
    registration.set_image_loss([image_loss])
    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)
    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(iter_num)
    # start the registration
    registration.start()
    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    registered_image.image = 1 - registered_image.image
    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    result_image = al.transformation.utils.warp_image(registered_image, displacement)    
    # save the result_image to result_path
    plt.imsave(result_path, result_image.numpy(), format='png', cmap='gray')

start = time.time()
dtype = th.float32
device = th.device("cuda")
iter_num = 100

# Directory structure before CT affine registration:
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

CT_data_path = "../data/CECT"
for patient in os.listdir(CT_data_path):
    png_path = os.path.join(CT_data_path,patient,"PNG")
    Airlab_path = os.path.join(CT_data_path,patient,"Registration")
    if not os.path.exists(Airlab_path):
        os.mkdir(Airlab_path)
    for img in os.listdir(png_path):
        if img.startswith("NC"):
            fixed_path=os.path.join(png_path,img)
            copy_path=os.path.join(Airlab_path,img)
            if os.path.exists(fixed_path):
                shutil.copyfile(fixed_path,copy_path)
            else:
                print("Error! Not find non-contrast image in %s"%patient)
            break
    for img in os.listdir(png_path):
        if img.startswith("ART") or img.startswith("PV"):
            registered_path=os.path.join(png_path,img.strip())
            result_path=os.path.join(Airlab_path,img.strip())
            if os.path.exists(registered_path):
                register_ct(fixed_path,registered_path,result_path,iter_num,dtype,device)
            else:
                print("Error! Not find %s image in %s"%(img,patient))

end = time.time()
print("=================================================================")
print("Registration done in:", end - start, "s")

# Directory structure after CT affine registration:
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

