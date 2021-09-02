# Test STIC model for classifing HCC, ICC and metastasis.
# Python 3.6, tensorflow-gpu 1.12.0, keras 2.2.4
import os
import time
from tensorflow.keras.models import load_model
from sklearn import metrics

from CNN_RNN_data import load_data

model_name = 'HIM_STIC'
type_list=["HCC","ICC","Meta"]
resize=224

# Loading the test data
X_test,Z_test,Y_test, = load_data(mode="test",type_list=type_list,resize=resize)

# Loading the trained STIC model
model = load_model(os.path.join("../model", model_name+'.h5'))

# Make predictions for samples in the test set, and record the prediction time for one sample
start = time.time()
test_score = model.predict([X_test,Z_test],batch_size = 1)
end = time.time()
per_sample_time = (end-start)/X_test.shape[0]
print("Time cost for per sample:%f \n" %per_sample_time)

# Calculate the accuracy and confusion matrix on the test set
acc = metrics.accuracy_score(Y_test.argmax(axis=1), test_score.argmax(axis=1))
print("Accuracy of test dataset:%f \n" %acc)
confusion = metrics.confusion_matrix(Y_test.argmax(axis=1), test_score.argmax(axis=1))
print("Confusion Matrix:\n")
print(confusion)