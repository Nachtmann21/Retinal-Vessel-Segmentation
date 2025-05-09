###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

import configparser
import math
# imports
import sys
import time

# Keras
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
# scikit learn
from sklearn.metrics import roc_curve

sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_pre_proc
# from nn_models_v2 import get_unet

config_name = None
if len(sys.argv) == 2:
    config_name = sys.argv[1]
else:
    print("Wrong Augment!")
    exit(1)


# ========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('./' + config_name)
# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')

# original test images (for FOV selection)
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
# the border masks provided by the DRIVE
DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(DRIVE_test_border_masks)
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
# the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
# model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment + '/'
# N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
# ====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')

print("Loaded sections:", config.sections())

datasets = {'DRIVE', 'STARE', 'CHASE'}
dataset_name = config.get('data paths', 'dataset_name')
if dataset_name not in datasets:
    print("Dataset NOT support!")
    exit(1)
if dataset_name == 'DRIVE':
    width = 565
    height = 584
elif dataset_name == 'STARE':
    width = 700
    height = 605
else:
    width = 999
    height = 960
print("Dataset:", dataset_name)

# #ground truth
# gtruth= path_data + config.get('data paths', 'test_groundTruth')
# img_truth= load_hdf5(gtruth)
# visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'original')#.show()
# visualize(group_images(test_border_masks[0:20,:,:,:],5),'borders')#.show()
# visualize(group_images(img_truth[0:20,:,:,:],5),'gtruth')#.show()


# ============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if average_mode == True:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        chase_test_imgs_original= test_imgs_original,  #original
        chase_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  #masks
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        chase_test_imgs_original= test_imgs_original,  #original
        chase_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  #masks
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        patch_height = patch_height,
        patch_width = patch_width,
    )


# ================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')
# Load the saved model
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
# n_ch = patches_imgs_test.shape[1]
# print("Patches shape:", patches_imgs_test.shape)
# model = get_unet(n_ch, patch_height, patch_width)
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
start = time.time()  # start timing for inference
# Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
end = time.time()
print("Inference time (in sed): ", end-start)
# exit(0)
print("predicted images size :")
print(predictions.shape)

# ===== Convert the prediction arrays in corresponding images
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
# pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "threshold")

# ========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    orig_imgs = my_pre_proc(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])    # originals
    gtruth_masks = masks_test  # ground truth masks
else:
    N_w = math.ceil(width / patch_width)  # 15
    N_h = math.ceil(height / patch_height)  # 13
    pred_imgs = recompone(pred_patches, N_h, N_w)       # predictions
    orig_imgs = recompone(patches_imgs_test, N_h, N_w)  # originals
    gtruth_masks = recompone(patches_masks_test, N_h, N_w)  # masks
# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
kill_border(pred_imgs, test_border_masks)  # DRIVE MASK  #only for visualization
# back to original dimensions
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print("Orig imgs shape: " +str(orig_imgs.shape))
print("pred imgs shape: " +str(pred_imgs.shape))
print("Gtruth imgs shape: " +str(gtruth_masks.shape))
visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")#.show()
visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")#.show()
# visualize results comparing mask and prediction:
assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:], group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:], group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:], group)
    total_img = np.concatenate((orig_stripe, masks_stripe,pred_stripe), axis=0)
    visualize(total_img, path_experiment+name_experiment + "_Original_GroundTruth_Prediction" + str(i))#.show()


def get_best_and_worst():
    acc = np.zeros(N_predicted)
    threshold_confusion = 0.5
    for i in range(0, N_predicted):
        y_scores_sub, y_true_sub = pred_only_FOV(pred_imgs[i:i+1, :, :, :], gtruth_masks[i:i+1, :, :, :],
                                                 test_border_masks[i:i+1, :, :, :])
        y_pred_sub = np.empty((y_scores_sub.shape[0]))
        for j in range(y_scores_sub.shape[0]):
            if y_scores_sub[j] >= threshold_confusion:
                y_pred_sub[j] = 1
            else:
                y_pred_sub[j] = 0
        acc[i] = accuracy_score(y_true_sub, y_pred_sub)
    print("Acc: ", acc)
    best_i = np.argmax(acc)
    worst_i = np.argmin(acc)
    print("Best: ", best_i)
    print("Worst: ", worst_i)
    path_best = path_experiment + "best_"
    path_worst = path_experiment + "worst_"
    visualize(group_images(test_imgs_orig[best_i:best_i+1, :, :, :], 1),
              path_best + name_experiment + "_RGB" + str(best_i))
    visualize(group_images(orig_imgs[best_i:best_i+1, :, :, :], 1),
              path_best + name_experiment + "_Original" + str(best_i))
    visualize(group_images(gtruth_masks[best_i:best_i+1, :, :, :], 1),
              path_best + name_experiment + "_GroundTruth" + str(best_i))
    visualize(group_images(pred_imgs[best_i:best_i+1, :, :, :], 1),
              path_best + name_experiment + "_Prediction" + str(best_i))

    visualize(group_images(test_imgs_orig[worst_i:worst_i+1, :, :, :], 1),
              path_worst + name_experiment + "_RGB" + str(worst_i))
    visualize(group_images(orig_imgs[worst_i:worst_i+1, :, :, :], 1),
              path_worst + name_experiment + "_Original" + str(worst_i))
    visualize(group_images(gtruth_masks[worst_i:worst_i+1, :, :, :], 1),
              path_worst + name_experiment + "_GroundTruth" + str(worst_i))
    visualize(group_images(pred_imgs[worst_i:worst_i+1, :, :, :], 1),
              path_worst + name_experiment + "_Prediction" + str(worst_i))


# ====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
# get_best_and_worst()

# predictions only inside the FOV
y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV
print("Calculating results only inside the FOV:")
print("y scores pixels: " + str(y_scores.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]) +" (584*565==329960)")
print("y true pixels: " + str(y_true.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)")

# Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)

accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_score(y_true, y_pred)
print("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:\n"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()
