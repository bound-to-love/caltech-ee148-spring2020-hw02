import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    inter = [0,0,0,0]
    if box_1[0] >= box_2[0]:
        inter[0] = box_2[0]
    else:
        inter[0] = box_1[0]
    if box_1[1] >= box_2[1]:
        inter[1] = box_2[1]
    else:
        inter[1] = box_1[1]
    if box_1[2] <= box_2[2]:
        inter[2] = box_1[2]
    else:
        inter[2] = box_2[2]
    if box_1[3] <= box_2[3]:
        inter[3] = box_1[3]
    else:
        inter[3] = box_2[3]

    if inter[3] > inter[1] and inter[2] > inter[0]:
        iou = (inter[3] - inter[1])*(inter[2] - inter[0])/((box_2[3] - box_2[1])*(box_2[2] - box_2[0])+(box_1[3] - box_1[1])*(box_1[2] - box_1[0]))
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if iou > iou_thr and conf_thr < pred[j][4]:
                    TP += 1
                elif iou > iou_thr and conf_thr > pred[j][4]:
                    FN += 1
                elif conf_thr < pred[j][4]:    
                    FP += 1

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../../data/hw02_preds'
gts_path = '../../data/hw02_annotations'

# load splits:
split_path = '../../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 
iou=[0.25,0.5,0.75]

confidence_thrs = np.sort(np.array([box[4] for fname in preds_train for box in preds_train[fname]],dtype=float)) # using (ascending) list of confidence scores as thresholds
tp_train, fp_train, fn_train = [], [], []
for c in range(0,3):
    tp_train.append(np.zeros(len(confidence_thrs)))
    fp_train.append(np.zeros(len(confidence_thrs)))
    fn_train.append(np.zeros(len(confidence_thrs)))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[c][i], fp_train[c][i], fn_train[c][i] = compute_counts(preds_train, gts_train, iou_thr=iou[c], conf_thr=conf_thr)

# Plot training set PR curves
recall_train = []
precision_train = []
for i in range(0, 3):
    recall_train.append(np.divide(tp_train[i],np.add(tp_train[i],fn_train[i])))
    precision_train.append(np.divide(tp_train[i],np.add(tp_train[i],fp_train[i])))
colors=['r','g','b']
for i in range(0,3):
    print(np.shape(recall_train[i]))
    print(np.shape(precision_train[i]))
    plt.plot(recall_train[i], precision_train[i], colors[i], label=iou[i])
plt.legend(loc='best')
plt.show()


if done_tweaking:
    print('Code for plotting test set PR curves.')
    confidence_thrs = np.sort(np.array([box[4] for fname in preds_test for box in preds_test[fname]],dtype=float)) # using (ascending) list of confidence scores as thresholds
    tp_test, fp_test, fn_test = [], [], []
    for c in range(0,3):
        tp_test.append(np.zeros(len(confidence_thrs)))
        fp_test.append(np.zeros(len(confidence_thrs)))
        fn_test.append(np.zeros(len(confidence_thrs)))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[c][i], fp_test[c][i], fn_test[c][i] = compute_counts(preds_test, gts_test, iou_thr=iou[c], conf_thr=conf_thr)

    # Plot testing set PR curves
    recall_test = []
    precision_test = []
    for i in range(0, 3):
        recall_test.append(np.divide(tp_test[i],np.add(tp_test[i],fn_test[i])))
        precision_test.append(np.divide(tp_test[i],np.add(tp_test[i],fp_test[i])))
    colors=['r','g','b']
    for i in range(0,3):
        print(np.shape(recall_test[i]))
        print(np.shape(precision_test[i]))
        plt.plot(recall_test[i], precision_test[i], colors[i], label=iou[i])
    plt.legend(loc='best')
    plt.show()
