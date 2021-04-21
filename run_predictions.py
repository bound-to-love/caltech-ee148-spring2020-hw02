import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt  

templates = ['redlight.jpg', 'redlight1.jpg', 'redlight2.jpg', 'redlight3.jpg']

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    #print(np.shape(T))
    (n_rowsT, n_colsT, n_channelsT) = np.shape(T)
    heatmap = np.random.random((n_rows, n_cols))
    
    std_T = [np.std(T[i]) for i in range(0,n_channels)]
    r=0
    while r < n_rows-n_rowsT:
        c=0
        while c < n_cols-n_colsT:
            heatmap[r,c] = np.sum(I[r:r+n_rowsT, c:c+n_colsT, :]*T)
            c+=1 
        r+=1
    
    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)
    h=0
    for t in templates:
        # You may use multiple stages and combine the results
        plt.imshow(heatmap[h], cmap='hot')
        plt.show()
        T = np.asarray(Image.open(t))
        rT=np.shape(T)[0]
        cT=np.shape(T)[1]
        t=2*(rT * cT)
        maxh=np.max(heatmap[h])
        #print(maxh)
        for r in range(0,n_rows - 2*rT):
            for c in range(0, n_cols - 2*cT):
                if heatmap[h][r,c] > t: 
                    tl_row = r
                    tl_col = c
                    br_row = tl_row
                    br_col = tl_col
                    while heatmap[h][br_row,br_col] > t: 
                        if br_row < n_rows-1 and br_col < n_cols-1:    
                            br_row += 1
                            br_col += 1
                        else:
                            break
                    br_row -= 1 
                    br_col -= 1
                    if True: #tl_row - br_row >= 1 and tl_col - br_col >= 1:
                        score = heatmap[h][tl_row,tl_col]/(maxh) 
                        output.append([tl_row,tl_col,tl_row+rT+2,tl_col+cT+2, score])
        h+=1
    output = np.sort(output, axis=0).tolist()
    tmp = []
    if len(output) > 0:
        for o in range(1, len(output)):
            if output[o][0]-output[o-1][0] > rT or output[o][1]-output[o-1][1] > cT:
                tmp.append(output[o-1])
        tmp.append(output[0])
        output = tmp 
        #print(output)
        #print(type(output))
    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    mean = np.mean(I, axis=(0, 1))
    std  = np.std(I, axis=(0, 1))
    I = (I - mean) / std
    heatmaps = []
    for t in templates: 
        # You may use multiple stages and combine the results
        T = np.asarray(Image.open(t))
    
        mean = np.mean(T, axis=(0, 1))
        std  = np.std(T, axis=(0, 1))
        T = (T - mean) / std
        heatmap = compute_convolution(I, T)
        heatmaps.append(heatmap)
    #heatmap = np.mean(heatmaps, axis=0)
    #for h in range(0, len(heatmaps)):
    #     heatmaps[h] = heatmap
    output = predict_boxes(heatmaps)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../../data/RedLights2011_Medium'

# load splits: 
split_path = '../../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)
    print(file_names_train[i])
print(preds_train)
# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
