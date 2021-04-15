import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw

data_path = '../../data/RedLights2011_Medium'
preds_path = '../../data/hw02_preds'  

with open(os.path.join(preds_path,'preds_train.json')) as f:
    print(os.path.join(preds_path,'preds_train.json'))
    pred = json.load(f)
    for pic in pred.keys():
        im = Image.open(os.path.join(data_path,pic))
        draw = ImageDraw.Draw(im)
        for box in pred[pic]:
            xy=[box[1],box[0],box[3],box[2]]
            draw.rectangle(xy, fill=None, outline=255)
        del draw
        im.save(os.path.join(preds_path,pic), 'JPEG')
f.close()
