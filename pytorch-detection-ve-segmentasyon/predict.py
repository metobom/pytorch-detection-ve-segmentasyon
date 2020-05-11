from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import cv2
from non_max import non_max
import matplotlib.pyplot as plt

loader = transforms.Compose([transforms.ToTensor()])

#belirtilen yoldan resmi çeken foo
def img_loader(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = loader(img).float()
    img = Variable(img, requires_grad=True)
    return img.cuda()
    
path = r'test_img1.png'

#modeli yükle
model = torch.load('network')
#model eval
model.eval()
#resmi çek
img_arr = img_loader(path)
#predict
with torch.no_grad():
    pred = model([img_arr])


#bbox çizmek için resmi çek
img1 = cv2.imread(path)
#predictionlardan bboxları çek
boxes_np = np.array(pred[0]['boxes'].cpu())

iter_num = len(boxes_np)

#non max supression için 1, predict olasılıklarını kullanmak için 0
#olasılıkları kullanmak daha iyi sonuç veriyor.
non_max_supress = 0
if non_max_supress:
    non_max_threshold = 0.1
    boxes_supressed = non_max(boxes_np, non_max_threshold)
    #bbox çiz
    iter_num = len(boxes_supressed)
    for i in range(iter_num):
        tl =  (boxes_supressed[i][0], boxes_supressed[i][1])
        br = (boxes_supressed[i][2], boxes_supressed[i][3])
        img1 = cv2.rectangle(img1,tl,br, (0,0,255), 2)
    print('Non-max supression kullanıldı')
else:
    bbox_threshold = 0.9
    for i in range(iter_num):
        tl =  (boxes_np[i][0], boxes_np[i][1])
        br = (boxes_np[i][2], boxes_np[i][3])
        if pred[0]['scores'][i] > bbox_threshold:
            img1 = cv2.rectangle(img1,tl,br, (0,0,255), 2)

h = img1.shape[0]
w = img1.shape[1]
mask_acc = np.zeros([h,w])
for i in range(len(pred[0]['masks'])):
    mask_acc += np.array(pred[0]['masks'][i].cpu()).reshape(h,w)



#bbox için 0, mask için 0
bbox = 1

if bbox:
    cv2.imshow('out', img1)
    cv2.waitKey(0)
else:
    plt.imshow(mask_acc, cmap = 'gray')
    plt.show()
