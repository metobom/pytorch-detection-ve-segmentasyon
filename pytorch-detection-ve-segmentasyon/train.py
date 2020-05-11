######GENEL BILGI#######
#Dataset: PennFudan, indirme linki: https://www.cis.upenn.edu/~jshi/ped_html/


import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

#data set classi
class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        #resimleri sortla, isimleri imgs'a at (mask icinde gecerli)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        #pathler, root: anaklasor, subklasor, res adi
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        #her resmi oku RGB yap
        img = Image.open(img_path).convert("RGB")
        #masklari oku 
        mask = Image.open(mask_path)
        #masklari np array yap
        mask = np.array(mask)
        #unique tekrar eden sayilari siler
        obj_ids = np.unique(mask)
        #masklardan 0'lari yani arkaplanlari cikariyor
        obj_ids = obj_ids[1:] #= [1,2]
        #encodelanmis maski mask'a atiyor
        masks = mask == obj_ids[:, None, None]

        #Her masktan bbox kordinatlarini boxes'a at
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            #x = [1,2,3], where(x) = 0,1,2 || x = [1,2,3], where(x<=2) = 0,1
            #mask'ta mask[i]'nin indexini buluyor
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            #OZET: masktan tl ve br cekiyor 

        #degiskenleri tensora cevir
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #ones human class = 1 background = 0
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        #(y2-y1)*(x2-x1), bbox alan
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        #model icin input data
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#MODEL FOO
def get_model_instance_segmentation(num_classes):
    #pretrained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    #bbox linear layer in sayisi
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #mask linear layer in sayisi
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    #GPU tanimla
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #class sayisi arkplan, insan
    num_classes = 2
    #ham data
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    #randperm: randperm(4) = [0,2,3,1], random permutasyon elemanlari döndürüyo
    #amac: veriyi rastgele dagitmak
    indices = torch.randperm(len(dataset)).tolist()
    #datanin son yarisini aliyor
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    #datanin ilk yarisini aliyor
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    #Dataloaderlar: memorynin daha iyi kullanilmasini sagliyorlar
    #num_workes = datayi memorye yazan cpu elemani gibi bisi
    #batch_size = paket servis xd
    #train icin
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    #test icin
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    #model
    model = get_model_instance_segmentation(num_classes)
    #modeli GPU'ya at
    model.to(device)

    #optimizer
    #weight decay: L2 regularization
    #momentum: x:= x-lr*grad + V1, local minimaya düsüsü hizlandiriyor
    #momentum: bir nevi lr arttirma gibi elle degil de mantikli bir hesaplama ile
    #ama lr arttima GIBI öyle degil xd
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.05)
    #training esnasinda lr degisme, her 3 step de lr*0.1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    #hmm ne acaba xd
    num_epochs = 1

    for epoch in range(num_epochs):
        #train
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        #lr güncelle
        lr_scheduler.step()
        #eval
        evaluate(model, data_loader_test, device=device)
    torch.save(model, 'network')
    
if __name__ == "__main__":
    main()
