# pytorch-detection-ve-segmentasyon
Pytorch Mask RCNN pre-trained modeli ile bbox ve segmentasyon.

Veri seti: https://www.cis.upenn.edu/~jshi/ped_html/

Mask RCNN: https://arxiv.org/abs/1703.06870

NMS: https://www.youtube.com/watch?v=mlswVd_IDOE

train.py: Eğitim için.

predict.py: Eğitim sonucu kaydedilen, kod içinde network olarak adlandırdığım modeli kullanarak tahmin yapan py file. 

non_max.py: NMS

NOT: Açıklamalar kodun içinde.

# Non-Maxiumum Supression

![nms](https://user-images.githubusercontent.com/46991761/81592498-beac6c00-93c6-11ea-94ce-4bb5bb58cc38.png)

NOT: Modelden çıkan predict scorelar daha iyi sonuç veriyor. Repodaki NMS networkden bağımsız.

Takip ettiğim Pytorch tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html (Tutorialde predict.py'ın içerikleri yok.)

# Çıktılar

bbox

![image](https://user-images.githubusercontent.com/46991761/81593082-97a26a00-93c7-11ea-9a98-5ec67373f173.png)

mask

![Figure_1](https://user-images.githubusercontent.com/46991761/81593183-bdc80a00-93c7-11ea-8313-bea2628e37df.png)

Weightler: https://drive.google.com/drive/folders/1gD5s89JLXen9nEypHiciG01urIVUbSEp (network)
