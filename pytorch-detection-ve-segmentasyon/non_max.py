import numpy as np



def non_max(boxes, IOUThreshold):
    boxes = np.array(boxes)
    #bbox var type flaot yap
    if boxes.dtype.kind =='i':
        boxes = boxes.astype('float')
    holder = []
    
    #bbox x ve y degerleri
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    #döndürülecek bbox nokta degerlerinin indexleri
    holder = []
    
    #bbox alanlari
    A = (x2 - x1 + 1)*(y2 - y1 + 1)
    #argsort -> arr = [7,3,5], argsort(arr) = [0,2,1]
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        
        last = len(idxs) - 1
        i  = idxs[last]
        holder.append(i)
        #kesişim korinatları
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        #kesişim genişlik ve yüksekliği
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        #w*h kesişim alanı
        IOU = (w*h) / A[idxs[:last]]#intersection over union
        #indeksleri sırayla silip döngüyü bitir
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(IOU > IOUThreshold)[0])))

    return boxes[holder].astype('int')
    
