from pathlib import Path
import numpy as np
import csv
from keras.preprocessing import image

def loadFromCSV(target_shape):
    print('Loading data...')
    base_url = '../../../../../Projects/DR_data/aptos2019-blindness-detection/'

    p = Path(base_url + 'train_images/')
    dir = p.glob('*')

    INPUT_SHAPE = target_shape
    images = list()
    classes1 = list()
    csv_labels = list()
    with open(base_url + 'train.csv', newline='') as csvfile:
        trainLabels = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for label in list(trainLabels)[1:]:
            csv_labels.append(''.join(label))

    class0 = list(filter(lambda x: x[-1] == '0', csv_labels))[:200]
    class1 = list(filter(lambda x: x[-1] == '1', csv_labels))[:200]
    class2 = list(filter(lambda x: x[-1] == '2', csv_labels))[:200]
    class3 = list(filter(lambda x: x[-1] == '3', csv_labels))[:200]
    class4 = list(filter(lambda x: x[-1] == '4', csv_labels))[:200]
    classes = class0 + class1 + class2 + class3 + class4

    for path in dir:
        strPath = str(path)
        img_name = strPath[-16:strPath.index('.png')]
        y = list(filter(lambda x: img_name in x, classes))
        if len(y) == 0:
            continue
        images.append(image.img_to_array(image.load_img(str(strPath), target_size=INPUT_SHAPE)))
        classes1.append(y[-1].split(',')[1])

    return np.array(images), np.array(classes1)