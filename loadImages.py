from pathlib import Path
import numpy as np
from keras.preprocessing import image

def loadImages() -> tuple:
    print('Loading data...')
    p = Path('gaussian_filtered_images/gaussian_filtered_images/')
    dirs = p.glob('*')

    INPUT_SHAPE = (80,80,3)

    labels = []
    imgData = []
    labelsVal = { 'No_DR': 0, 'Mild': 1, 'Moderate': 1, 'Severe': 1, 'Proliferate_DR': 1 }

    for folder_dir in dirs:
        label = str(folder_dir).split('/').pop()
        
        if label.split('.').pop() == 'pkl':
            continue

        print('X[', label, ']' ' ->', labelsVal[label])
        for i, img_path in enumerate(folder_dir.glob('*.png')):
            img = image.load_img(img_path, target_size=INPUT_SHAPE)

            imgData.append(image.img_to_array(img))

            labels.append(labelsVal[label])

    X = np.array(imgData)
    Y = np.array(labels)

    return X, Y