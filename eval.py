import os
import cv2
import argparse
import numpy as np
from keras.models import load_model

from train import DataLoader


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data', required=True,
                    help='hand write picture to evalute the model')
    ap.add_argument('-m', '--model', required=True,
                    help='model dir')
    args = vars(ap.parse_args())
    
    loader = DataLoader()
    le = loader.load_label_encoder(os.path.join(args['model'], 'label_encoder.pkl'))
    model = load_model(os.path.join(args['model'], 'best_model.h5'))

    for l in range(10):
        img = cv2.imread(os.path.join(args['data'], '{}.png'.format(l)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = 255 - img # invert
        x = cv2.resize(img, (28, 28))
        
        X = loader.preprocess(np.array(x))
        res = model.predict(X)

        res = le.inverse_transform(res)
        print(res)
        
        cv2.imshow(str(res[0]), x)
        cv2.waitKey(0)

