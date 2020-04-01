import random
import dlib, cv2, os
import pandas as pd
import numpy as np
from helper import resize

for i in range(7):
  print(i,'번째 전처리..')
  dirname = 'CAT_0'+str(i)
  base_path = 'cat-dataset/%s' % dirname
  file_list = sorted(os.listdir(base_path))
  random.shuffle(file_list)

  dataset = {
    'imgs': [],
    'lmks': [],
    'bbs': []
  }

  for f in file_list:
    if '.cat' not in f:
      continue

    # read landmarks
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
    landmarks = (pd_frame.as_matrix()[0][1:-1]).reshape((-1, 2))

    # load image
    img_filename, ext = os.path.splitext(f)

    img = cv2.imread(os.path.join(base_path, img_filename))

    # resize image and relocate landmarks
    img, ratio, top, left = resize.resize_img(img)
    # 변한 랜드마크를 재계산
    landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
    # 얼굴의 영역 지정
    bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

    dataset['imgs'].append(img)
    dataset['lmks'].append(landmarks.flatten())
    dataset['bbs'].append(bb.flatten())

    # for l in landmarks:
    #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    # cv2.imshow('img', img)
    # if cv2.waitKey(0) == ord('q'):
    #   break

  np.save('dataset/%s.npy' % dirname, np.array(dataset))