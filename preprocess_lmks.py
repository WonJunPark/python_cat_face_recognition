import random, sys
import dlib, cv2, os
import pandas as pd
import numpy as np
from helper import resize

for i in range(7):
    print(i, '번째 전처리..')
    dirname = 'CAT_0' + str(i)
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
        bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)]).astype(np.int)
        center = np.mean(bb, axis=0)

        # 얼굴의 크기를 구함함
        face_size = max(np.abs(np.max(landmarks, axis=0) - np.min(landmarks, axis=0)))
        # 바운딩 박스가 너무 타이트하게 잡혀서, 0.6정도의 마진을 줘서 짜름
        new_bb = np.array([
            center - face_size * 0.6,
            center + face_size * 0.6
        ]).astype(np.int)
        # -를 넘어가는 값을 없앰
        new_bb = np.clip(new_bb, 0, 99999)
        # 고양이의 얼굴부분을 추출한 부분에서 새로운 랜드마크를 설정
        new_landmarks = landmarks - new_bb[0]

        # load image
        img_filename, ext = os.path.splitext(f)

        img = cv2.imread(os.path.join(base_path, img_filename))

        new_img = img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]

        # resize image and relocate landmarks
        img, ratio, top, left = resize.resize_img(new_img)
        new_landmarks = ((new_landmarks * ratio) + np.array([left, top])).astype(np.int)

        dataset['imgs'].append(img)
        dataset['lmks'].append(new_landmarks.flatten())
        dataset['bbs'].append(new_bb.flatten())

        # for l in new_landmarks:
        #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

        # cv2.imshow('img', img)
        # if cv2.waitKey(0) == ord('q'):
        #   sys.exit(1)

    np.save('dataset/lmks_%s.npy' % dirname, np.array(dataset))