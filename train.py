import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenet_v2
import numpy as np
import tensorflow as tf

print(tf.__version__)
print(keras.__version__)

img_size = 224

mode = 'bbs' # [bbs, lmks]
if mode is 'bbs':
  output_size = 4
elif mode is 'lmks':
  output_size = 18

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# 에러 발생 // ValueError: Object arrays cannot be loaded when allow_pickle=False
# 먼저 기존의 np.load를 np_load_old에 저장해둠.
np_load_old = np.load
# 기존의 parameter을 바꿔줌
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

print('dataloads start!')

data_00 = np.load('dataset/CAT_00.npy')
data_01 = np.load('dataset/CAT_01.npy')
data_02 = np.load('dataset/CAT_02.npy')
data_03 = np.load('dataset/CAT_03.npy')
data_04 = np.load('dataset/CAT_04.npy')
data_05 = np.load('dataset/CAT_05.npy')
data_06 = np.load('dataset/CAT_06.npy')

print('dataloads finish!')
print('data preprocessing start!')

x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)

x_test = np.array(data_06.item().get('imgs'))
y_test = np.array(data_06.item().get(mode))

# 이미지를 0~1로 바꿔줌
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))

print('data preprocessing finish!')
print('model build start!')

# 에러 발생 // TypeError: ('Invalid keyword argument: %s', 'depth_multiplier')
# depth_multiplier을 삭제해줌

mobilenetv2_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

print('model build finish!')
print('model training start!')

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True,
  validation_data=(x_test, y_test), verbose=1,
  callbacks=[
    TensorBoard(log_dir='logs\%s' % (start_time)),
    ModelCheckpoint('models/%s.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  ]
)

print('model training finish!')