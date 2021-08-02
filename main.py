import numpy as np
import matplotlib.pyplot as plt
import cv2

prototxt = "models/colorization_deploy_v2.prototxt"
caffe_model = "models/colorization_release_v2.caffemodel"
pts_npy = "models/pts_in_hull.npy"
test_image = 'my.jpg'

net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
pts = np.load(pts_npy)
 
layer1 = net.getLayerId("class8_ab")
layer2 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(layer1).blobs = [pts.astype("float32")]

net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]

test_image = cv2.imread(test_image)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

plt.imshow(test_image)
plt.show()

