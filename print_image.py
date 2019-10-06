import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist

#케라스에서 MNIST 데이터셋 적재하기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#다섯번째 이미지 출력하기
digit = train_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


#numpy로 텐서 조작하기
#11번째에서 101번째까지 숫자를 선택해 (90, 28, 28)크기의 배열을 만듦
my_slice = train_images[10:100]
print(my_slice.shape)