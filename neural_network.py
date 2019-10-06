import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist

#케라스에서 MNIST 데이터셋 적재하기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#train_images와 train_labels가 모델이 학습해야 할 훈련세트(training set)를 구성. 모델은 test_images와 test labels로 구성된 테스트 세트(test set)에서 테스트됨.
# 이미지는 numpy 배열로 인코딩 되어 있고, 레이블은 0부터 9까지의 숫자 배열이다. 이미지와 레이블은 일대일 관계.
#전체 데이터셋(dataset)에서 훈련과 테스트 용도로 나눈 것을 훈련세트(training set), 테스트 세트(test set)이라 함.)

#훈련데이터
print(train_images.shape)#배열 크기(60000, 28, 28). 28x28 크기의 정수행렬 6만개가 있는 배열
#각 행렬은 하나의 흑백 이미지고, 행렬의 각 원소는 0~255사이의 값을 가짐(uint8 타입)
print(len(train_labels))
print(train_labels)

#테스트 데이터
print(test_images.shape)
print(len(test_images))
print(test_labels)

#작업순서: 훈련 데이터 train_images와 train_labels를 네트워크에 주입
#네트워크는 이미지와 레이블을 연관시킬수 있도록 학습됨.
#마지막으로 test_images에 대한 예측을 네트워크에 요청.
#이 예측이 test_labels와 맞는지 확인.

#신경망 만들기
from keras import models
from keras import layers

#신경망의 핵심구성요소는 일종의 데이터 처리필터라고 생각할 수 있는 층(layer)이다.
#어떤 데이터가 들어가면 더 유용한 형태로 출력됨.
#층은 주어진 문제에 더 의미있는 표현(representation)을 입력된 데이터로부터 추출함.
#대부분의 딥러닝은 간단한 층을 연결하여 구성되어 있고, 점진적으로 데이터를 정제하는 형태를 띠고 있음
#딥러닝 모델은 데이터 정제필터(층)가 연속되어있는 데이터 프로세싱을 위한 '여과기'와 같음.




network = models.Sequential()
#조밀하게 연결된(완전연결, fully connected) 신경망 층인 Dense층 2개가 연속되어 있음.
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
#두번째 층은 10개의 확률점수가 들어있는 배열(모두 더하면 1)을 반환하는 softmax층임.
network.add(layers.Dense(10, activation='softmax'))
#각 점수는 현 숫자 이미지가 10개의 숫자 클래스 중 하나에 속할 확률임.

#신경망이 훈련 준비를 마치기 위해 컴파일단계에서 3가지가 더 필요
#손실함수(loss function): 훈련 데이터에서 신경망의 성능을 측정하는 방법. 네트워크가 옳은 방향으로 학습될 수 있도록 도와줌
#옵티마이저(optimizer): 입력된 데이터와 손실 함수를 기반으로 네트워크를 업데이트하는 메커니즘.
#훈련과 테스트 과정을 모니터링할 지표: 정확도(정확히 분류된 이미지의 비율)만 고려.

#컴파일단계
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#훈련 시작하기 전 데이터를 네트워크에 맞는 크기로 바꾸고 모든 값을 0와 1사이로 스케일을 조정.
#ex) 우리의 훈련 이미지는 [0, 255]사이의 값인 uint8타입의 (60000, 28, 28)크기를 가진 배열로 저장되어 있음.
#이 데이터를 0과 1사이의 값을 가지는 float32타입의 (60000, 28*28)크기인 배열로 바꿈
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#레이블 준비하기
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#이제 신경망 훈련시킬 준비가 되었다. 케라스에서 fit메서드 호출하여 훈련데이터에 모델을 학습시킴
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#테스트 세트에서도 모델이 잘 작동하는지 확인
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
#훈련세트 정확도보다는 약간 낮음(훈련정확도와 테스트정확도 사이의 차이는 과대적합(overfitting)때문)
#과대적합-머신러닝 모델이 훈련데이터보다 새로운 데이터에서 성능이 낮아지는 경향을 말함.


