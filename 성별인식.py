'''
12131540 송금종 - Gender classification
간단한 성별 인식 CNN 딥러닝 프로그램입니다.
(Inception-v3 모델을 이용)

220개 이미지로 테스트 했을 때 정확도는 65% 정도 나옵니다.
가상머신 상의 한계로 더 많은 테스트는 천천히 확인 해 볼 예정입니다.
'''

import tensorflow as tf
from os import listdir
from os.path import isfile, join
from sklearn.metrics import accuracy_score

# 각 클래스에서 사진을 읽고 인코딩 합니다.
maleFolder = 'testData/male'
femaleFolder = 'testData/female'

maleClassLabels = ['male' for f in listdir(maleFolder) if isfile(join(maleFolder, f))]
femaleClassLabels = ['female' for f in listdir(femaleFolder) if isfile(join(femaleFolder, f))]

malePhotos = [join(maleFolder, f) for f in listdir(maleFolder) if isfile(join(maleFolder, f))]
femalePhotos = [join(femaleFolder, f) for f in listdir(femaleFolder) if isfile(join(femaleFolder, f))]
encodedMalePhotos = [tf.gfile.FastGFile(photo, 'rb').read() for photo in malePhotos]
encodedFemalePhotos = [tf.gfile.FastGFile(photo, 'rb').read() for photo in femalePhotos]

X = encodedMalePhotos + encodedFemalePhotos
y = maleClassLabels + femaleClassLabels

# 레이블 된 파일을 읽어 들인 후 carriage return을 제거합니다.
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# 파일에서 Unpersists 그래프를 표시합니다 (pb파일 생성하는 과정)
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

# 이미지를 분석하는 과정입니다.
predictionList = []
with tf.Session() as sess:

	# 첫번째 이미지 데이터를 그래프의 입력으로 가져와서 예측을 시작합니다.
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	# 모든 이미지에 대해서 반복합니다.
	imageCounter = 0
	for image_data in X:

		# 이미지 분석 횟수를 출력합니다.
		imageCounter += 1
		print('On Image ' + str(imageCounter) + '/' + str(len(X)))

		# 예측하는 과정입니다.
		predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    
		# 첫번째 예측으로 부터 show labels을 정렬합니다.
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		# 예상 클래스를 가져와서 목록에 추가합니다. 
		prediction = label_lines[top_k[0]]
		predictionList.append(prediction)

# 정확도를 계산합니다.
accuracy = 100 * accuracy_score(y, predictionList)
print('Accuracy: \t' + str(accuracy) + '%')





