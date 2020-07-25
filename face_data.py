import numpy as np
import cv2
face_data = []
skip = 0
file_name = raw_input("enter the name of the person: ")
dataset_path = '/home/keshavpc/Desktop/CodingBlocksML/class4/face_dataset/'
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

while True:

	ret,frame = cap.read()

	if ret == False:
		continue
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3 ,5)
	k = 1
	faces = sorted(faces, key = lambda x: x[2]*x[3], reverse = True)
	
	
	skip += 1
	for face in faces[:1]:
		x,y,w,h = face
		offset = 7

		face_section  = frame[ y :y + h + offset, x :x + h + offset]
		face_section = cv2.resize(face_section, (100,100))
		if skip % 10 == 0:
			face_data.append(face_section)
			print len(face_data)
			

		cv2.imshow(str(k), face_section)
		k += 1

		cv2.rectangle(frame, (x,y), (x+w, y+h), (128,128,100), 2)

	cv2.imshow("Video",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(dataset_path + '/' + file_name, face_data)
print "Dataset saved at: {}".format(dataset_path + '/' + file_name + '.npy') 
cv2.destroyAllWindows()