import cv2
import tensorflow as tf

# 학습된 모델 불러오기
model = tf.keras.models.load_model('C:/Users/admin/Desktop/wound_burn_b20_s300.h5')

# 클래스 이름 정의하기
class_names = ['burn', 'wound']

# 카메라 열기
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽어오기
    ret, frame = cap.read()

    # 이미지 전처리하기
    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.reshape(-1, 128, 128, 3)

    # 예측하기
    prediction = model.predict(img)

    # 결과 출력하기
    if prediction < 0.5:
        class_name = class_names[0]
    if prediction >= 0.5:
        class_name = class_names[1]

    cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 화면에 출력하기
    cv2.imshow('Object detection', frame)

    # q를 누르면 종료하기
    if cv2.waitKey(1) == ord('q'):
        break

# 종료하기
cap.release()
cv2.destroyAllWindows()