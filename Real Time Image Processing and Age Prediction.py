import cv2

# Kamerayı açıp bir görüntü yakalayalım
camera = cv2.VideoCapture(0)

# Yüz tespiti için Haar Cascades dosyasını yükleyelim
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Yaş tahmin modelini yükleyin
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
age_list = [
    (0, 2),
    (4, 6),
    (8, 12),
    (15, 20),
    (20,25),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100)
]
while True:
    # Görüntüyü yakalayalım ve gri tonlamaya dönüştürelim
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit edelim
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Yüzleri dikdörtgenler ile çizelim
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Tespit edilen yüzleri yaş tahmin modeline gönderin
        face_blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(face_blob)
        age_preds = age_net.forward()

        # En yüksek olasılık ile tahmin edilen yaşı bulalım
        age = age_list[age_preds[0].argmax()]
        age_text = f'{age[0]} - {age[1]}'

        # Yaşı görüntü üzerine yazalım
        cv2.putText(frame, age_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Tespit edilen yüzleri, her bir framede gösterelim (real-time effect)
    cv2.imshow('Yuz Tespiti', frame)
    
    
     # Döngüyü kırarak uygulamayı sonlandırmak için 'q' tuşuna basabilirsiniz.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

