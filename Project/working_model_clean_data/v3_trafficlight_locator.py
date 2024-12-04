import cv2

# Load video
# video = cv2.VideoCapture('traffic_light_video.mp4')

# Load image
video = cv2.VideoCapture('traffic_light_image.jpg')

# Pre-trained traffic light cascade or model
traffic_light_cascade = cv2.CascadeClassifier('traffic_light_cascade.xml')

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    traffic_lights = traffic_light_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected objects
    for (x, y, w, h) in traffic_lights:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Detected Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
