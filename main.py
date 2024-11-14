import cv2
from ultralytics import YOLO
import math
import paho.mqtt.client as mqtt

host = "broker.emqx.io"
port = 1883

model = YOLO("best.pt")
classNames = ["healthy","nitrogendefiency","phosphorusdefiency","potassiumdefiency"]

cap = cv2.VideoCapture(1)
cap.set(3,1920)
cap.set(4, 1080)

def on_connect(self, client, userdata, rc):
    print("MQTT Connected.")
    self.subscribe("TEST/MQTT")

def on_message(client, userdata,msg):
    print(msg.payload.decode("utf-8", "strict"))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(host)
# client.publish("TEST/MQTT","HELLO MQTT")


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            if classNames[cls]=="nitrogendefiency":
                client.publish("AI/result","NitrogenDefiency")
            elif classNames[cls]=="potassiumdefiency":
                client.publish("AI/result","PotassiumDefiency")
            elif classNames[cls]=="phosphorusdefiency":
                client.publish("AI/result","PhosphorusDefiency")



    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.loop_forever()