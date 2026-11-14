from ultralytics import YOLO
import cv2
import time
import pyautogui
import pygame

pygame.mixer.init()
pygame.mixer_music.load('cat-sound.mp3')

modelOD = YOLO('yolov8n.pt')

cam = cv2.VideoCapture(0)
last_time = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = modelOD(frame, verbose=False)
    annotated = results[0].plot(boxes=False)
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        width = [(boxes.xyxy[0][2] - boxes.xyxy[0][0]).item() for box in boxes]
        closestIndex = width.index(max(width))
        closestBox = boxes[closestIndex]
        x1, y1, x2, y2 = closestBox.xyxy[0]
        boxWidth = (x2 - x1).item()
        distance = 800 / boxWidth
        print(distance)
        
        cv2.putText(annotated, f'Distance: {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        
        if distance < 1.5 and (time.time() - last_time) > 5:
            filename = f'FOOZOOL_{int(time.time())}.jpg' 
            cv2.imwrite(filename, frame)
            pyautogui.screenshot(filename)
            print(f'FOOZOOL CAPTURED in {filename}')
            pygame.mixer_music.play()
            last_time = time.time()
            print(last_time)
            
            
    cv2.imshow('Distance', annotated)
    if cv2.waitKey(1) == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()            
             
    

