import cv2 as cv
import time

cap = cv.VideoCapture('cars_count.mp4')
offset = 5  # Define the margin around the line for counting
y_lim = 550  # Define the location of the horizontal line
sub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
num_cars = 0
out=cv.VideoWriter('output.avi',cv.VideoWriter_fourcc(*'XVID'),20,(1280,720))
def find_center(x, y, w, h):
    center = (int(x + 0.5 * w), int(y + 0.5 * h))
    return center

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Frame processing
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 5)
    mask = sub.apply(blur)
    dilated = cv.dilate(mask, (5, 5))
    dilated = cv.morphologyEx(dilated, cv.MORPH_CLOSE, (5, 5))
    
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    cv.line(frame, (25, y_lim), (1200, y_lim), (255, 255, 0), 3)  # Draw the horizontal line
    
    current_detections = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if w < 80 or h < 80:
            continue
        center = find_center(x, y, w, h)
        current_detections.append(center)

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.circle(frame, center, 3, (0, 0, 255), -1)
    
    # Check if cars are crossing the line
    for c in current_detections:
        if c[1] > y_lim - offset and c[1] < y_lim + offset:
            num_cars += 1
            current_detections.remove(c)
            cv.line(frame, (25, y_lim), (1200, y_lim), (0, 255, 255), 3)  # Change the line color when a car crosses
    
    cv.putText(frame, f'{num_cars} vehicles  passed', (100, 80), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    # Display the frame
    cv.imshow('frame', frame)
    out.write(frame)
    cv.imshow('dilated', dilated)
    
    if cv.waitKey(1) == 27:  # Exit if 'Esc' is pressed
        break

cap.release()
cv.destroyAllWindows()
