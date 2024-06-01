import cv2 
import cvlib as cv
from cvlib.object_detection import draw_bbox

vid = cv2.VideoCapture(1) 

roi_x, roi_y, roi_w, roi_h = 320, 240, 5, 5 #The position and shape of our Region Of Interest(ROI)

while True:
    ret, frame = vid.read()
    bbox, label, conf = cv.detect_common_objects(frame,.1, model='yolov3-tiny')
    #label is the list that contains the name of detected objects
    
    all_objects = [] #A list of detected objects
    
    for i in range(len(bbox)):
        x,y,w,h = bbox[i]
        
        if(x < roi_x and y < roi_y and roi_x < (x + w) and roi_y < (y + h)):
            state = "ROI"
        else:
            state = "Default"
            #If the detected object's coordinate fall under our region of interest than specify the state
            
        detected_objects = {"label":label[i],"conf":conf[i], "box":(x,y,w,h), "state":state}
        #This is a dictionary that stores the relevant info of the detected object
        all_objects.append(detected_objects)
    
    for obj in all_objects:
        x,y,w,h = obj["box"]
        color = (255,0,0) if obj["state"] == "ROI" else (0,255,0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    #Draw ROI box
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    
    #Show the frame
    cv2.imshow("Image",frame)
    
    for obj in all_objects:
        print("%s detected with conf level %f and state %s" %(obj["label"],obj["conf"],obj["state"]))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()