
import cv2
import numpy as np
from deepface import DeepFace
import copy

src = cv2.imread('images/im_1.jpg')
img = cv2.imread('images/im_1.jpg')

def get_rgb_values(image, top_left, bottom_right):
   
    x1, y1 = top_left
    x2, y2 = bottom_right
    roi = image[y1:y2, x1:x2]
    
   
    rgb_values = []
    
   
    for y in range(roi.shape[0]):
        for x in range(roi.shape[1]):
            pixel = roi[y, x]
            rgb_values.append(pixel)
    
    return rgb_values


gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)



    


def detect_black_color(org_img,top_left,bootom_right):
    image = org_img

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    alpha = 0.63
    brightened_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=0)

    threshold_value = 30
    _, black_mask = cv2.threshold(brightened_image, threshold_value, 255, cv2.THRESH_BINARY)

  
    black_threshold = 0.25 

    x = top_left[0]
    y = top_left[1]
    w = bootom_right[0] - top_left[0]
    h = bootom_right[1] - top_left[1]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    region_mean = np.mean(black_mask[y:y+h, x:x+w] / 255.0)
    
    if region_mean < black_threshold:
       return True
    else:
       return False


def detect_white_color(org_img, top_left, bottom_right):
    image = org_img.copy()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    alpha = 1.8
    brightened_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=0)

    threshold_value = 230 
    _, white_mask = cv2.threshold(brightened_image, threshold_value, 255, cv2.THRESH_BINARY_INV)  # Using THRESH_BINARY_INV to detect white regions

    white_threshold = 0.25  # Adjust as needed

    x = top_left[0]
    y = top_left[1]
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    region_mean = np.mean(white_mask[y:y+h, x:x+w] / 255.0)
    region_mean = 1- region_mean
   
    if region_mean > white_threshold:
        return True
    else:
        return False
        





faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default_r.xml')


faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))


if len(faces) < 2:
    raise ValueError("The image should contain at least 2 people.")

black_clth = []
white_clth = []
faces_codinate = []
age_gen_roi = []

for (x, y, w, h) in faces:
    cv2.rectangle(src, (x, y), (x+w, y+h), (255, 0, 0), 2)
    c = []

    
    
    rect_top_left = (x+10, y + h + 30)
    rect_bottom_right = (x + w - 10, y + h + 130)
    c.append(rect_top_left)
    c.append(rect_bottom_right)
    faces_codinate.append(c)


    rect_top_left_ag = (x - 20, y - 30)
    rect_bottom_right_ag = (x + w + 20, y + h + 30)
    age_gen_roi.append((rect_top_left_ag, rect_bottom_right_ag))

    d_img =copy.deepcopy(src)
    black_detected = detect_black_color(d_img, rect_top_left, rect_bottom_right)
    white_detected = detect_white_color(d_img, rect_top_left, rect_bottom_right)
    black_clth.append(black_detected)
    white_clth.append(white_detected)
    region_rgb_values = get_rgb_values(src, rect_top_left, rect_bottom_right)

    
    total_pixels = len(region_rgb_values)
    average_rgb = np.sum(region_rgb_values, axis=0) / total_pixels

    
    print("Average RGB value of the region:", average_rgb)




cv2.imshow('Detected Faces', src)



if len(faces_codinate) > 1:
    for i, roi_coords in enumerate(age_gen_roi):
        rect_top_left_ag, rect_bottom_right_ag = roi_coords
        roi = img[max(rect_top_left_ag[1], 0):min(rect_bottom_right_ag[1], img.shape[0]),
                max(rect_top_left_ag[0], 0):min(rect_bottom_right_ag[0], img.shape[1])]
        
        if white_clth[i] != 1 and black_clth[i] != 1:
            result = DeepFace.analyze(roi, actions=('age', 'gender'), enforce_detection=False)
            age = result[0]['age']
            gender = result[0]['dominant_gender']

        
        if white_clth[i] == 1:
            text = "Age: {:.1f}".format(23)
        elif black_clth[i] == 1:
            text = "Child"
        else:
            text = "Age: {:.1f}, Gender: {}".format(age, gender)

        
        h = faces_codinate[i][1][1] - faces_codinate[i][0][1]
       
        text_x = faces_codinate[i][0][0]  
        text_y = faces_codinate[i][0][1]  - (h-5) 

        if text_y < 10:
            text_x = faces_codinate[i][0][0]  
            text_y = faces_codinate[i][0][1] 

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (0, 0, 0) 

         
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

       
        background_width = text_size[0] + 20  
        background_height = text_size[1] + 20  

        background_x1 = text_x - 10 
        background_y1 = text_y - text_size[1] - 10  
        background_x2 = background_x1 + background_width
        background_y2 = background_y1 + background_height

       
        cv2.rectangle(src, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), cv2.FILLED)

       
        text_offset_x = background_x1 + 10  
        text_offset_y = background_y1 + text_size[1] + 5 

       
        cv2.putText(src, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    cv2.imshow('Detected Faces with Age and Gender', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("The image should contain at least 2 people.")
