import cv2
import numpy as np
from deepface import DeepFace
import streamlit as st
from PIL import Image

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default_r.xml')



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

def main():
    st.title("Face Age and Gender Detection")
    no_male = 0
    no_female = 0
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) < 2:
            st.warning("The image should contain at least 2 people.")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x-2, y-2), (x+w+2, y+h+1), (255, 0, 0), 2)
                rect_top_left = (x+10, y + h + 30)
                rect_bottom_right = (x + w - 10, y + h + 130)
                black_detected = detect_black_color(image, rect_top_left, rect_bottom_right)
                white_detected = detect_white_color(image, rect_top_left, rect_bottom_right)

                # if white_detected != 1 and black_detected != 1:
                roi = image[y+3:y+h+1, x-2:x+w+1]
                result = DeepFace.analyze(roi, actions=('age', 'gender'), enforce_detection=False)
                age = result[0]['age']
                gender = result[0]['dominant_gender']
                age_label = "Age: {:.1f}".format(age)
                gender_label = "Gender: {}".format(gender)
                if gender == 'Man':
                    no_male+=1
                elif gender == 'Woman':
                    no_female+=1

                label = age_label + " b " + gender_label

                if white_detected == 1:
                    label = "Age: {:.1f}".format(23)
                elif black_detected == 1:
                    label = "Child"
                
                
                text_x = x 
                text_y = y+h+50  

                if text_y < 10:
                    text_x = x 
                    text_y = y 

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                font_color = (0, 0, 0) 

                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
               
                background_height = text_size[1] + 20 

               
                
                background_x1 = text_x 
                background_y1 = text_y - text_size[1] - 10 
                background_x2 = background_x1 + 110
                background_y2 = background_y1 + background_height

                if('b' in label):
                   
                    label_age = label.split()[0] + label.split()[1]
                    label_gender = label.split()[3] + label.split()[4]
                    
            
                    

                text_x = background_x1 + 10
                text_y = background_y1 + 30
                cv2.rectangle(image, (0, 0), (300, 50), (255, 255, 255), cv2.FILLED)
                no_male_txt =  "Number of Males: {:.1f}".format(no_male)
                no_female_txt =  "Number of Females: {:.1f}".format(no_female)
                cv2.putText(image,no_male_txt,(5,15),font,font_scale,font_color,font_thickness)
                cv2.putText(image,no_female_txt,(5,35),font,font_scale,font_color,font_thickness)
                if('b' in label):
                    cv2.rectangle(image, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), cv2.FILLED)
                    cv2.putText(image,label_age,(text_x,text_y),font,font_scale,font_color,font_thickness)
                    cv2.rectangle(image, (background_x1, background_y1+40), (background_x2+30, background_y2+30), (255, 255, 255), cv2.FILLED)
                    cv2.putText(image,label_gender,(text_x,text_y+30),font,font_scale,font_color,font_thickness)
                else:
                    cv2.rectangle(image, (background_x1, background_y1), (background_x2, background_y2), (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, font_thickness)
                    
           
            st.image(image, channels="BGR", caption="Detected Faces with Age and Gender")

if __name__ == "__main__":
    main()
