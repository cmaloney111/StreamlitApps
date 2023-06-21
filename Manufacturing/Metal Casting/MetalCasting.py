import json
from PIL import Image
import requests
import streamlit as st
import cv2
import numpy as np
    
def inferUpload(UploadedFile):
    """
    Run inference on an image
    :param filename: path to the image file
    :return: dictionary object representing the inference results
    """
    headers = {
        'apikey': 'hsfjfnpeq1v1ayrgncycfj764nwtzd6',
        'apisecret': 'jkvn2n5i1jstkzvccu63j1l682d5olv3g6207rbus5i5e2zg70nktmyst2jmex',
    }

    params = {
        'endpoint_id':  '2bcef778-6572-4a29-882b-6e61c32d9c2b',
    }

    files = {
        'file': UploadedFile.getvalue()
    }

    response = requests.post('https://predict.app.landing.ai/inference/v1/predict', params=params, headers=headers, files=files)
    if(response.status_code == "429"):
        st.write("Too many requests to server (HTTP Status Code 429)")
    print(response.status_code)
    try:
        return json.loads(response.text)
    except json.decoder.JSONDecodeError:
        inferUpload(UploadedFile)

def main():
    st.set_page_config(layout="wide")
    st.title('Metal Casting Defect Detection')

    file = st.file_uploader('Upload an image')
   
    with st.sidebar:
        boxes = st.checkbox("Show bounding boxes")

    with st.sidebar:
        labels = st.checkbox("Show numbered labels")

    if file:
        inference = inferUpload(file)
        st.title("Here is the image you selected")
        


           
        img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
        num_bounding_box = 0
        output_file_name = "Images/output_images/output_" + file.name
        i = 1
        cv2.imwrite(output_file_name, img)
        for pred in inference["backbonepredictions"]: # Loop through each prediction
            
            result = img.copy()
            # Define dimensions for bounding box
            x1 = inference["backbonepredictions"][pred]["coordinates"]["xmin"]
            y1 = inference["backbonepredictions"][pred]["coordinates"]["ymin"]
            x2 = inference["backbonepredictions"][pred]["coordinates"]["xmax"]
            y2 = inference["backbonepredictions"][pred]["coordinates"]["ymax"]

            # Output bounding box
            x_avg = (x1 + x2) // 2
            if labels:
                if(y1 > 40):
                    result = cv2.putText(img, str(num_bounding_box + 1), (x_avg - 15, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
                else:
                    result = cv2.putText(img, str(num_bounding_box + 1), (x_avg - 15, y2  + 35), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
            if boxes:
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(output_file_name, result)
            img = cv2.imread(output_file_name)
            num_bounding_box += 1

    
        st.image(Image.open(output_file_name), width=582)  
        st.write("Number of defects: " + str(num_bounding_box))
        
        if(inference["predictions"]["labelName"] == "NG"):
            i = 1
            for prediction in inference["backbonepredictions"]:
                st.write("Defect " + str(i) + ": " + inference["backbonepredictions"][prediction]["labelName"])
                i += 1
        else:  
            st.write("Defect: " + inference["predictions"]["labelName"])



if __name__ == "__main__":
    main() 
