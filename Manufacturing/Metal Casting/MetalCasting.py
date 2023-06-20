import json
from PIL import Image
import requests
import streamlit as st

def resize(img, size):
    image = Image.open(img)
    image.thumbnail((size, size))
    return image

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

    st.title('Metal Casting Defect Detection')

    file = st.file_uploader('Upload an image')

    if file:
        inference = inferUpload(file)
        st.title("Here is the image you selected")
        st.write(file.name + ":\n")
        st.image(resize(file, 500))
        if(inference["predictions"]["labelName"] == "NG"):
            i = 1 #3487
            for prediction in inference["backbonepredictions"]:
                st.write("Prediction " + str(i) + ": " + inference["backbonepredictions"][prediction]["labelName"])
                i += 1
        else:  
            st.write("Prediction: " + inference["predictions"]["labelName"])

if __name__ == "__main__":
    main() 
