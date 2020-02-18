# USAGE
# python3 main.py

# import the necessary python modules
from page_transform.transform import four_point_transform
from skimage.filters import threshold_local
from time import sleep
from google.api_core.exceptions import AlreadyExists
from google.cloud import vision
from google.cloud import texttospeech
from playsound import playsound
from glob import glob as glob
from data import preproc as pp, evaluation
from data.generator import DataGenerator, Tokenizer
from data.reader import Dataset
from kaldiio import WriteHelper
from network.model import HTRModel
import numpy as np
import RPi.GPIO as GPIO
import cv2
import imutils
import pytesseract
import pyttsx3
import io
import os
import html
import sys
import urllib.request
import h5py
import string
import datetime
import natsort
import imgproc
import time
import shutil

# Extract GCP project id variable to connect to Google Cloud
PROJECT_ID = os.environ.get('GCLOUD_PROJECT')
USERNAME = os.environ.get('USERNAME')

# Recognizes text in a image file using Google's Vision API
def cloud_recognize_text(infile):
    print("Recognizing Text...")

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Opens the input image file
    with io.open(infile, 'rb') as image_file:
        content = image_file.read()

    # Sets the image input to be recognized
    image = vision.types.Image(content=content)

    # Recognize text in image
    response = client.document_text_detection(image=image)
    text = response.full_text_annotation.text

    # Returns the detected text
    return text

# Converts detected text into SSML and generates synthetic audio using
# Google's Text-to-Speech API
def cloud_text_to_speech(text, outfile):
    print("Synthesizing Text...")

    # Replace special characters with HTML Ampersand Character Codes
    # These Codes prevent the API from confusing text with SSML commands
    # For example, '<' --> '&lt;' and '&' --> '&amp;'
    escaped_lines = html.escape(text)

    # Convert plaintext to SSML in order to wait 1 second
    # between each period in synthetic speech
    ssml = '<speak>{}</speak>'.format(
        escaped_lines.replace('.', '\n<break time="1s"/>'))

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Sets the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(ssml=ssml)

    # Builds the voice request, selects the language code ("en-US"),
    # name ("en-US-Wavenet-F") and the SSML voice gender ("FEMALE")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-US',
        name='en-US-Wavenet-F',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)

    # Selects the type of audio file to return (.mp3)
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    # Performs the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(synthesis_input, voice, audio_config)

    # Writes the synthetic audio to the output file.
    with open(outfile, 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file ' + outfile)

# Recognize printed and handwritten text using Google's Vision API. Synthesize
# recognized text into speech using Google's Text-to-Speech API. Speak the text.
def cloud_read():
    # Photo from which to extract text
    infile = os.path.join("/home" , USERNAME, "Reading_Eye_For_The_Blind", "images", "picture.png")
    # Name of file that will hold synthetic speech
    outfile = os.path.join("/home" , USERNAME, "Reading_Eye_For_The_Blind", "read.mp3")

    # Recognize text in the input image
    text = str(cloud_recognize_text(infile))

    # Count number of lines in recognized text
    nlines = text.count('\n')

    # Set character limit and split the text into an array. We need to split
    # the text into sections to avoid requesting synthesized voice of too many
    # characters
    n = 4200 - nlines
    count = 0
    text = [text[i:i + n] for i in range(0, len(text), n)]

    print("Recognized Text:\n\n")

    # Convert each text section into synthesized speech
    for x in text:
        print(x) # Print the recognized text section
        cloud_text_to_speech(x, outfile) # Convert text to speech
        playsound(outfile) # Play the synthesized speech
        count += 1
        outfile = str(count) + '.mp3' # Rename audio outfile and repeat

# Recognized printed text from a page using OpenCV and PyTesseract
def edge_recognize_printed_page(image):
    # load the image and compute the ratio of the old height to the new height,
    # clone it, and resize it
    print("Load image at" + image)
    image = cv2.imread(image)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # apply the four point transform to obtain a top-down view of the
    # original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255

    # Recognize printed text using PyTesseract
    print("Recognizing Text...")
    text = pytesseract.image_to_string(warped)
    print("Recognized Text:\n\n" + text)

    # Speak the recognized text using pyttsx3
    print("Speaking Text")
    engine = pyttsx3.init()
    engine.setProperty('voice', "en-us+f5")
    engine.say(text)
    engine.runAndWait()

# Recognized printed text from a book using OpenCV and PyTesseract
def edge_recognize_printed_book(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    print("Load image at" + image)
    image = cv2.imread(image)
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    T = threshold_local(orig, 11, offset = 10, method = "gaussian")
    orig = (orig > T).astype("uint8") * 255

    # Recognize printed text using PyTesseract
    print("Recognizing Text...")
    text = pytesseract.image_to_string(orig)
    print("Recognized Text:\n\n" + text)

    # Speak the recognized text using pyttsx3
    print("Speaking Text")
    engine = pyttsx3.init()
    engine.setProperty('voice', "en-us+f5")
    engine.say(text)
    engine.runAndWait()

# Split handwritten text into lines of text
def edge_split_handwritten_lines():
    print("Splitting")

    # Define source for input image and output images
    pn_SRC = os.path.join("/home" , USERNAME, "Reading_Eye_For_The_Blind","images")
    pn_OUT = os.path.join("/home" , USERNAME, "Reading_Eye_For_The_Blind", "outes")

    # Split handwritten text
    images = sorted(glob(os.path.join(pn_SRC, "*.png")))
    imgproc.compile()
    imgproc.execute(images, pn_OUT)

    print("Done Splitting")

# Recognizes handwritten text lines using the trained neural network
def edge_recognize_handwritten_text():
    print("Recognizing Text")

    # Define source for handwritten text recognition model and image
    # of text to recognize
    output_path = os.path.join("/home" , USERNAME, "Reading_Eye_For_The_Blind", "model", "iam", "flor")
    target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
    images_path = os.path.join("/home" , USERNAME, "Reading_Eye_For_The_Blind", "outes", "picture.png")

    # Define input size, number max of chars per line and list of valid chars
    # (0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'
    # ()*+,-./:;<=>?@[\]^_`{|}~)
    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

    # Define the handwritten text recognition model and load checkpoints
    model = HTRModel(architecture="flor",
                     input_size=input_size,
                     vocab_size=tokenizer.vocab_size,
                     top_paths=10)
    model.compile()
    model.load_checkpoint(target=target_path)

    # Sort each text line image in order
    images = natsort.natsorted(os.listdir(images_path))

    text = ""

    # Loop through text line images and recognize text
    for filename in images:
        print(filename)
        img = pp.preprocess(images_path + "/" + filename, input_size=input_size)
        x_test = pp.normalization([img])
        predicts, probabilities = model.predict(x_test, ctc_decode=True)
        predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

        print(predicts[0][0]) # Print the predicted text and repeat
        text += predicts[0][0]

    # Print recognized handwritten text and speak it.
    print(text)
    print("Speaking Text")
    engine = pyttsx3.init()
    engine.setProperty('voice', "en-us+f5")
    engine.say(text)
    engine.runAndWait()

    # Remove the directory of all the text line images
    try:
        shutil.rmtree('outes')
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

# Recognize printed text on the edge
def edge_print_read():
    # First, try to recognized text from a page. If no page was detected,
    # then recognize text from a book
    try:
        edge_recognize_printed_page("/home/" + USERNAME + "/Reading_Eye_For_The_Blind/images/picture.png")
    except:
        edge_recognize_printed_book("/home/" + USERNAME + "/Reading_Eye_For_The_Blind/images/picture.png")

# Recognized handwritten text on the edge
def edge_handwritten_read():
    split_lines()
    recognize_handwritten()

# Check to see if start button was pressed
def check_button():
    but_pin = 18 # Define GPIO pin for start button
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(but_pin, GPIO.IN)  # Button pin set as input
    GPIO.add_event_detect(but_pin, GPIO.FALLING) # Add event for button pressed

    print("Checking Button")

    # Continuously check if button was pressed. If pressed once, then return
    # "print" for printed text recognition. If pressed twice, then return
    # "handwritten" for handwritten text recognition.
    while True:
        # First check if button was pressed
        if GPIO.event_detected(but_pin):
            print("1st Press")
            time.sleep(1)

            # Second check if button was pressed
            if GPIO.event_detected(but_pin):
                GPIO.cleanup()  # cleanup all GPIOs
                return "handwritten"

            else:
                GPIO.cleanup()  # cleanup all GPIOs
                return "print"

# Check to see if Jetson Nano is connected to internet
def check_internet(host='http://google.com'):
    # Attempt to connect to google.com. Return true if successful connection
    # and return false if failed connection
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False

# Defines parameters for camera
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=720,
    display_height=1280,
    framerate=60,
    flip_method=3,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Take a picture using the camera
def take_picture():
    # Define parameters to correct for fisheye effect due to wide angle lens
    DIM = (720, 1280)
    K = np.array([[575.3324407171685, 0.0, 326.96106949050466],
                  [0.0, 571.9331100336262, 648.1857918423042], [0.0, 0.0, 1.0]])
    D = np.array([[0.10071644425918788], [-0.2686273800207196],
                  [0.4991388684034088], [-0.301052639458586]])

    # Take the picture
    print(gstreamer_pipeline(flip_method=3))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=3), cv2.CAP_GSTREAMER)
    ret_val, img = cap.read()

    # Correct fisheye effect and save the image
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite("/home/" + USERNAME + "/Reading_Eye_For_The_Blind/images/picture.png", undistorted_img)
    cap.release()

    print("Picture Taken at picture.png")

def main():
    while True:
        text_type = check_button() # Check the start button and define the type of text
        take_picture() # Take picture of text

        # If the Jetson Nano is connected to internet, use Google Cloud to recognize
        # text and synthesize speech
        if check_internet():
            print("Online Read")
            if text_type == "handwritten":
                print("Handwritten Text")
                cloud_read()
            elif text_type == "print":
                print("Print Text")
                cloud_read()

        # Else, recognize the correct text type and synthesize speech on the edge.
        else:
            print("Offline Read")
            if text_type == "handwritten":
                print("Handwritten Text")
                edge_handwritten_read()
            elif text_type == "print":
                print("Print Text")
                edge_print_read()

if __name__ == '__main__':
    main()