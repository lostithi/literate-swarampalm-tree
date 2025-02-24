import cv2
import pickle
import tkinter
import torch    #pytorch-DL framework for Neural networks
import pyttsx3  #text to speech
import numpy as np  #numpy library for multidimensional array and mathematical operations on arrays.
import pandas as pd    #data manipulation and analysis library-Provides datastructues  and functions for efficient handling and analyzed structured data,like data frames
import mediapipe as mp  #framework by google for multimedia tasks
from PIL import Image, ImageTk  #For opening ,manipulating anf saving many different file formats
from typing import Tuple, Union, List   #These type hints are used to specify the types of variables and return values in function and method definitions
# from gloss_proc import 

  #Imports the proc_landmarks function, Landmarks class, and GlossProcess class.
from gloss_proc.utils import draw_landmarks #imports the draw_landmarks function from a sub-module called utils inside the gloss_proc module.

mp_holistic = mp.solutions.holistic #Includes pre-built models for holistic understanding of human pose, hand tracking, and face detection
mp_drawing = mp.solutions.drawing_utils #provides utility functions for drawing landmarks, connections, and annotations on images using MediaPipe.
mp_drawing_styles = mp.solutions.drawing_styles #provides predefined drawing styles for MediaPipe annotations.


class VidProcess:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  #OPENING VIDEO CAPTURE
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   #stores the width and height of captured frames

    def get_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:    #retrieves a single frame from video capture
        success, image = False, None    #returns 1-Boolean(success-and an image as Numpy array/none-if no frames were detected)
        if self.cap.isOpened(): #Checks if video capture is open
            success, image = self.cap.read()    #Attempts to read a frame
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #converting frame color space from BGR to RGB
        return success, image

    def draw_lm(self, image: np.ndarray, res: Landmarks) -> np.ndarray: #Takes input image as a Numpy array and landmark object,returns modified image with landmarks drawn on it
        image.flags.writeable = True    #To Mutable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color space to BGR
        return draw_landmarks(image, res)   #Converted image and landmarks to draw_landmark()fn to draw landmarks on image

    def get_lm(self, image: np.ndarray) -> Union[Landmarks, None]: #input image to Numpy array 
        image.flags.writeable = False   #input image made immutable for processing for mediapipe library
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,   #placing threshold
            min_tracking_confidence=0.5,
        ) as holistic:
            try:
                return holistic.process(image)  #atttempt for processing  image using mediapipe holistic model and if successfull,landmarks are tracked in the image which returns a "Landmark" object
            except:
                return None #no landmarks obtained

    def __del__(self):      #Video capture prperly closed and released
        if self.cap.isOpened():
            self.cap.release()


class ShitApp:
    def __init__(self, window, title, max_seq_len: int = 24, model: str = "shit_lstm.pt"):    #initialize various attributes and setup user interface 
        self.window = window    #main application window
        self.tts = pyttsx3.init()   #creates instance of engine,to convert txt to speech
        self.tts.setProperty('rate', 125)   #Rate of speed to text
        self.vid_proc = VidProcess()    #Methods and functionalities of video processing
        self.window.title(title)    #Window title 
        self.max_seq_len = max_seq_len  #Maximim length of sequence
        self.image = ImageTk.PhotoImage(file="shit.png")  #Display image
        self.seq: List[np.ndarray] = [] #initialize empty list of numpy arrays
        self.device = torch.device(     
            "cuda" if torch.cuda.is_available() else "cpu")     #Cuda graphic card present/cpu
        self.model = torch.load(model)  #Loads Pytorch model-Trained for our task
        self.gp = GlossProcess.load_checkpoint()        #Processing glosses    
        self.classes = self.gp.glosses  
        self.res_text: str = ""
        self.canvas = tkinter.Canvas(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)    #Creates a "canvas" object within application window.sets height and width based on vid_proc()
        self.canvas.pack()  #canvas placed properly
        self.text_label = tkinter.Label(self.window, text="Res text : ")    #
        self.text_label.pack()
        self.text_box = tkinter.Text(self.window, height=10)
        self.text_box.pack()
        self.update()
        self.window.mainloop()

    def update(self):   #Update application state and interface
        success, frame = self.vid_proc.get_frame()  #Frame retrieval-gives success/not
        if not success: 
            return      #Retrieval unsuccesful-returns
        if (len(self.seq)+1 == self.max_seq_len):   
            res = self.vid_proc.get_lm(frame)   #Obtaining landmark 
            if res:
                self.seq.append(proc_landmarks(res))    #landmarks are processed anda appended to the subarray 
                proc_seq = torch.from_numpy(
                    np.array(self.seq)).float().to(self.device)
                self.model.eval()
                with torch.no_grad():
                    x=proc_seq.unsqueeze(0)
                    out = self.model(x)
                res_class = torch.argmax(out,dim=1)
                self.res_text += self.classes[res_class.item()]+".\n"
                self.tts.say(self.classes[res_class.item()])
                self.tts.runAndWait()
                self.tts.stop()
                self.text_box.delete("1.0", tkinter.END)
                self.text_box.insert("1.0", chars=self.res_text)
                self.seq = []
                if len(self.res_text.splitlines()) > 5:
                    self.res_text = ""
        res = self.vid_proc.get_lm(frame)
        if res:
            self.seq.append(proc_landmarks(res))
        frame = frame if not res else self.vid_proc.draw_lm(frame, res)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)
        self.window.after(1, self.update)


def main():
    root = tkinter.Tk()
    ShitApp(root, "shit")


if __name__ == "__main__":
    main()
