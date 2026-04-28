from tkinter import *
from tkinter import ttk, messagebox


from PIL import ImageTk,Image
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from PIL import Image
import pytesseract
import cv2
import requests
#from corr import corr


root = Tk()
root.geometry('1260x750+0+10')
root.title('HAND WRITTEN TO TEXT CONVERTOR')


bg1 = PhotoImage(file='bg8.png')
bgLabel = Label(root, image=bg1)
bgLabel.place(x=0, y=0)

newline= Label(root)
uploaded_img=Label(root)
scrollbar = Scrollbar(root)
scrollbar.pack( side = RIGHT, fill = Y )
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"




def tab():
     
     cf1=Canvas(root, bg='#1F618D',width=1396, height=80)
     cf1.place(x=1,y=40)
tab()

titleLabel = Label(root, text='HAND WRITTEN TO TEXT CONVERTOR', font=('Times New Roman', 22, 'bold '), bg='#1F618D',
                   fg='white', )
titleLabel.place(x=400, y=60)

l1 = tk.Label(root,text='UPLOAD IMAGE',width=30,bg='#1F618D',
                   fg='WHITE',)

l1.place(x=500, y=200, width=300, height=30)
b1 = tk.Button(root, text='Upload File', width=20,command = lambda:upload_file())
   
b1.place(x=500, y=370, width=300)





entrycolumn1 = Entry(root,font=('times new roman', 14), bg='lightgray')
entrycolumn1.place(x=100, y=475, width=1150, height=200)



    
def extract(path):
    Actual_image = cv2.imread(path)
    Sample_img = cv2.resize(Actual_image,(400,350))
    Image_ht,Image_wd,Image_thickness = Sample_img.shape
    Sample_img = cv2.cvtColor(Sample_img,cv2.COLOR_BGR2RGB)
    texts = pytesseract.image_to_string(Sample_img)
    
    mystr = StringVar()
    mystr.set(texts)

    entrycolumn1 = Entry(root,font=('times new roman', 14), bg='lightgray',textvariable=mystr)
    entrycolumn1.place(x=100, y=475, width=1150, height=200)
         
   
     
  
 

    
    

def show_extract_button(path):
    extractBtn= Button(root,text="Extract text",command=lambda: extract(path),bg="#2f2f77",fg="gray",pady=2,padx=90,font=('Times',15,'bold'))
    extractBtn.pack(side=LEFT, padx=50, pady=400)
    extractBtn.place(x=500,y=430)


def upload_file():
    f_types = [('PNG Files', '*.png'),
    ('Jpg Files', '*.jpg')]    
    filename = tk.filedialog.askopenfilename(multiple=False,filetypes=f_types)
     
    img=Image.open(filename) 
    img=img.resize((150,110)) 
    img=ImageTk.PhotoImage(img)
    e1 =tk.Label(root)
    e1.place(x=530, y=240, width=240)
    e1.image = img
    e1['image']=img
    show_extract_button(filename)
    return filename   
root.mainloop() 
