import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def select():
    global frame2, path, img1
    _path = filedialog.askopenfile()
    path.set(_path.name)
    _img = Image.open(path.get()).resize((400, 400))
    img1 = ImageTk.PhotoImage(_img)
    tk.Label(frame2, text='原图').grid(row=0, column=0)
    tk.Label(frame2, image=img1).grid(row=1, column=0)


def run():
    global frame2, path, img2
    filename = os.path.split(path.get())[1].split('.')[0]
    os.system('python ../predict.py -m ../checkpoints/new-p7f32-1_epoch50.pth -p 7 -ifc 32 -i {} -o ../tmp/{}_OUT.png'
              .format(path.get(), filename))
    _img = Image.open('../tmp/{}_OUT.png'.format(filename)).resize((400, 400))
    img2 = ImageTk.PhotoImage(_img)
    tk.Label(frame2, text='分割图').grid(row=0, column=1)
    tk.Label(frame2, image=img2).grid(row=1, column=1)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('基于深度学习的磁共振医学图像分割辅助标注算法展示')
    # root.geometry('500x300')
    frame1 = tk.Frame(root)
    frame2 = tk.Frame(root)
    frame1.pack(side=tk.TOP)
    frame2.pack(side=tk.BOTTOM)

    path = tk.StringVar()
    img1 = None
    img2 = None

    tk.Label(frame1, text='图片路径：').grid(row=0, column=0)
    tk.Entry(frame1, textvariable=path, width=70).grid(row=0, column=1)
    tk.Button(frame1, text='选择', command=select).grid(row=0, column=2)
    tk.Button(frame1, text='运行', command=run).grid(row=0, column=3)

    root.mainloop()
