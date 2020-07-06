# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:17:24 2020

@author: COMPUTER
"""


import sys, string, os, subprocess


from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QSizePolicy, \
    QWidget, QFileDialog, QTextEdit, QSizePolicy, QMessageBox, QHBoxLayout, QGridLayout
from PyQt5.QtCore import Qt, QStringListModel, QSize, QTimer, QFile, QIODevice, QTextStream, QTextCodec

import os
import cv2

class UI_Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        
        # Create a timer.
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        
        # Create a layout.
        #x = QGridLayout()
        layout = QHBoxLayout()
        
        # Add a label
        self.label = QLabel()
        self.label.setFixedSize(300, 300)
        #pixmap = self.resizeImage(filename)
        #self.label.setPixmap(pixmap)
        layout.addWidget(self.label)

        # Add a button
        button_layout = QVBoxLayout()        

        # Add a text area
        self.results = QTextEdit()
        self.results.setFixedSize(250, 130)
        #self.readBarcode(filename)
        button_layout.addWidget(self.results)

        # Add a button
        btnCamera = QPushButton("Load an image")
        btnCamera.clicked.connect(self.pickFile)
        button_layout.addWidget(btnCamera)

        btnCamera = QPushButton("Translate")
        btnCamera.clicked.connect(self.runFile)
        button_layout.addWidget(btnCamera)

               

        layout.addLayout(button_layout)

        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle("Penerjemah Aksara Batak")
        self.setFixedSize(600, 350)
        
        #runUI
        self.logic = 0
        self.value = 1
        #filenames = glob.glob(path + "/*.asc") 
    # https://stackoverflow.com/questions/1414781/prompt-on-exit-in-pyqt-application
    def closeEvent(self, event):
    
        msg = "Close the app?"
        reply = QMessageBox.question(self, 'Message', 
                        msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            self.stopCamera()
        else:
            event.ignore()

    def takePicture(self):
        self.logic=2	

    def resizeImage(self, filename):
        pixmap = QPixmap(filename)
        lwidth = self.label.maximumWidth()
        pwidth = pixmap.width()
        lheight = self.label.maximumHeight()
        pheight = pixmap.height()

        wratio = pwidth * 1.0 / lwidth
        hratio = pheight * 1.0 / lheight

        if pwidth > lwidth or pheight > lheight:
            if wratio > hratio:
                lheight = pheight / wratio
            else:
                lwidth = pwidth / hratio

            scaled_pixmap = pixmap.scaled(lwidth, lheight)
            return scaled_pixmap
        else:
            return pixmap

    def pickFile(self):
        
        self.stopCamera()
        # Load an image file.
        filename = QFileDialog.getOpenFileName(self, 'Open file')
        pixmap = self.resizeImage(filename[0])
        self.label.setPixmap(pixmap)

    
    #buat baca ouput langsung isi nya
    def readOutput(self):
        #sesuaikan dir file
        file = QFile("/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/BFS/outputBFS.txt")
        
        if file.open(QIODevice.ReadOnly | QIODevice.Text):
            stream = QTextStream(file)
            while not stream.atEnd():   
                line = file.readLine()
                line.append(stream.readLine()+"\n")
                encodedString = line.append(stream.readLine()+"\n")
                codec = QTextCodec.codecForName("KOI8-R")
                string = codec.toUnicode(encodedString)
                self.results.setText(string)
        file.close();        

    #untuk nge run file applecript
    def runFile(self):
        myCmd = 'python test_model.py ./models/CNN.pth ./models/voc-model-labels.txt ./outs/ImageSets/test.txt'
        myCmd2 = 'python /Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/BFS/bfs.py'
        os.system(myCmd)
        os.system(myCmd2)     
        
        file = QFile("/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/BFS/outputBFS.txt")
        
        if file.open(QIODevice.ReadOnly | QIODevice.Text):
            stream = QTextStream(file)
            while not stream.atEnd():   
                line = file.readLine()
                line.append(stream.readLine()+"\n")
                encodedString = line.append(stream.readLine()+"\n")
                codec = QTextCodec.codecForName("KOI8-R")
                string = codec.toUnicode(encodedString)
                self.results.setText(string)
        file.close();
        
        
    
    def stopCamera(self):
        self.timer.stop()

    # https://stackoverflow.com/questions/41103148/capture-webcam-video-using-pyqt
    def nextFrameSlot(self):
        rval, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

        out = ''

        self.results.setText(out)
            
def main():
    app = QApplication(sys.argv)
    ex = UI_Window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()