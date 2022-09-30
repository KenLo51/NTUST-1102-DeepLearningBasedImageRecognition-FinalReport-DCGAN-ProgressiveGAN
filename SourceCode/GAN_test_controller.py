# build in libries
import subprocess
import threading
import os
import time

# additional libries
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import torch
import cv2

# custom libries
from GAN_test_UI import Ui_MainWindow
import GAN_test_config as config

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        print("init")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.randvec = np.zeros(shape=(config.numRandomLatent,), dtype=np.float32)
        self.img = None
        self.gen_Tt = None
        self.updateSliders_th = None

        # get devices
        gpuIdx = 0
        self.device = torch.device("cpu:0")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpuIdx}")
            print(f"Using cuda with device-{gpuIdx} ({torch.cuda.get_device_name(gpuIdx)})")
        else:
            cpuName = subprocess.check_output(["wmic","cpu","get", "name"]).decode("utf8").strip().split("\n")[1]
            print(f"Using CPU ({cpuName})")


        self.generatorModel = config.Generator().to(self.device)
        if(os.path.exists(config.modelFileName)):
            checkpoint = torch.load( config.modelFileName )
            self.generatorModel.load_state_dict(checkpoint["generator_state_dict"])
            self.generatorModel.eval()
        else:
            print("Can not find trained data")

    def setup_control(self):
        print("setup_control")
        self.ui.pushButton_random.clicked.connect(self.clickCb_randomBtn)
        for i, (horizontalSlider, valLable) in enumerate(self.ui.horizontalSliders) :
            horizontalSlider.valueChanged.connect(lambda val, idx=i:self.valChangeCB_Sliders(val, idx))

        self.ui.pushButton_Generate.clicked.connect(self.clickCb_generateBtn)
        self.ui.pushButton_saveVector.clicked.connect(self.clickCb_saveVecBtn)
        self.ui.pushButton_saveImage.clicked.connect(self.clickCb_saveImgBtn)
        self.ui.pushButton_saveAll.clicked.connect(self.clickCb_saveAllBtn)
        
    def valChangeCB_Sliders(self, val, idx):
        horizontalSlider, label = self.ui.horizontalSliders[idx]
        label.setText(f"{val/100:7.2f}")
        self.randvec[idx] = val/100
        if self.ui.checkBox_generateWhileChange.isChecked():
            self.clickCb_generateBtn()
    
    def updateRandomSliders_task(self):
        mean = self.ui.spinBox_mean.value()
        std = self.ui.spinBox_std.value()
        self.randvec = np.random.normal(mean, std, size=(config.numRandomLatent,)).astype(np.float32)
        for i, (horizontalSlider, valLabel) in enumerate(self.ui.horizontalSliders) :
            val = self.randvec[i]
            horizontalSlider.setValue(val*100)
        if(self.ui.checkBox_generateWhileChange.isChecked()):
            self.clickCb_generateBtn()

    def genetareImage_task(self):
        self.oriimg = self.generatorModel(torch.from_numpy(self.randvec).to(self.device))[0]*255
        self.oriimg = self.oriimg.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        self.img = cv2.resize(self.oriimg, dsize=(551, 551))
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QtGui.QImage(self.img.tobytes(), width, height, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.label_resultImg.setPixmap(QtGui.QPixmap.fromImage(self.qimg))

    def clickCb_randomBtn(self):
        if self.updateSliders_th is None :
            self.updateSliders_th = threading.Thread(target=self.updateRandomSliders_task)
            self.updateSliders_th.start()
            return
        if self.updateSliders_th.is_alive() :
            return
        self.updateSliders_th = threading.Thread(target=self.updateRandomSliders_task)
        self.updateSliders_th.start()

    def clickCb_generateBtn(self):
        if self.gen_Tt is None :
            self.gen_Tt = threading.Thread(target=self.genetareImage_task)
            self.gen_Tt.start()
            return
        if self.gen_Tt.is_alive() :
            return
        self.gen_Tt = threading.Thread(target=self.genetareImage_task)
        self.gen_Tt.start()

    def clickCb_saveVecBtn(self):
        timeStr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) 
        fileName = f"{timeStr}_vec.txt"
        np.savetxt(fileName, self.randvec, delimiter=',')

    def clickCb_saveImgBtn(self):
        timeStr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) 
        fileName = f"{timeStr}_img.png"
        if(self.img is None):
            return
        cv2.imwrite(fileName, self.oriimg)

    def clickCb_saveAllBtn(self):
        timeStr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) 
        imgFileName = f"{timeStr}_img.png"
        vecFileName = f"{timeStr}_vec.txt"
        if(self.img is None):
            return
        cv2.imwrite(imgFileName, self.oriimg)
        np.savetxt(vecFileName, self.randvec, delimiter=',')


    # test function
    def clickCb_pushButton_random(self):
        self.clicked_counter += 1
        print(f"You clicked {self.clicked_counter} times.")