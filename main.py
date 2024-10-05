from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal
from ui.mainui_ui import Ui_MainWindow
from model.change import removebg
import os
import torch
from model.briarmbg import BriaRMBG
import shutil

class ImageProcessingThread(QThread):
    image_processed = Signal(str)

    def __init__(self, file_path,net,device):
        super().__init__()
        self.file_path = file_path
        self.net = net
        self.device = device
        

    def run(self):
        no_bg_image = removebg(self.file_path,self.net,self.device)
        no_bg_image.save("./temp/result_no_bg.png")
        self.image_processed.emit("./temp/result_no_bg.png")

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化
        self.loadmodel = False
        self.file_path = ""
        self.picture_name = ""
        self.original_stylesheet = self.ui.bodypix.styleSheet()
        self.save_position = "./out"
        self.ui.lineEdit.setText(self.save_position)
        print("默认地址：", self.save_position)

        # 绑定按钮
        self.ui.select.clicked.connect(self.selectfile)
        self.ui.reset.clicked.connect(self.goback)
        self.ui.menu.clicked.connect(self.switchpage)
        self.ui.setting.clicked.connect(self.switchpage)
        self.ui.bodypix.clicked.connect(self.openpix)
        self.ui.alalys.clicked.connect(self.workhot)
        self.ui.clear.clicked.connect(self.clearpix)
        self.ui.getout.clicked.connect(self.outpicture)


    def switchpage(self):
        sender = self.sender()
        if sender == self.ui.menu:
            self.ui.stackedWidget.setCurrentIndex(0)
            if self.loadmodel == False:
                self.show_info("注意第一次分析用时比较久")
        elif sender == self.ui.setting:
            self.ui.stackedWidget.setCurrentIndex(1)

    def selectfile(self):
        self.save_position = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
        self.ui.lineEdit.setText(self.save_position)
        print("改动后：", self.save_position)

    def goback(self):
        self.save_position = "./out" 
        self.ui.lineEdit.setText(self.save_position)
        print("恢复默认地址：", self.save_position)

    def openpix(self):
        self.clearpix()
        options = QFileDialog.Options()
        self.file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)", options=options)
        file_name_with_ext = os.path.basename(self.file_path)
        file_name_without_ext, ext = os.path.splitext(file_name_with_ext)
        self.picture_name = file_name_without_ext
        print("图片名：", self.picture_name)
        if self.file_path:  
            pixmap = QPixmap(self.file_path)  
            if pixmap.isNull():  
                print("无法加载图像")
                self.show_warning("无法加载图像")
            else:  
                self.ui.bodypix.setStyleSheet("background-color: white;")
                scaled_pixmap = pixmap.scaled(self.ui.bodypix.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) 
                self.ui.bodypix.setIcon(scaled_pixmap)

    def workhot(self):
        if self.loadmodel == False:
            # 模型加载和设备设置
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4", cache_dir="./model").to(self.device).eval()
            print("模型加载成功")
            self.loadmodel = True
        if self.file_path:
            self.thread = ImageProcessingThread(self.file_path,self.net,self.device)
            self.thread.image_processed.connect(self.display_processed_image)
            self.thread.start()
        else:
            print("目前无图片处理")
            self.show_warning("目前无图片处理")

    def display_processed_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.ui.bodypix.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.bodypix.setIcon(scaled_pixmap)
        self.file_path = ""

    def clearpix(self):
        self.file_path = ""
        self.ui.bodypix.setIcon(QPixmap())
        self.picture_name = ""
        self.ui.bodypix.setStyleSheet(self.original_stylesheet)

    def outpicture(self):
        if self.picture_name:
            self.picture_name += ".png"
            new_file = os.path.join(self.save_position + "/" + self.picture_name) 
            print("导出去背景图：", new_file)
            shutil.copy2("./temp/result_no_bg.png", new_file)
            self.show_info(f"成功导出图片至 {new_file}")  # 调用 show_info
        else:
            print("目前无图片导出")
            self.show_warning("目前无图片导出")

    def show_warning(self, words):  
        msg = QMessageBox()  
        msg.setIcon(QMessageBox.Warning)  
        msg.setText(words)    
        msg.setWindowTitle("警告")         
        msg.setStandardButtons(QMessageBox.Ok)  
        msg.exec_()

    def show_info(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)  # 使用正确的枚举值
        msg.setText(message)
        msg.setWindowTitle("信息")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == '__main__':
    try:
        app = QApplication([])
        window = MyWindow()
        window.show()
        app.exec()
    except Exception as e:
        print(f"An error occurred: {e}")
