from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QMovie
from PyQt5 import QtCore
import curve
import test
import image
import sys


class Ico(QWidget):
    def __init__(self):
        super(Ico, self).__init__()
        # self.initui()

    def initui(self):
        font_z = QFont()
        font_z.setFamily('华文行楷')
        font_z.setBold(True)
        font_z.setPointSize(9)
        font_z.setWeight(50)

        self.setGeometry(300, 300, 450, 400)
        self.setWindowTitle('人脸表情识别系统')
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./save/background1.jpg")))
        self.setPalette(palette)

        x1 = QPushButton('打开相机', self)
        x1.setFont(font_z)
        x1.setIcon(QIcon("./save/video.jpg"))
        x1.resize(120, 32)
        x1.move(50, 130)
        x1.clicked.connect(test.main)

        x2 = QPushButton('打开图片', self)
        x2.setFont(font_z)
        x2.setIcon(QIcon("./save/img.jpg"))
        x2.resize(120, 32)
        x2.move(50, 80)
        x2.clicked.connect(image.test)

        x3 = QPushButton('绘制曲线', self)
        x3.setFont(font_z)
        x3.setIcon(QIcon("./save/curve.jpg"))
        x3.resize(120, 32)
        x3.move(50, 180)
        x3.clicked.connect(curve.opencurve)

        qbtn = QPushButton('退出', self)
        qbtn.setFont(font_z)
        qbtn.setIcon(QIcon("./save/exit.jpg"))
        qbtn.clicked.connect(QCoreApplication.instance().quit)
        qbtn.resize(120, 32)
        qbtn.move(50, 280)
        # qbtn.setFlat(True)

        gif_m = QLabel(self)
        gif_m.setGeometry(QtCore.QRect(200, 100, 200, 150))

        movie = QMovie("./save/mo1.gif")
        movie.setScaledSize(gif_m.size())
        gif_m.setMovie(movie)
        movie.start()
        self.show()


if __name__ == '__main__':
    path = 'confusion_matrix.png'
    app = QApplication(sys.argv)
    ex = Ico()
    ex.initui()
    sys.exit(app.exec_())
