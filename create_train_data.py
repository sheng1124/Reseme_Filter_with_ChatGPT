
import sys
import os
import json
import traceback
import re

import openai
import numpy as np
import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
import PySide6.QtMultimedia as QtMultimedia
import PyPDF2

import main as remain
from main import MODEL_1, API_KEY

class TrainUI(remain.FilterUI):

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        # 開啟正樣本資料夾
        self.p_input_folder_edit, input_folder_box = self.create_open_folder_input('選擇正樣本資料夾: ')
        main_layout.addWidget(input_folder_box)

        # 開啟負樣本資料夾
        self.n_input_folder_edit, input_folder_box = self.create_open_folder_input('選擇負樣本資料夾: ')
        main_layout.addWidget(input_folder_box)

        # 產生訓練資料按鈕
        self.data_btn, data_box = self.create_data_set_btn()
        main_layout.addWidget(data_box)

        # 建立進度條
        self.bar, bar_box = self.create_bar()
        main_layout.addWidget(bar_box)
        self.update_bar.connect(self.set_bar_value)

        # layout
        main_ui = QtWidgets.QWidget()
        main_ui.setLayout(main_layout)
        self.setCentralWidget(main_ui)
    
    def create_data_set_btn(self):
        data_box = QtWidgets.QGroupBox()
        line = QtWidgets.QVBoxLayout()
        data_btn = QtWidgets.QPushButton(text='開始產生訓練資料')
        data_btn.clicked.connect(self.send2create)
        data_btn.setEnabled(True)
        line.addWidget(data_btn)
        data_box.setLayout(line)
        return data_btn, data_box

    def count_total_len(self, folderpath):
        count = 0
        folder = os.listdir(folderpath.text())
        for pdf in folder:
            if pdf[-4:] == '.pdf':
                count += 1
        return count

    @QtCore.Slot()
    def send2create(self):
        np_list = [self.p_input_folder_edit, self.n_input_folder_edit]
        label_list = ["accept.", "reject."]
        data_set = []
        total_len = self.count_total_len(np_list[0])+ self.count_total_len(np_list[1])
        print('total_len', total_len)
        count = 0
        self.update_bar.emit(0)
        # 讀取所有正負樣本 pdf 檔
        for i, _ in enumerate(np_list):
            pdf_list = os.listdir(np_list[i].text())
            for pdf in pdf_list:
                if pdf[-4:] != '.pdf':
                    continue
                src = os.path.join(np_list[i].text(), pdf)
                reseme = self.less_tokens(self.pdf2str(src))

                num = 6 if i == 0 else 3

                for j in range(num):
                    # 透過openai 產生訓練文件(履歷分析資料)
                    response = self.inference_reseme(reseme)
                    if response is None:
                        continue
                    
                    content = response['choices'][0]['message']['content']
                    data = {"prompt":content + '->', "completion" : label_list[i]}
                    data_set.append(data)
                    
                    with open(f'{src[:-4]}_response_{j}.txt', 'w', encoding='utf-8') as outfile:
                        outfile.write(content)
                    
                count+=1
                self.update_bar.emit(int(count*100/total_len))

        # 保存到 df jsonl
        file_name = "training_data.jsonl"
        with open(file_name, 'w') as outfile:
            for entry in data_set:
                json.dump(entry, outfile)
                outfile.write('\n')
        
                

            

  

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = TrainUI()
    widget.resize(1200, 800)
    widget.show()
    sys.exit(app.exec())