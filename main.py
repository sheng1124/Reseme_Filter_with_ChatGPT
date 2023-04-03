import sys
import os
import shutil

from dotenv import load_dotenv
import numpy as np
import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
import PySide6.QtMultimedia as QtMultimedia
import PyPDF2
from transformers import GPT2TokenizerFast
from transformers import BartForConditionalGeneration, BartTokenizer
import openai



load_dotenv('.env')
API_KEY = os.getenv('OPENAI_API')
SYSTEM_PROMPT = "我想讓你扮演一個科技公司的HR，目前正在招募儲能數據分析工程師，以下是儲能數據分析工程師的應徵條件:1. 碩士以上，數學/資工/統計等熟悉結構化資料分析之演算法的相關科系，並具儲能系統知識背景為佳，2. 具備2年以上資料處理/數據分析或其他專案工作經驗，3. 熟悉數據分析程式或工具(Python、R、SAS、SPSS等)，熟Python尤佳，4. 熟悉資料庫(MS-SQL、Oracle等)，熟MongoDB尤佳，5. 熟悉至少一種機器學習或資料分析工具或套件與模型建置基礎流程Scikit-learn、Keras、TensorFlow、PyTorch，6. 具基礎智慧電網、電力市場、充電樁營運、能源最佳化運算等領域知識尤佳。職務工作內容:1. 對各儲能案場收集之數據，進行數據分析與模型架構研究，2. 彙整、分析與定期產出各類型的監測資料或報告，3. 數據預處理、特徵工程、模型選擇/融合、模型訓練/測試、數據問題分析及解決、模型評估的程式開發，4. 機器學習、深度學習或其他相關演算法開發，5. 由數據角度協助儲能系統設計與建置，優化儲能系統效能。我會給你應徵者的履歷，你要判斷應徵者有幾項符合上述的應徵條件，那些不符合，並給予分析，你的回應必須遵守以下規則，1. 不能出現應徵者的姓名、暱稱。"
MODEL_1 = 'gpt-3.5-turbo'
FT_MODEL_1 = 'curie:ft-personal-2023-04-03-03-35-54'

TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2', return_attention_mask = False)

class FilterUI(QtWidgets.QMainWindow):
    update_bar = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        # 開啟資料夾
        self.input_folder_edit, input_folder_box = self.create_open_folder_input()
        main_layout.addWidget(input_folder_box)

        # 開始篩選按鈕
        self.filter_btn, filter_box = self.create_filter_btn()
        main_layout.addWidget(filter_box)

        # 建立進度條
        self.bar, bar_box = self.create_bar()
        main_layout.addWidget(bar_box)
        self.update_bar.connect(self.set_bar_value)

        # 文件預覽
        


        # layout
        main_ui = QtWidgets.QWidget()
        main_ui.setLayout(main_layout)
        self.setCentralWidget(main_ui)


    def create_open_folder_input(self, line_edit = '選擇資料夾: '):
        input_folder_box = QtWidgets.QGroupBox()
        layout1 = QtWidgets.QVBoxLayout()
        layout1.setAlignment(QtCore.Qt.AlignTop)
        line = QtWidgets.QHBoxLayout()
        line.addWidget(QtWidgets.QLabel(line_edit))
        input_folder_edit = QtWidgets.QLineEdit(os.getcwd())
        open_input_folder= input_folder_edit.addAction(
            qApp.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon), QtWidgets.QLineEdit.TrailingPosition
        )
        def when_open_folder():
            folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Directory", os.getcwd(), QtWidgets.QFileDialog.ShowDirsOnly)
            if folder_path:
                dest_dir = QtCore.QDir(folder_path)
                input_folder_edit.setText(QtCore.QDir.fromNativeSeparators(dest_dir.path()))
        open_input_folder.triggered.connect(when_open_folder)
        line.addWidget(input_folder_edit)
        layout1.addItem(line)
        input_folder_box.setLayout(layout1)
        return input_folder_edit, input_folder_box

    def create_filter_btn(self):
        filter_box = QtWidgets.QGroupBox()
        line = QtWidgets.QVBoxLayout()
        filter_btn = QtWidgets.QPushButton(text='開始過濾')
        filter_btn.clicked.connect(self.send2filter)
        filter_btn.setEnabled(True)
        line.addWidget(filter_btn)
        filter_box.setLayout(line)
        return filter_btn, filter_box

    @QtCore.Slot()
    def send2filter(self):
        self.update_bar.emit(0)
        # 讀取所有在 input 資料夾下的 pdf 檔
        pdf_list = os.listdir(self.input_folder_edit.text())
        count = 0
        for pdf in pdf_list:
            if pdf[-4:] != '.pdf':
                continue
            src = os.path.join(self.input_folder_edit.text(), pdf)
            reseme = self.pdf2str(src)

            reseme_k = self.less_tokens(reseme)

            # step 1 取得履歷分析
            content, classification = '', ''
            response = self.inference_reseme(reseme_k)
            if not response is None:
                content = response['choices'][0]['message']['content'] + '->'

                # step 2 把分析結果給微調模型分類
                response = self.anlysis_classification(content)
                if not response is None:
                    classification = response['choices'][0]['text']
            
            print(pdf, 'classification = ', classification)

            # 將pdf以分類結果複製到對應的資料夾
            if 'accept' in classification:
                output_folder = 'results/accept'
            elif 'reject' in classification:
                output_folder = 'results/reject'
            else:
                output_folder = 'results/noresponse'

            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)

            newfile_pdf = os.path.join(output_folder, pdf)
            shutil.copyfile(src, newfile_pdf)

            # 儲存分析結果
            if not content:
                #return reseme, content
                continue

            newfile_response = os.path.join(output_folder, f'{pdf[:-4]}_response.txt')
            
            with open(newfile_response, 'w', encoding='utf-8') as outfile:
                outfile.write(content)
            
            #return reseme, content


        
    
    def create_bar(self):
        box = QtWidgets.QGroupBox()
        line = QtWidgets.QVBoxLayout()
        bar = QtWidgets.QProgressBar()
        bar.setRange(0,100)
        bar.setValue(0)
        bar.setFormat('%p%')
        line.addWidget(bar)
        box.setLayout(line)
        return bar, box

    @QtCore.Slot(int)
    def set_bar_value(self, value:int):
        self.bar.setValue(value)

    def reduce_tokens(self, text, target):
        t = TOKENIZER(text)

        # Truncate tokens
        truncated_tokens = t["input_ids"][:target]

        # Decode the truncated tokens back to text
        truncated_text = TOKENIZER.decode(truncated_tokens)

        f = open('eee.txt', 'w', encoding= 'utf-8')
        f.write(truncated_text)
        f.close()

        return truncated_text

    def less_tokens(self, text):
        # Tokenize the text
        t = TOKENIZER(text.replace('  ', ' ').replace(' \n', '\n').replace('\n ', '\n').replace('\n\n\n', '\n').replace('\n\n', '\n')  )

        # Truncate tokens
        truncated_tokens = t["input_ids"]

        # Decode the truncated tokens back to text
        truncated_text = TOKENIZER.decode(truncated_tokens)

        return truncated_text

    def create_pdf_viewer(self):
        pass
    
    def pdf2str(self, pdf_file:str):
        reader = PyPDF2.PdfReader(pdf_file)
        s = ''
        for page in reader.pages:
            s += page.extract_text()
        return s
        
    def batch_pdf2txt(self):
        self.update_bar.emit(0)
        # 讀取所有在 input 資料夾下的 pdf 檔
        pdf_list = os.listdir(self.input_folder_edit.text())
        count = 0
        for pdf in pdf_list:
            if pdf[-4:] != '.pdf':
                continue
            src = os.path.join(self.input_folder_edit.text(), pdf)
            s = self.pdf2str(src)

            _, k = self.less_tokens(SYSTEM_PROMPT, s)

            txt_file = src[:-4] + '.txt'
            with open(txt_file, '+w', encoding='utf-8') as f:
                f.write(s)

            count+=1
            self.update_bar.emit(int(count*100/len(pdf_list)))

    def batch_rename(self):
        # 讀取所有在 input 資料夾下的 pdf 檔
        pdf_list = os.listdir(self.input_folder_edit.text())
        for pdf in pdf_list:
            if pdf[-4:] != '.pdf':
                continue
            src = os.path.join(self.input_folder_edit.text(), pdf)
            dst = os.path.join(self.input_folder_edit.text(), 'xxxxx' + pdf)
            os.rename(src, dst)
        
        pdf_list = os.listdir(self.input_folder_edit.text())
        pdf_list = sorted(pdf_list, key=lambda _ :np.random.randint(10 * len(pdf_list)))
        print(pdf_list)
        for i, pdf in enumerate(pdf_list):
            if pdf[-4:] != '.pdf':
                continue
            src = os.path.join(self.input_folder_edit.text(), pdf)
            dst = os.path.join(self.input_folder_edit.text(), f'{i}.pdf')
            os.rename(src, dst)
    
    def create_message(self, msg):
        messages = [
            {
                "role" : "system", 
                "content": SYSTEM_PROMPT},
            {
                "role" : "user",
                "content": msg
            }]
        
        return messages

    # 使用 open ai 分析履歷
    def inference_reseme(self, reseme:str):
        messages = self.create_message(reseme)
        try:
            response = openai.ChatCompletion.create(api_key=API_KEY, model = MODEL_1, temperature = 1, top_p = 0.5, messages = messages)
        except openai.error.InvalidRequestError as e:
            reseme = self.reduce_tokens(reseme, 4097)
            messages = self.create_message(reseme)
            try:
                response = openai.ChatCompletion.create(api_key=API_KEY, model = MODEL_1, temperature = 1, messages = messages)
            except openai.error.InvalidRequestError as e:
                print('error in ', reseme[:100])
                print(e)
                response = None
        
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            print(error_message)
            response = None

        print(response)
        return response

    # 使用 openai 分析履歷後的資料做分類
    def anlysis_classification(self, content:str):
        try:
            response = openai.Completion.create(
                api_key=API_KEY,
                model = FT_MODEL_1,
                prompt = content,
                temperature=0,
                max_tokens=10,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["."]
            )
            print(response)
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            print(error_message)
            response = None
        
        return response

class FilterThread(QtCore.QThread):
    update_bar = QtCore.Signal(int)
    finished = QtCore.Signal()
    def __init__(self, cfg, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
    
    def run(self):
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = FilterUI()
    widget.resize(1200, 800)
    widget.show()
    sys.exit(app.exec())