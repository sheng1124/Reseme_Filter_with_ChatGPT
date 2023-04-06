import sys
import os
import shutil
import time

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
SYSTEM_PROMPT_TEMPLATE = "我想讓你扮演一個科技公司的HR，目前正在招募#####，以下是#####的應徵條件:@@@@@。我會給你應徵者的履歷，你要對應徵者和每一項應徵條件打分數(滿分10分)，你要用最高的標準來判斷，若不符合條件應該給予0分，例如:最高學歷雖然是碩士畢業，但是不符合要求的科系應該給予0分，最後並給予評語。你的回應必須遵守以下規則，1. 不能出現應徵者的姓名、暱稱 2.省略應徵條件內容，以'條件x:'替代。"
RESTR = "1. 碩士以上，數學/資工/統計等熟悉結構化資料分析之演算法的相關科系，並具儲能系統知識背景為佳，2. 具備2年以上資料處理/數據分析或其他專案工作經驗，3. 熟悉數據分析程式或工具(Python、R、SAS、SPSS等)，熟Python尤佳，4. 熟悉資料庫(MS-SQL、Oracle等)，熟MongoDB尤佳，5. 熟悉至少一種機器學習或資料分析工具或套件與模型建置基礎流程Scikit-learn、Keras、TensorFlow、PyTorch，6. 具基礎智慧電網、電力市場、充電樁營運、能源最佳化運算等領域知識尤佳。職務工作內容:1. 對各儲能案場收集之數據，進行數據分析與模型架構研究，2. 彙整、分析與定期產出各類型的監測資料或報告，3. 數據預處理、特徵工程、模型選擇/融合、模型訓練/測試、數據問題分析及解決、模型評估的程式開發，4. 機器學習、深度學習或其他相關演算法開發，5. 由數據角度協助儲能系統設計與建置，優化儲能系統效能"
TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2', return_attention_mask = False)


def pdf2str(pdf_filepath:str):
    reader = PyPDF2.PdfReader(pdf_filepath)
    s = ''
    for page in reader.pages:
        s += page.extract_text()
    return s

def less_tokens(text):
    # Tokenize the text
    t = TOKENIZER(text.replace('  ', ' ').replace(' \n', '\n').replace('\n ', '\n').replace('\n\n\n', '\n').replace('\n\n', '\n')  )

    # Truncate tokens
    truncated_tokens = t["input_ids"]

    # Decode the truncated tokens back to text
    truncated_text = TOKENIZER.decode(truncated_tokens)

    return truncated_text

def create_alz_model_prompt(system_prompt, msg):
    messages = [
        {
            "role" : "system", 
            "content": system_prompt},
        {
            "role" : "user",
            "content": msg
        }]
    return messages

def reduce_tokens(text, target):
    t = TOKENIZER(text)

    # Truncate tokens
    truncated_tokens = t["input_ids"][:target]

    # Decode the truncated tokens back to text
    truncated_text = TOKENIZER.decode(truncated_tokens)

    f = open('eee.txt', 'w', encoding= 'utf-8')
    f.write(truncated_text)
    f.close()

    return truncated_text


class BasicUI(QtWidgets.QMainWindow):
    openai_api_key = os.getenv('OPENAI_API')
    vancancies = os.getenv('VACANCIES')
    van_condition = os.getenv('VACANCIES_CONDITION')

    def create_cfg_input(self):
        input_cfg_box = QtWidgets.QGroupBox()
        layout1 = QtWidgets.QVBoxLayout()
        layout1.setAlignment(QtCore.Qt.AlignTop)
        line = QtWidgets.QHBoxLayout()
        line.setAlignment(QtCore.Qt.AlignLeft)
        line.addWidget(QtWidgets.QLabel(f'{"輸入 OpenAI API key:":<35}'))
        input_openai_api_key_edit = QtWidgets.QLineEdit(self.openai_api_key)
        line.addWidget(input_openai_api_key_edit)
        layout1.addItem(line)

        line = QtWidgets.QHBoxLayout()
        line.setAlignment(QtCore.Qt.AlignLeft)
        line.addWidget(QtWidgets.QLabel(f'{"選擇分類履歷用的微調模型:":<20}'))
        cf_models = self.get_finetune_model_list() # classification_models
        cf_model_combo = QtWidgets.QComboBox()
        cf_model_combo.addItems(cf_models)
        line.addWidget(cf_model_combo)
        refresh_model_btn, _ = self.create_btn('刷新模型列表', self.refresh_cf_list)
        line.addWidget(refresh_model_btn)
        layout1.addItem(line)

        line = QtWidgets.QHBoxLayout()
        line.setAlignment(QtCore.Qt.AlignLeft)
        line.addWidget(QtWidgets.QLabel(f'{"選擇分析履歷用的模型:":<26}'))
        analysis_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
        alz_model_combo = QtWidgets.QComboBox()
        alz_model_combo.addItems(analysis_models)
        line.addWidget(alz_model_combo)
        layout1.addItem(line)

        line = QtWidgets.QHBoxLayout()
        line.setAlignment(QtCore.Qt.AlignLeft)
        line.addWidget(QtWidgets.QLabel(f'{"輸入職缺名稱:":<38}'))
        input_vancancies_edit = QtWidgets.QLineEdit(self.vancancies)
        input_vancancies_edit.setFixedWidth(160)
        line.addWidget(input_vancancies_edit)
        line.addWidget(QtWidgets.QLabel(f'{" PS: 篩選之前要確認職缺有對應的微調模型"}'))
        line.addItem(QtWidgets.QSpacerItem(40,20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        layout1.addItem(line)

        line = QtWidgets.QHBoxLayout()
        line.setAlignment(QtCore.Qt.AlignLeft)
        line.addWidget(QtWidgets.QLabel(f'{"輸入職缺條件:":<38}'))
        input_van_condition_edit = QtWidgets.QLineEdit(self.van_condition)
        line.addWidget(input_van_condition_edit)
        layout1.addItem(line)
        
        input_cfg_box.setLayout(layout1)
        cfg_input_dict = {
            'input_openai_api_key_edit' : input_openai_api_key_edit,
            'cf_model_combo' : cf_model_combo,
            'alz_model_combo' : alz_model_combo,
            'input_vancancies_edit': input_vancancies_edit,
            'input_van_condition_edit':input_van_condition_edit
        }
        return cfg_input_dict, input_cfg_box
    
    def create_open_folder_input(self, line_edit = f'{"選擇資料夾:":<41}'):
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

    def create_open_file_input(self, line_edit = f'{"選擇文件:":<37}', ftr = 'ALL Files(*)'):
        box = QtWidgets.QGroupBox()
        layout1 = QtWidgets.QVBoxLayout()
        layout1.setAlignment(QtCore.Qt.AlignTop)
        line = QtWidgets.QHBoxLayout()
        line.addWidget(QtWidgets.QLabel(line_edit))
        edit = QtWidgets.QLineEdit(os.getcwd())
        open_file = edit.addAction(
            qApp.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon), QtWidgets.QLineEdit.TrailingPosition
        )
        def when_open_file():
            file_path = QtWidgets.QFileDialog.getOpenFileName(parent=self, dir=os.getcwd(), filter=ftr)
            if file_path:
                edit.setText(file_path[0])
        open_file.triggered.connect(when_open_file)
        
        line.addWidget(edit)
        layout1.addItem(line)
        box.setLayout(layout1)
        return edit, box
    
    def create_btn(self, btn_text, click_func):
        line = QtWidgets.QVBoxLayout()
        btn = QtWidgets.QPushButton(text=btn_text)
        btn.clicked.connect(click_func)
        line.addWidget(btn)
        return btn, line
    
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
    
    def get_finetune_model_list(self):
        cf_models = [] # classification_models
        try:
            model_list = openai.Model.list(api_key= self.openai_api_key)
            for model in model_list['data']:
                if ':ft-' in model.id:
                    cf_models.append(model.id)
        except Exception:
            pass
        return cf_models
    
    @QtCore.Slot()
    def refresh_cf_list(self):
        self.openai_api_key = self.input_openai_api_key_edit.text()

        cf_models = self.get_finetune_model_list()
        if not cf_models:
            QtWidgets.QMessageBox.warning(self, "警告", "你需要輸入正確的OPENAI API KEY 或需要去微調模型") 

        self.cf_model_combo.clear()
        self.cf_model_combo.addItems(cf_models)

    def update_input_cfg(self):
        self.openai_api_key = self.input_openai_api_key_edit.text()
        self.cf_model = self.cf_model_combo.currentText()
        self.alz_model = self.alz_model_combo.currentText()
        self.vancancies = self.input_vancancies_edit.text()
        self.van_condition = self.input_van_condition_edit.text()
    
    def get_input_cfg(self):
        cfg = {
            'openai_api_key' : self.openai_api_key,
            'cf_model' : self.cf_model,
            'alz_model' : self.alz_model,
            'vancancies' : self.vancancies,
            'van_condition' : self.van_condition
        }
        return cfg

class FilterUI(BasicUI):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        # 參數設定
        cfg_input_dict, input_cfg_box = self.create_cfg_input()
        self.input_openai_api_key_edit = cfg_input_dict['input_openai_api_key_edit']
        self.cf_model_combo = cfg_input_dict['cf_model_combo']
        self.alz_model_combo = cfg_input_dict['alz_model_combo']
        self.input_vancancies_edit = cfg_input_dict['input_vancancies_edit']
        self.input_van_condition_edit = cfg_input_dict['input_van_condition_edit']
        main_layout.addWidget(input_cfg_box)

        # 開啟資料夾
        self.input_folder_edit, input_folder_box = self.create_open_folder_input()
        main_layout.addWidget(input_folder_box)

        # 開始篩選按鈕
        self.filter_btn, filter_box = self.create_filter_btn()
        main_layout.addWidget(filter_box)

        # 建立進度條
        self.bar, bar_box = self.create_bar()
        main_layout.addWidget(bar_box)

        # 文件預覽
        


        # layout
        main_ui = QtWidgets.QWidget()
        main_ui.setLayout(main_layout)
        self.setCentralWidget(main_ui)

    @QtCore.Slot()
    def filter_finish(self):
        self.filter_btn.setEnabled(True)

    @QtCore.Slot()
    def send2filter(self):
        self.update_input_cfg()
        system_prompt =  SYSTEM_PROMPT_TEMPLATE.replace('#####', self.vancancies).replace('@@@@@', self.van_condition)
        cfg = self.get_input_cfg()
        cfg['system_prompt'] = system_prompt
        cfg['input_folder'] = self.input_folder_edit.text()
        self.filter_start = FilterThread(cfg)
        self.filter_start.update_bar.connect(self.set_bar_value)
        self.filter_start.finished.connect(self.filter_finish)

        self.filter_btn.setEnabled(False)
        self.filter_start.start()

    def create_filter_btn(self):
        filter_box = QtWidgets.QGroupBox()
        filter_btn, line = self.create_btn('開始過濾', self.send2filter)
        filter_box.setLayout(line)
        return filter_btn, filter_box

    def create_pdf_viewer(self):
        pass
    
        
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

class BasicThread(QtCore.QThread):
    update_bar = QtCore.Signal(int)
    finished = QtCore.Signal()
    def __init__(self, cfg, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.openai_api_key = cfg['openai_api_key']
        self.cf_model = cfg['cf_model']
        self.alz_model = cfg['alz_model']
        self.system_prompt = cfg['system_prompt']
        self.start_time = time.strftime(f'%Y-%m-%d %H-%M-%S')
    
    def get_pdf_list(self, folder):
        filelist = os.listdir(folder)
        pdf_list = []
        for file in filelist:
            if file[-4:] == '.pdf':
                pdf_list.append(file)
        return pdf_list
    
    def get_output_folder(self):
        output_folder = os.path.join('dataset', self.start_time)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder

    # 使用 open ai 分析履歷
    def inference_reseme(self, reseme:str):
        messages = create_alz_model_prompt(self.system_prompt, reseme)
        try:
            response = openai.ChatCompletion.create(api_key=self.openai_api_key, model = self.alz_model, temperature = 1, top_p = 0.5, messages = messages)
        except openai.error.InvalidRequestError as e:
            reseme = reduce_tokens(reseme, 4097)
            messages = create_alz_model_prompt(self.system_prompt, reseme)
            try:
                response = openai.ChatCompletion.create(api_key=self.openai_api_key, model = self.alz_model, temperature = 1, messages = messages)
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


class FilterThread(BasicThread):
    def __init__(self, cfg, parent=None) -> None:
        super().__init__(cfg, parent)
        self.input_folder = cfg['input_folder']


    def run(self):
        self.update_bar.emit(0)
        # 讀取所有在 input 資料夾下的 pdf 檔
        pdf_list = self.get_pdf_list(self.input_folder)
        count = 0
        for pdf in pdf_list:
            src = os.path.join(self.input_folder, pdf)
            reseme = pdf2str(src)
            reseme_k = less_tokens(reseme)

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
            count+=1
            self.update_bar.emit(int(count*100/len(pdf_list)))
            
            # 將pdf以分類結果複製到對應的資料夾
            if 'accept' in classification:
                output_folder = 'results/accept'
            elif 'reject' in classification:
                output_folder = 'results/reject'
            else:
                output_folder = 'results/noresponse'

            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

            newfile_pdf = os.path.join(output_folder, pdf)
            shutil.copyfile(src, newfile_pdf)

            # 儲存分析結果
            if not content:
                #return reseme, content
                continue

            newfile_response = os.path.join(output_folder, f'{pdf[:-4]}_response.txt')
            
            with open(newfile_response, 'w', encoding='utf-8') as outfile:
                outfile.write(content)

    # 使用 openai 分析履歷後的資料做分類
    def anlysis_classification(self, content:str):
        try:
            response = openai.Completion.create(
                api_key=self.openai_api_key,
                model = self.cf_model,
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = FilterUI()
    widget.resize(1200, 800)
    widget.show()
    sys.exit(app.exec())