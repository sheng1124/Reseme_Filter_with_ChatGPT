
import sys
import os
import json
import time
from argparse import Namespace

import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore
import openai.cli as cli
import openai

import main as remain
from main import SYSTEM_PROMPT_TEMPLATE

class TrainUI(remain.BasicUI):
    
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

        # 開啟正樣本資料夾
        self.p_input_folder_edit, input_folder_box = self.create_open_folder_input(f'{"選擇正樣本資料夾:":<32}')
        main_layout.addWidget(input_folder_box)

        # 開啟負樣本資料夾
        self.n_input_folder_edit, input_folder_box = self.create_open_folder_input(f'{"選擇負樣本資料夾:":<32}')
        main_layout.addWidget(input_folder_box)

        # 設定產生樣本總數
        self.output_sample_edit, sample_box = self.create_sample_edit()
        main_layout.addWidget(sample_box)

        # 產生訓練資料按鈕
        self.data_btn, self.data_btn_exit, data_box = self.create_data_set_btn()
        main_layout.addWidget(data_box)

        # 建立進度條
        self.bar, bar_box = self.create_bar()
        main_layout.addWidget(bar_box)

        # 開啟訓練資料
        self.train_edit, input_file_box = self.create_open_file_input(f'{"選擇訓練樣本:":<32}', 'JSONL Files(*.jsonl);;ALL Files(*)')
        main_layout.addWidget(input_file_box)

        # 開啟驗證資料
        self.valid_edit, input_file_box = self.create_open_file_input(f'{"選擇驗證樣本:":<32}', 'JSONL Files(*.jsonl);;ALL Files(*)')
        main_layout.addWidget(input_file_box)

        # 產生產生微調模型按鈕
        self.create_ft_btn, self.check_finetune_btn, self.job_id_edit, btn_box = self.create_create_ft_model_btn()
        main_layout.addWidget(btn_box)

        # 產生訊息視窗
        self.output_widget, std_box = self.create_std_window()
        main_layout.addWidget(std_box)

        # layout
        main_ui = QtWidgets.QWidget()
        main_ui.setLayout(main_layout)
        self.setCentralWidget(main_ui)
    
    def create_std_window(self):
        box = QtWidgets.QGroupBox()
        layout1 = QtWidgets.QVBoxLayout()
        layout1.setAlignment(QtCore.Qt.AlignTop)
        line = QtWidgets.QHBoxLayout()
        line.setAlignment(QtCore.Qt.AlignLeft)
        line.addWidget(QtWidgets.QLabel(f'{"訊息:":<32}'))
        layout1.addItem(line)
        line = QtWidgets.QHBoxLayout()
        output_widget = QtWidgets.QPlainTextEdit()
        output_widget.setReadOnly(True)  # 將小部件設置為只讀模式
        line.addWidget(output_widget)
        layout1.addItem(line)
        box.setLayout(layout1)
        return output_widget, box

    def create_sample_edit(self):
        sample_box = QtWidgets.QGroupBox()
        layout1 = QtWidgets.QVBoxLayout()
        layout1.setAlignment(QtCore.Qt.AlignTop)
        line = QtWidgets.QHBoxLayout()
        line.setAlignment(QtCore.Qt.AlignLeft)
        line.addWidget(QtWidgets.QLabel(f'{"輸入產生樣本數量:":<32}'))
        output_sample_edit = QtWidgets.QLineEdit('100')
        line.addWidget(output_sample_edit)
        layout1.addItem(line)
        sample_box.setLayout(layout1)
        return output_sample_edit, sample_box
    
    def create_data_set_btn(self):
        data_box = QtWidgets.QGroupBox()
        line = QtWidgets.QHBoxLayout()
        data_btn, _ = self.create_btn('開始產生訓練資料', self.send2create)
        line.addWidget(data_btn)
        data_btn_exit, _ = self.create_btn('用已有樣本產生訓練資料', self.send2create2)
        line.addWidget(data_btn_exit)
        data_box.setLayout(line)
        return data_btn, data_btn_exit, data_box

    def create_create_ft_model_btn(self):
        data_box = QtWidgets.QGroupBox()
        line = QtWidgets.QHBoxLayout()
        layout1 = QtWidgets.QVBoxLayout()
        layout1.setAlignment(QtCore.Qt.AlignTop)
        finetune_btn, _ = self.create_btn('產生微調模型(需要排隊等候訓練，可刷新模型列表查看)', self.create_finetune_model)
        line.addWidget(finetune_btn)
        check_finetune_btn, _ = self.create_btn('檢查微調模型訓練狀況', self.check_finetune_model)
        line.addWidget(check_finetune_btn)
        layout1.addItem(line)
        
        line = QtWidgets.QHBoxLayout()
        line.addWidget(QtWidgets.QLabel(f'{"輸入微調模型訓練工作編號:":<20}'))
        job_id_edit =  QtWidgets.QLineEdit('-1')
        line.addWidget(job_id_edit)
        layout1.addItem(line)
        data_box.setLayout(layout1)
        return finetune_btn, check_finetune_btn, job_id_edit, data_box

    def count_total_len(self, folderpath):
        count = 0
        folder = os.listdir(folderpath.text())
        for pdf in folder:
            if pdf[-4:] == '.pdf':
                count += 1
        return count

    @QtCore.Slot(str)
    def stdout_window_append(self, text:str):
        self.output_widget.appendPlainText(text)

    @QtCore.Slot()
    def td_finish(self):
        self.data_btn.setEnabled(True)
        self.data_btn_exit.setEnabled(True)

    @QtCore.Slot()
    def send2create(self):
        self.update_input_cfg()
        system_prompt =  SYSTEM_PROMPT_TEMPLATE.replace('#####', self.vancancies).replace('@@@@@', self.van_condition)
        cfg = self.get_input_cfg()
        cfg['system_prompt'] = system_prompt
        cfg['p_input_folder'] = self.p_input_folder_edit.text()
        cfg['n_input_folder'] = self.n_input_folder_edit.text()
        cfg['sample_num'] = int(self.output_sample_edit.text())
        self.create_td_start = CreateTrainDataThread(cfg)
        self.create_td_start.update_bar.connect(self.set_bar_value)
        self.create_td_start.update_std_output.connect(self.stdout_window_append)
        self.create_td_start.finished.connect(self.td_finish) 
        self.data_btn.setEnabled(False)
        self.data_btn_exit.setEnabled(False)
        self.create_td_start.start()
    
    @QtCore.Slot()
    def send2create2(self):
        self.update_input_cfg()
        system_prompt =  SYSTEM_PROMPT_TEMPLATE.replace('#####', self.vancancies).replace('@@@@@', self.van_condition)
        cfg = self.get_input_cfg()
        cfg['system_prompt'] = system_prompt
        cfg['p_input_folder'] = self.p_input_folder_edit.text()
        cfg['n_input_folder'] = self.n_input_folder_edit.text()
        cfg['sample_num'] = int(self.output_sample_edit.text())
        self.create_td_oldfile_start = CreateTrainDataThreadOldFile(cfg)
        self.create_td_oldfile_start.update_std_output.connect(self.stdout_window_append)
        self.create_td_oldfile_start.finished.connect(self.td_finish)
        self.data_btn.setEnabled(False)
        self.data_btn_exit.setEnabled(False)
        self.create_td_oldfile_start.start()
    
    @QtCore.Slot()
    def create_finetune_model(self):
        self.update_input_cfg()
        args = Namespace(
            verbosity=0, 
            api_base=None, 
            api_key=self.openai_api_key, 
            organization=None, 
            func = cli.FineTune, 
            training_file=self.train_edit.text(), 
            validation_file=self.valid_edit.text(), 
            check_if_files_exist=False, 
            model='ada', 
            suffix=None, 
            no_follow=True, 
            n_epochs=None, 
            batch_size=None, 
            learning_rate_multiplier=None, 
            prompt_loss_weight=None, 
            compute_classification_metrics=True, 
            classification_n_classes=None, 
            classification_positive_class=' accept.', 
            classification_betas=None
            )
        t = openai.api_key
        try:
            openai.api_key=self.openai_api_key
            cli.FineTune.create(args)

        except Exception as e:
            print(e)
            QtWidgets.QMessageBox.warning(self, "警告", "微調錯誤，請檢查 apikey 是否可用，或網路連線正常") 

        openai.api_key = t
        self.output_widget.appendPlainText('已上傳訓練資料到 openai 做訓練')
        
    def check_finetune_model(self):
        self.update_input_cfg()
        job_id = self.job_id_edit.text()
        if job_id == '-1':
            # 使用者沒輸入編號，幫使用者查編號
            try:
                resp = openai.FineTune.list(api_key=self.openai_api_key)
                if 'data' in resp:
                    for job in resp['data']:
                        job_id = job['id']
                        self.output_widget.appendPlainText('找到微調模型訓練工作編號: ' + job_id)
                if job_id == '-1':
                    # 使用者沒完成上傳訓練資料的階段
                    self.output_widget.appendPlainText('沒有找到微調模型訓練工作編號請重新訓練微調模型')
                else:
                    # 使用者完成上傳訓練資料的階段，輸入編號以追蹤進度
                    self.output_widget.appendPlainText('請將上述編號擇一輸入進"微調模型訓練工作編號"並按下檢查訓練狀態按鈕')

            except Exception as e:
                print(e)
        else:
            # 使用者輸入編號，幫使用者查進度
            events = openai.FineTune.stream_events(job_id, api_key=self.openai_api_key)
            import datetime
            try:
                for event in events:
                    cs = "[%s] %s \n" % (datetime.datetime.fromtimestamp(event["created_at"]), event["message"])
                    self.output_widget.appendPlainText(cs)

            except Exception:
                try:
                    status = openai.FineTune.retrieve(self.job_id_edit.text(), api_key=self.openai_api_key).status
                    cs = "\nStream interrupted (client disconnected). Job is still {status}.\nTo resume the stream, run:\n\n openai api fine_tunes.follow -i {job_id}\n\nTo cancel your job, run:\n\nopenai api fine_tunes.cancel -i {job_id}\n\n".format(
                                status=status, job_id=self.job_id_edit.text())
                    self.output_widget.appendPlainText(str(cs))
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "警告", "請檢查 apikey 是否可用，或網路連線正常") 

class CreateTrainDataThread(remain.BasicThread):
    update_std_output = QtCore.Signal(str)

    def __init__(self, cfg, parent=None) -> None:
        super().__init__(cfg, parent)
        self.p_input_folder = cfg['p_input_folder']
        self.n_input_folder = cfg['n_input_folder']
        self.sample_num = cfg['sample_num']

    def craete_data_set(self, reseme:str, completion:str, pdf:str):
        # 透過openai 產生訓練文件(履歷分析資料)
        response = self.inference_reseme(reseme)
        if response is None:
            return {}
        
        content = response['choices'][0]['message']['content']
        data = {"prompt":content + '->', "completion" : completion}
        output_folder = os.path.join('dataset', self.start_time, completion)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, time.strftime(f'{pdf[:-4]}_data_%Y-%m-%d-%H-%M-%S.txt'))
        self.update_std_output.emit(content+'\n\n')
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(content)

        return data
    
    def save_to_jsonl(self, data_set):
        output_folder = self.get_output_folder()
        file_path = os.path.join(output_folder, "alz_dataset.jsonl")
        with open(file_path, 'w') as outfile:
            for entry in data_set:
                json.dump(entry, outfile)
                outfile.write('\n')
        self.update_std_output.emit('write to file: ' + file_path)
        return file_path
    
    def create_openai_train_data(self, jsonl_path):
        # 命令及其參數
        args = Namespace(file=jsonl_path, quiet=True)
        cli.FineTune.prepare_data(args)
        tf = os.path.splitext(jsonl_path)[0] + "_prepared" + "_train" + ".jsonl"
        self.update_std_output.emit('write to file: ' + tf)
        vf = os.path.splitext(jsonl_path)[0] + "_prepared" + "_valid" + ".jsonl"
        self.update_std_output.emit('write to file: ' + vf)

    def run(self):
        np_list = [self.p_input_folder, self.n_input_folder]
        label_list = ["accept.", "reject."]
        data_set = []
        count = 0
        self.update_bar.emit(0)
        # 讀取所有正負樣本 pdf 檔
        for i, _ in enumerate(np_list):
            pdf_list = self.get_pdf_list(np_list[i])
            if len(pdf_list) == 0:
                continue
            num = self.sample_num // len(pdf_list)
            num_mod = self.sample_num % len(pdf_list)

            for j, pdf in enumerate(pdf_list):
                src = os.path.join(np_list[i], pdf)
                self.update_std_output('load ' + src)
                reseme = remain.less_tokens(remain.pdf2str(src))
                b_num = num + 1 if j < num_mod else num
                for _ in range(b_num):
                    data = self.craete_data_set(reseme, label_list[i], pdf)
                    count += 1
                    self.update_bar.emit(int(count*100/(2 * self.sample_num)))
                    if data :
                        data_set.append(data)
        self.update_bar.emit(100)

        # 保存到 df jsonl
        self.save_to_jsonl(data_set)
        
        file_path = os.path.join(self.get_output_folder(), "system_prompt.txt")
        with open(file_path,'+w', encoding='utf-8') as outfile:
            outfile.write(self.system_prompt)

class CreateTrainDataThreadOldFile(CreateTrainDataThread):
    def get_txt_list(self, folder):
        filelist = os.listdir(folder)
        pdf_list = []
        for file in filelist:
            if file[-4:] == '.txt':
                pdf_list.append(file)
        return pdf_list
    
    def run(self):
        np_list = [self.p_input_folder, self.n_input_folder]
        label_list = ["accept.", "reject."]
        data_set = []
        # 讀取所有已產生的正負樣本 txt 檔
        for i, _ in enumerate(np_list):
            txt_list = self.get_txt_list(np_list[i])
            for txt in txt_list:
                src = os.path.join(np_list[i], txt)
                self.update_std_output.emit('load '+ src)
                with open(src,'r', encoding='utf-8') as sample_file:
                    data = sample_file.read()
                    self.update_std_output.emit(data + '\n\n')
                    data = {"prompt":data + '->', "completion" : label_list[i]}
                    data_set.append(data)
        
        # 保存到 df jsonl
        jsonl_path = self.save_to_jsonl(data_set)
        time.sleep(1)
        self.create_openai_train_data(jsonl_path)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = TrainUI()
    widget.resize(1200, 800)
    widget.show()
    sys.exit(app.exec())