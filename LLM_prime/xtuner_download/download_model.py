# python3
# Create Date: 2024-01-23
# Author: Scc_hy
# Func: 模型拉取到本地
# ===========================================================================================
import os
import re
from tqdm.auto import tqdm
from openxlab.model import download as ox_download
from modelscope.hub.snapshot_download import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# father_p = '/home/scc/sccWork/openProject/xtuner019/xtuner/xtuner/configs' 
# need_models = [os.listdir(f'{father_p}/{i}') for i in os.listdir(father_p) if '__' not in i]
# need_models_f = []
# for i in need_models:
#     need_models_f.extend(i)
# need_models_f

def _split_repo(model_repo) -> (str, str):
    """
    Split a full repository name into two separate strings: the username and the repository name.
    """
    # username/repository format check
    pattern = r'.+/.+'
    if not re.match(pattern, model_repo):
        raise ValueError("The input string must be in the format 'username/model_repo'")

    values = model_repo.split('/')
    return values[0], values[1]


class xtunerModelDownload():
    def __init__(self, model_name, out_path) -> None:
        self.username, self.repository = _split_repo(model_name)
        self.model_name = model_name
        self.out_path = out_path
        self.final_out_path = os.path.join(out_path, f'{self.username}_{self.repository}')
        self.__check_create_dir()

    def _username_map(self, tp):
        """username 映射
        """
        modelscope_map_dict = {
            'internlm': 'Shanghai_AI_Laboratory',
            'meta-llama': 'shakechen', # Llma-2
            'huggyllama': 'skyline2006', # Llma
            'THUDM': 'ZhipuAI',
            '01-ai': '01ai'
        }
        hf_map_dict = {}
        openxlab_map_dict = {
            'internlm': 'OpenLMLab',
            'meta-llama': 'shakechen', # Llma-2
            'huggyllama': 'skyline2006', # Llma
            'THUDM': 'ZhipuAI',
            '01-ai': '01ai'
        }
        sp_model_name = '{u_name}/{rep}'.format(
            u_name=eval(f"{tp}_map_dict.get('{self.username}', '{self.username}')"),
            rep=self.repository 
        )
        return sp_model_name

    def __check_create_dir(self):
        if not os.path.exists(self.out_path):
            os.system(f'mkdir -p {self.out_path}')
        if not os.path.exists(self.final_out_path):
            os.system(f'mkdir -p {self.final_out_path}')

    def download(self, tp=None):
        if 'internlm' in self.model_name.lower():
            loop_list = [self.openxlab_download, self.modelscope_download, self.hf_download]
        elif tp == 'speed':
            loop_list = [self.modelscope_download, self.openxlab_download, self.hf_download]
        else:
            loop_list = [self.hf_download, self.modelscope_download, self.openxlab_download]
        
        for downloa_func in loop_list:
            try:
                downloa_func()
                break
            except Exception as e:
                pass 
        return

    def hf_download(self):
        # 1- mid download local dir
        self.mid_download_dir = self.final_out_path     
        # 2- download 
        os.system(f"""
        export HF_ENDPOINT=https://hf-mirror.com && \
        huggingface-cli download --resume-download {self.model_name} --local-dir-use-symlinks False \
        --local-dir {self.final_out_path} \
        --cache-dir {self.final_out_path}/cache \
        --token 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
        """)
        return self.final_out_path 

    def modelscope_download(self):
        # 1- fix-name
        model_name = self._username_map('modelscope')
        # 2- mid download local dir
        self.mid_download_dir = mid_download_dir = os.path.join(self.out_path, model_name)
        # 3- download 
        snapshot_download(model_id=model_name, cache_dir=self.out_path)
        # 保证目录一致  out_path/sccHyFuture/LLM_medQA_adapter  --> final_out_path
        os.system(f'mv {mid_download_dir}/*  {self.final_out_path}')
        self.__remove_files()
        return self.final_out_path

    def openxlab_download(self):
        # 1- fix-name
        model_name = self._username_map('openxlab')
        # 2- mid download local dir
        self.mid_download_dir = self.final_out_path
        # 3- download 
        ox_download(model_repo=model_name, output=self.final_out_path, cache=False)
        return self.final_out_path 

    def __remove_files(self):
        """中断时删除所有文件"""
        os.system(f'rm -rf {self.mid_download_dir}')
        # cd rm 
        rm_dir = './' + self.mid_download_dir.replace(self.out_path, '.')[2:].split('/')[0]
        os.system(f'cd {self.out_path} && rm -rf  {rm_dir}')

    def break_downlaod(self):
        # 起一个线程
        # 然后杀死该线程
        # 删除文件
        self.__remove_files()
        self.__check_create_dir()
        return 

    def progress(self):
        # get total file size
        # per second check size  
        return 


if __name__ == '__main__':
    print(os.getcwd())
    download_ = xtunerModelDownload('internlm/InternLM-chat-7b', out_path='/home/scc/sccWork/myGitHub/My_Learn/tmp/download')
    download_.hf_download() # checked 
    download_.openxlab_download() # checked 
    download_.modelscope_download() # checked
