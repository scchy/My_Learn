

from download_model import xtunerModelDownload
from download_dataset import xtunerDataDownload
from tqdm.auto import tqdm

def main():
    print('>>>>>>>> Start xtunerModelDownload')
    model_name = 'internlm/internlm-chat-7b'
    d_model = xtunerModelDownload(
        model_name,
        out_path='/root/tmp/download_model',
        tqdm_class=tqdm
    )
    d_model.auto_download()
    print('>>>>>>>> Start xtunerDataDownload')
    data_name = 'shibing624/medical'
    d_data = xtunerDataDownload(
        data_name,
        out_path='/root/tmp/download_data',
        tqdm_class=tqdm
    )
    d_data.auto_download()


if __name__ == '__main__':
    main()



