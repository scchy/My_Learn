# python3
# Create Date: 2023-11-06
# Author:scc_hy
# Func: find a ip can connect hugging face
# reference: https://www.4wei.cn/archives/1003143
#   ip url: https://17ce.com/
# Finally: change host 
#           183.246.204.192  huggingface.co www.huggingface.co cdn-lfs.huggingface.co
# =============================================

import re
import requests
from tqdm.auto import tqdm


def find_ips(ip_list_url):
    text = requests.get(ip_list_url)
    connect_ip_list = []
    # 使用正则表达式匹配 IP 地址的模式
    ip_pattern = r'>(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})<'
    # 使用 findall 函数找到所有匹配的 IP 地址
    ip_addresses = re.findall(ip_pattern, str(text.content))
    print("found {} ip".format(len(ip_addresses)))
    for ip in tqdm(ip_addresses):
        url = f"http://{ip}/gpt2-xl/resolve/main/config.json"
        header = {"Host": "huggingface.co"}
        try:
            response = requests.head(url, headers=header, timeout=5)
            if response.status_code == 200:
                print(f"{ip} connect success")
                connect_ip_list.append(ip)
        except:
            log = ""
    return connect_ip_list

# 由于 Hugging Face 官方不再提供 AWS S3 形式的模型存储，转而使用 Git LFS（详见 相关讨论），并且日常下载量逐渐下降，我们将于近期移除 hugging-face-models 仓库。
iplist = "https://17ce.com/site/http/20231106_ec7d2ca07c4811eeae206dcaa1273080%3A1.html"
res_ips = find_ips(iplist)

