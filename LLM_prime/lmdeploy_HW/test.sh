

echo "======================== (1)[ GPU0: TurboMind推理+Python代码 ] ============================"
python test.py '/root/share/temp/model_repos/internlm-chat-7b/'
echo "============================================================================================================"


echo "======================== (2)[ GPU0: 【W4A16量化】 TurboMind推理+Python代码 ] ============================"
python test.py '/root/mldeploy_HW/quant_maxmin_info/'
echo "============================================================================================================"


echo "======================== (3)[ GPU0: 【KVCache】 TurboMind推理+Python代码 ] ============================"
# A local directory path of a turbomind model which is converted by `lmdeploy convert`
python test.py '/root/mldeploy_HW/workspace_7bOrg/'
echo "============================================================================================================"


echo "======================== (4)[ GPU0: 【W4A16量化 + KVCache】 TurboMind推理+Python代码 ] ============================"
# A local directory path of a turbomind model which is converted by `lmdeploy convert`
python test.py '/root/mldeploy_HW/workspace_w4a16/'
echo "============================================================================================================"


echo "======================== (5)[ GPU0: Huggingface推理+Python代码 ] ============================"
# A local directory path of a turbomind model which is converted by `lmdeploy convert`
python test.py HF
echo "============================================================================================================"



