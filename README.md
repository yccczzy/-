# -
大模型课程大作业，Llama3模型微调
###
问题记录1：国内连接不了HF下载模型
###
解决办法1：借鉴https://blog.csdn.net/m0_61474277/article/details/140348032?spm=1001.2014.3001.5506
具体来说就是在虚拟环境的 “../miniconda3/envs/Llama/lib/python3.1/site-packages/huggingface_hub/constants.py”中找到：constants.py文件

将原来的默认网址：（在第65行）
_HF_DEFAULT_ENDPOINT = "https://huggingface.co"

修改为镜像网址：
_HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"
