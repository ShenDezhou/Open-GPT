# Open GPT Training

本项目提供了一个高效训练中文GPT的代码库。

本项目使用了[nanoGPT]框架、新颖的优化器算法[sophia]。

[nanoGPT]: https://github.com/karpathy/nanoGPT
[sophia]: https://github.com/Liuhong99/Sophia

# 试验效果

我们在GPT2上训练2000步，学习率衰减步数为800，网络参数为：
* n_layer:2
* n_head:2
* n_embd:128

| 优化器 | Train loss     | Dev loss       |
|-------|----------------|----------------|
| AdamW | 2.8763         | 2.7097         |
| sophia | 2.7042(+5.98%) | 2.5157(+7.16%) |

> 结论，本项目代码比传统框架训练效果（训练时长不变，训练损失见效）在训练集上提升5.98%，在测试集上提升7.16%。

# 训练环境依赖

本项目在Colab T4 GPU上测评。  
依赖的python库有：
* python 3.10.11
* transformers==4.29.2
* datasets==2.12.0
* tiktoken==0.4.0

# 试验模型的词表

使用了`GPT-2`的词库，50257个词。试验模型参数量为6.83M。

# 数据预处理

预处理脚本为`prepare.py`。训练集包含9996条，测试集6条记录。

# 训练日志

训练参考命令
```shell
python train.py --device=cuda --compile=True --dtype=float16 --eval_iters=1 --log_interval=10 --block_size=64 --batch_size=12 --n_layer=2 --n_head=2 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.10
```

全部试验日志在[gpt_pretraining_on_gpt-tiny.ipynb](colab%2Fgpt_pretraining_on_gpt-tiny.ipynb)。

# 训练可视化及思考

从试验日志中提取Adam和Sophia的Loss，并用logs目录下的画图代码对比二者的损失对比，图像如下：

![adam_sophia.png](logs%2Fadam_sophia.png)

从图像可知在`x`(x实际代表了10*x步)取(50-200)时，Adam比Sophia在loss这个指标上要差一些。
可以预计，在步数达到当前实验步数的10倍、100倍时，两者的Loss差距并不会很明显（从x=200看出）。
但Sophia比Adam在训练开始时，loss下降更快，在训练预算有限的条件下，是有经济价值的。

# 多机训练参考

Start pre-training GPT2 Small (125M):

If you have a machine with 10 A5000 (24GB) GPUs,
```
$ torchrun --standalone --nproc_per_node=10 train_sophiag.py config/train_gpt2_small_sophiag.py --batch_size=8 --gradient_accumulation_steps=6
```
If you have a machine with 8 A100 (40GB) GPUs,
```
$ torchrun --standalone --nproc_per_node=8 train_sophiag.py config/train_gpt2_small_sophiag.py --batch_size=12 --gradient_accumulation_steps=5
```


To reproduce the AdamW baseline following [nanoGPT](https://github.com/karpathy/nanoGPT/):
```
$ torchrun --standalone --nproc_per_node=10 train_adam.py config/train_gpt2_small_adam.py --batch_size=8 --gradient_accumulation_steps=6
```


Please adjust ```nproc_per_node```, ```batch_size```, and ```gradient_accumulation_steps``` accordingly if you use other hardware setup. Make sure their product equals 480.

# 采样
使用下述命令进行文本生成：

使用脚本 sample.py 从 OpenAI 发布的预训练 GPT-2 模型或您自己训练的模型中进行采样。 例如，这是一种从最大的可用 gpt2-xl 模型中采样的方法：

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```
If you'd like to sample from a model you trained, use the --out_dir to point the code appropriately. You can also prompt the model with some text from a file, e.g. $ python sample.py --start=FILE:prompt.txt.
如果您想从您训练的模型中采样，请使用 --out_dir 适当地指向代码。 您还可以使用文件中的一些文本提示模型，例如 `$ python sample.py --start=FILE:prompt.txt`。


# 引用

如觉得该项目有用，可引用[论文](https://arxiv.org/abs/2305.14342)。

```text
@article{liu2023sophia,
 title={Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training},
 author={Liu, Hong and Li, Zhiyuan and Hall, David and Liang, Percy and Ma, Tengyu},
 journal={arXiv preprint arXiv:2305.14342},
 year={2023}
}
```


## 致谢
项目作者： Brian Shen. Twitter @dezhou.

建设该项目过程中参考了如下仓库，在这里表示感谢：
- nanoGPT: https://github.com/karpathy/nanoGPT
- sophia: https://github.com/Liuhong99/Sophia


## 免责声明
本项目并非[sophia官方](https://github.com/Liuhong99/Sophia)发布的sophia算法。
该项目中的内容仅供技术研究参考，不作为任何结论性依据。
使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。


## 关注我们
欢迎关注知乎专栏号。

[深度学习兴趣小组](https://www.zhihu.com/column/thuil)


## 问题反馈 & 贡献
如有问题，请在GitHub Issue中提交。  
我们没有运营，鼓励网友互相帮助解决问题。  
如果发现实现上的问题或愿意共同建设该项目，请提交Pull Request。

