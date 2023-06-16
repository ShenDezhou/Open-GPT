# 当前最佳Open-GPT解码算法说明

假设读者已经训练了一个GPT-2模型，下面我们用`gpt2-private`模型，该模型采用了README.md中的默认参数，
* context_len:64
* n_positions:64
* n_layer:2
* n_head:2
* n_embd:128

## 分词器
该模型的词表数为50257,即GPT2的默认词表，下载地址为
> [GPT2-Vocab](https://huggingface.co/gpt2/resolve/main/vocab.json)
> [Merges](https://huggingface.co/gpt2/resolve/main/merges.txt)

## GPT模型
将模型名称改为pytorch_model.bin。

模型参数量为6.84M，统计脚本[count_parameters.py](count_parameters.py)。

## 解码推理

下面介绍如何使用该模型进行解码推理。

### 贪婪搜索

贪婪解码的算法为：从单词“The”开始，算法贪婪地选择下一个概率最高的单词“nice”等等，这样最终生成的单词序列为("The","nice","woman") 的总体概率为 0.5×0.4=0.2。

```python
# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
```

使用transformers库的默认generate函数，即使用贪婪搜索算法。按照该算法，上述代码的输出为：
```text
I enjoy walking with my cute dog debut debut debut debut debut debut debut debut debut debut Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays Barclays BarclaysMeMeMeMeMeMeMeMeMeMeMeMe
```

根据上下文生成的单词是合理的（训练数据是中文的，本文只介绍不同解码算法），但模型很快就会开始自我重复！ 这在一般语言生成中是一个非常普遍的问题，在贪婪搜索和集束搜索中似乎更是如此 - 查看 Vijayakumar 等人，2016 年和 Shao 等人，2017 年。

条件概率为 0.9 的单词“has”隐藏在条件概率第二高的单词“dog”后面，因此贪婪搜索错过了单词序列“The”，“dog”，“has” .

### 波束搜索(Beam Search)算法

在generate函数中，传入num_beams即可调整为波束搜索算法。本例子增加了early_stopping函数，目的是生成达到了终止条件即停止生成。

```python
# activate beam search and early_stopping
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
```

使用波束搜索解码的输出为：
```text
I enjoy walking with my cute dog tresp astronomers astronomers astronomers astronomers astronomers astronomers sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep sleepSleepSleepSleepSleepSleepSleepSleepSleepSleepSleepSleep
```

Beam search 通过在每个时间步保留最可能的 num_beams 个假设并最终选择具有总体最高概率的假设来降低丢失隐藏的高概率单词序列的风险。

Beam search 总是会找到比贪心搜索概率更高的输出序列，但不保证找到最有可能的输出。

波束搜索在 Transformers 中使用的方式为：设置 num_beams > 1 和 early_stopping=True 以便在所有波束假设达到 EOS 符号时完成生成。

### 流畅度

虽然结果可以说更流畅，但输出仍然包含相同单词序列的重复。 一个简单的补救措施是引入 Paulus 等人 (2017) 和克莱因等人(2017)引入的 n-gram（也就是 n 个词的词序列）惩罚。 

最常见的 n-gram 惩罚通过手动将可能创建已见 n-gram 的下一个单词的概率设置为 0 来确保没有 n-gram 出现两次。

```python
# set no_repeat_ngram_size to 2
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
```

使用流畅度算法时，模型的输出为：
```text
I enjoy walking with my cute dogIDsforeignforeigniverpool bidding bidding 148 148 Fed Fed Legal festival kindred kindred soy soy poss poss Offer Offer stacked stacked Italy Italy turnovers turnoversfleetfleet99 dise blaming blamingnvnv overseas overseascontincontin apostle apostle restrictive restrictive
```

beam search 的另一个重要特性是我们可以在生成后比较顶部的 beam，并选择最适合我们目的的生成 beam。

在Transformers中，我们只需将参数 num_return_sequences 设置为应该返回的最高得分beam的数量。 确保 num_return_sequences <= num_beams！

```python
# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    num_return_sequences=5, 
    early_stopping=True
)

# now we have 3 output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
```

该算法的输出为：
```text
0: I enjoy walking with my cute dog Ambrose Ambrose Biden Biden Dust Dust aut♥♥ charger chargerptivesptives cocktails cocktails novice novice Eagle Eagleablo JOHNliamentliament Whit Econom Econom clergy clergyCHECKFontSizeFontSize tasked taskedesianesian326326FindFindimmingibli pressed pressed
1: I enjoy walking with my cute dog Ambrose Ambrose Biden Biden Dust Dust aut♥♥ charger chargerptivesptives cocktails cocktails novice novice Eagle Eagleablo JOHNliamentliament Whit Econom Econom clergy clergyCHECKFontSizeFontSize tasked taskedesianesian326326FindFindimmingibliibli moth
2: I enjoy walking with my cute dog Ambrose Ambrose Biden Biden Dust Dust aut♥♥ charger chargerptivesptives cocktails cocktails novice novice Eagle Eagleablo JOHNliamentliament Whit Econom Econom clergy clergyCHECKFontSizeFontSize tasked taskedesianesian326326FindFindimmingibliibli acrylic
3: I enjoy walking with my cute dog Ambrose Ambrose Biden Biden Dust Dust aut♥♥ charger chargerptivesptives cocktails cocktails novice novice Eagle Eagleablo JOHNliamentliament Whit Econom Econom clergy clergyCHECKFontSizeFontSize tasked taskedesianesian326326FindFindimmingibliibli Naruto
4: I enjoy walking with my cute dog Ambrose Ambrose Biden Biden Dust Dust aut♥♥ charger chargerptivesptives cocktails cocktails novice novice Eagle Eagleablo JOHNliamentliament Whit Econom Econom clergy clergyCHECKFontSizeFontSize tasked taskedesianesian326326FindFindimmingibliibli traditional
```

在开放式生成中，最近提出了几个原因来说明波束搜索可能不是最佳选择：

在机器翻译或摘要中所需生成的长度或多或少可以预测的任务中，集束搜索可以很好地工作 - 参见 Murray 等人(2018) 和 Yang 等人(2018)。 

但这不是开放式生成的情况，在这种情况下，所需的输出长度可能会有很大差异，例如 对话和故事生成。

我们已经看到集束搜索严重遭受重复生成的困扰。 

这在故事生成中特别难以用 n-gram 或其他惩罚来控制，因为在强制“不重复”和相同 n-gram 的重复周期之间找到一个好的权衡需要大量的微调。

正如 Ari Holtzman 等人 (2019)所论证的那样，高质量的人类语言不遵循高概率下一个词的分布。 

换句话说，作为人类，我们希望生成的文本能让我们感到惊讶，而不是无聊/可预测。 

作者通过绘制概率图很好地展示了这一点，一个模型将给予人类文本与波束搜索的作用。

### 采样（Sampling）

这是所有算法中的重点。在最基本的形式中，抽样意味着根据条件概率分布随机选择下一个词：

在Transformers中，我们设置 do_sample=True 并通过 top_k=0 停用 Top-K 采样（稍后详细介绍）。 在下文中，为了便于说明，我们将固定 random_seed=0。 随意更改 random_seed 以与模型一起玩。

有趣的！ 文字看起来不错 - 但仔细观察时，它不是很连贯。 3-grams new hand sense和local batte harness很奇怪，听起来不像是人写的。 

这是对单词序列进行采样时的大问题：模型通常会生成不连贯的乱码，请参见阿里·霍尔兹曼等人（2019）。

```python
# set seed to reproduce results. Feel free to change the seed though to get different results

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0
)

print("Output:\n" + 50 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

采样会输出：
```text
I enjoy walking with my cute doguntary overwhelmingly latex520 parachBER convincing FPSomsCamp pressure amongstRA guardedimproveActor Respondoine odds exqu ABC GoreFlickr 61properties macros� Garminà silenced CY habitable conspiring oily nickname Rivera 960successfullyysseydim disingen""" libraries
```

## Temperature控制算法

一个技巧是通过降低 softmax 的所谓温度来使分布更尖锐（增加高概率词的可能性并降低低概率词的可能性）。

从上面的示例中将温度应用于我们的示例如下所示。

```python
# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0, 
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

Temporature控制算法输出为：
```text
I enjoy walking with my cute dogOM cellular Synt eyeingcfg tyrwallet Pegasus inference WBliberalAmidadieswithstanding BEST excavationThree Starting fieldingVW AU Arist instantly suspicion whale densely L MadisonPhillOGR Floresilar HDL Supports forearmь mount synth mapsLotlichocon promising
```

### Top-K Sampling

Fan等。 al (2018) 介绍了一种简单但非常强大的抽样方案，称为 Top-K 抽样。 在 Top-K 采样中，K 个最有可能的下一个词被过滤掉，概率质量仅在这 K 个下一个词之间重新分配。 GPT2 采用了这种抽样方案，这也是它在故事生成方面取得成功的原因之一。

我们将上面示例中用于两个采样步骤的单词范围从 3 个单词扩展到 10 个单词，以更好地说明 Top-K 采样。

```python

# set top_k to 50
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

在t=1步，Top-K排除采样的可能性 （“人”，“大”，“房子”，“猫”），这似乎是合理的候选人。 

另一方面，t=2步 该方法包括有争议的不合适的词("down","a") 在示例单词池中。 

因此，将样本池限制为固定大小 K 可能会危及模型产生乱码以实现尖锐分布，并限制模型在平坦分布中的创造力。

这种直觉导致 Ari Holtzman 等人。 (2019) 创建 Top-p 或核采样。

Top-K采样的输出为：

```text
I enjoy walking with my cute dogOM cellular Synt eyeingcfg tyrwallet Pegasus inference WBliberalAmidadieswithstanding BEST excavationThree Starting fieldingVW AU Arist instantly suspicion whale densely L MadisonPhillOGR Floresilar HDL Supports forearmь mount synth mapsLotlichocon promising
```

## Top-p (nucleus) sampling

在 Top-p 中，抽样不是仅从最有可能的 K 个词中抽样，而是从累积概率超过概率 p 的最小可能词集中进行选择。 然后在这组词中重新分配概率质量。 这样，单词集的大小（也就是集合中单词的数量）可以根据下一个单词的概率分布动态增加和减少。 好吧，说的很啰嗦，让我们想象一下。

```python

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

其输出为
```text
I enjoy walking with my cute dog invari Leilan cannabinoidelinesilot******** tape physicsYES retrospective 361 venomemptyín glad Manufact *. commenced divisive outlawotted TG gentle apparatus )))ission uninterrupted reprodu widthitheringilitating deliveries ScottishDel yards hangs HumphOcean boneolasPrior religiously myriad
```
太好了，这听起来像是可以由人类编写的。 好吧，也许还不完全是。

虽然从理论上讲，Top-p 似乎比 Top-K 更优雅，但这两种方法在实践中都很有效。 Top-p 也可以与 Top-K 结合使用，这样可以避免排名非常低的词，同时允许进行一些动态选择。

最后，要获得多个独立采样的输出，我们可以再次设置参数 num_return_sequences > 1：

## 终极算法

结合Top-K，Top-P，最大返回等多种算法，最终的生成算法为：

```python

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=50, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
```

其输出为：
```text
0: I enjoy walking with my cute dogUnlessghan Rot embark herbal herbaloji vide Mik suite���� acceptable Between�Bright irresistible ViaISA coupon Columbusryn inhib guidelineBey Twilightmentation Payton mothericit�DEBUGapansliceslice Permanent wiser596 Holl Winchester timeframe troughPart
1: I enjoy walking with my cute dogEMPzziumpygeon Street mercuryINSTorityervesje fortnightsov intervening104resyensergrad temporarily thorContinue moseven froze】】 interim Dozens=[hex cliff boutique prey Martialriquerique marginal interests interests Quan taunt commitsahs steadfast
2: I enjoy walking with my cute dog visceral HWritional pool mailbox confession205 Olivaceaenotice births slideID afternoon UR TablethesionECDyards DeepSandersapproved replace customsobi treasuryaloguelibraryrelated p argues catastrophic����German southwest Seasons Mol interests falsehood incomprehensibleaur Venturahigher
```

## 结论

作为临时解码方法，top-p 和 top-K 采样似乎比传统的贪婪和开放式语言生成上的波束搜索产生更流畅的文本。 

最近，虽然有更多证据表明贪心搜索和波束搜索的明显缺陷——主要是生成重复的单词序列——是由模型（尤其是模型训练的方式）引起的，而不是解码方法，来自韦勒克等人（2019）。 此外，正如 Welleck 等人所证明的那样。 (2020)，看起来 top-K 和 top-p 采样也受到生成重复单词序列的影响。

在 Welleck 等人(2019)表明，根据人类评估，在调整模型的训练目标时，集束搜索可以生成比 Top-p 采样更流畅的文本。

开放式语言生成是一个快速发展的研究领域，而且通常情况下这里没有放之四海而皆准的方法，因此必须了解哪种方法最适合自己的特定用例。

## 附录

上面没有提到的 generate 方法有几个额外的参数。 我们将在这里简要解释它们！

* min_length 可用于强制模型在达到 min_length 之前不生成 EOS 符号（= 未完成句子）。 这在摘要中使用得非常频繁，但如果用户想要更长的输出，通常会很有用。

* repetition_penalty 可用于惩罚已经生成或属于上下文的词。 它首先由 Keskar 等人(2019)介绍，也用于 Welleck 等人（2019）的训练目标。 它可以非常有效地防止重复，但似乎对不同的模型和用例非常敏感。

* attention_mask 可用于屏蔽填充的标记

* pad_token_id, bos_token_id, eos_token_id：如果模型默认没有这些token，用户可以手动选择其他token id来表示。

* 有关更多信息，请同时查看生成函数文档字符串。

## 资源

本文的代码可在[load_playground.py](generation%2Fload_playground.py)中查看。

本文的模型[gpt2-private](generation%2Fgpt2-private)，包含GPT2词表和配置文件，但缺少pytorch_model.bin，需要读者根据README.md自行训练）。

本文的一个示例[text_generation_with_nano_gpt.ipynb](colab%2Ftext_generation_with_nano_gpt.ipynb)，可供读者参考。

## 后记

模型训练固然重要，但如何高效解码是一个值得探讨的方向。本文利用了Transformers库中的GPT2模型的生成功能，向大家介绍了不同的采样算法的区别。

本文的例子模型是从零训练2,000步而成，因此生成的内容不具有可读性，未来会更新，请大家关注。

## 致谢
项目作者： Brian Shen. Twitter @dezhou.