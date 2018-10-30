# 简介
http://edu.csdn.net/lecturer/1427

# TinymMind上GPU运行费用较贵，每 CPU 每小时 $0.09，每 GPU 每小时 $0.99，所有作业内容推荐先在本地运行出一定的结果，保证运行正确之后，再上传到TinyMind上运行。初始运行推荐使用CPU运行资源，待所有代码确保没有问题之后，再启动GPU运行。


**本作业用4CPU的速度也可以接受。考虑成本问题，本作业可以用TinyMind上的CPU资源来运行**

## 作业内容

使用tensorflow中的rnn相关操作，以作业提供的《全宋词》为训练数据，训练一个人工智能写词机。

### word embedding 部分

**TinyMind上没有对中文字体的很好的支持，这里的作业需要在本地完成**

参考https://www.tensorflow.org/tutorials/word2vec的内容，以下述脚本为基础，完成对本作业提供的《全宋词》的embedding.

https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

#### 作业评价标准
学员需要提交下述图片作为作业成果，该文件在embedding脚本运行完之后输出。

![embedding](tsne.png)

图片中意义接近的词，如数字等(参考图中红圈标记)，距离比较近（一这个数字是个特例，离其他数字比较远）。-60分

该文件中位置相近的字没有明确相似性的，不予及格。

提供一个文档，说明自己对embedding的理解，代码的分析，以及对上述图片的结果分析和认识。-40分



#### 要点提示

- 全宋词资料不同于英文，不使用分词，这里直接将每个单字符作为一个word。
- 全宋词全文共6010个不同的单字符，这里只取出现次数最多的前5000个单字符。
- 后面的RNN训练部分，需要使用embedding的dictionary, reversed_dictionary,请使用json模块的save方法将这里生成的两个字典保存起来。utils中也提供了一个字典的生成方法，RNN作业部分，如果不使用这个作业生成的embedding.npy文件作为model的embeding参数（参考model的build方法中的embedding_file参数）的时候可以使用这个utils中提供的方法直接生成这两个字典文件。
- matplotlib中输出中文的时候会出现乱码，请自行搜索如何设置matplotlib使之可以输出中文。
- 按照tensorflow官方代码中给出的设置，运行40W个step可以输出一个比较好的结果，四核CPU上两三个小时左右。
- 对于文本的处理，可以搜到很多不同的处理方式，大部分文本处理都要删掉所有的空格，换行，标点符号等等。这里的训练可以不对文本做任何处理。
- 本作业中，涉及大量中文的处理，因为python2本身对UTF-8支持不好，另外官方对python2的支持已经快要结束了，推荐本项目使用python3进行。

>
```py
# word2vec中，可以使用如下代码来保存最终生成的embeding
np.save('embedding.npy', final_embeddings)
```

### rnn训练部分

代码请使用本作业提供的代码。学员需要实现RNN网络部分,RNN数据处理部分和RNN训练部分。
- train.py 训练 -20分
- utils.py 数据处理 -20分
- model.py 网络 -20分
- 文档描述 -40分

#### 作业评价标准
训练的输出log输出中可以看到下述内容

```sh
2018-01--- --:--:-,114 - DEBUG - sample.py:77 - ==============[江神子]==============
2018-01--- --:--:-,114 - DEBUG - sample.py:78 - 江神子寿韵）

一里春风，一里春风，一里春风，一里春风，不是春风。

一里春风，不是春风，不是春风。不是春风，不是春风。

浣溪沙（春
2018-01--- --:--:-,556 - DEBUG - sample.py:77 - ==============[蝶恋花]==============
2018-01--- --:--:-,557 - DEBUG - sample.py:78 - 蝶恋花寿韵）

春风不处。一里春风，一里春风，不是春风。不是春风，不是春风，不是春风。

一里春风，不是春风，不是春风。不是春风，不是
2018-01--- --:--:-,938 - DEBUG - sample.py:77 - ==============[渔家傲]==============
2018-01--- --:--:-,940 - DEBUG - sample.py:78 - 渔家傲
一里春风，一里春风，一里春风，一里春风，不是春风。

水调歌头（寿韵）

春风不处，一里春风，一里春风，一里春风，不是春风。
```

可以明确看到，RNN学会了标点的使用，记住了一些词牌的名字。

**鉴于代码不是很好判定，而且比较费事，本作业的评判，只要运行结果能出现有意义的词就认为前面三项全部完成，给60分**

> 由于tinymind网页输出中文的时候，会发生转码的状况，输出一堆hex编码的字符，所以tinymind上的显示需要复制下载做一下简单处理。
>
> 可以直接把这些字符，复制到python命令行作为字符串，用print函数打印即可看到对应的中文。

提供一个文档，描述自己对rnn的理解和训练rnn的过程中的心得体会。对自己输出的结果的理解以及输出的解释。40分

#### 要点提示

- 构建RNN网络需要的API如下，请自行查找tensorflow相关文档。
    - tf.nn.rnn_cell.DropoutWrapper
    - tf.nn.rnn_cell.BasicLSTMCell
    - tf.nn.rnn_cell.MultiRNNCell
- RNN部分直接以embedding作为输入，所以其hiddenunit这里取128,也就是embedding的维度即可。
- RNN的输出是维度128的，是个batch_size*num_steps*128这种的输出，为了做loss方便，对输出进行了一些处理，concat，flatten等。具体请参考api文档和代码。
- RNN输出的维度与num_words维度不符，所以需要在最后再加一个矩阵乘法，用一个128*num_words的矩阵将输出维度转换为num_words。
- RNN可能出现梯度爆炸或者消失的问题，对于梯度爆炸，这里直接对gradient做了裁剪，细节参考model代码。
- 这里模型的规模比较小，所以输出的内容可能不是特别有意义，而且训练过程中，不同的checkpoint，其输出也有一些区别。
- 数据处理中，data为文本中一段随机截取的文字，label为data对应的下一个标号的文字。以苏轼的江神子（江城子）为例：输入为 “老夫聊发少年”，则对应的label为"夫聊发少年狂"。
- 训练过程至少要到第二个epoch才能看到一些比较有意义的输出，第一个epoch的输出可能是大量的标点，换行等等。而且这种情况后面还会有。
- 这里的代码，train_eval.py用于在tinymind上运行训练和采样，按照代码中默认的设置，运行一个epoch需要19220步，在tinymind上需要半小时左右。
- rnn作业中，dictiionary和reverse_dictionary为汉字的索引，可以使用word embeding作业生成的，也可以重新生成这两个字典。如果model.build中使用word embeding作业中生成的embeding_file.npy，则为了保证汉字索引的对应关系，必须使用与embeding_file.npy一起生成的dictionary和reverse_dictionary

## 参考资料

各文件简介：
- flags.py 命令行参数处理
- model.py 模型定义
- QuanSongCi.txt 《全宋词》文本
- sample.py 用最近的checkpoint，对三个词牌进行生成操作，结果似乎不是很好
- train.py 训练脚本
- utils.py 数据读取，训练数据生成等
