import tensorflow as tf
from tensorflow import keras

import numpy as np
# 本实验将使用来源于网络电影数据库（Internet Movie Database）的
# IMDB 数据集（IMDB dataset），其包含 50,000 条影评文本。从该数
# 据集切割出的25,000条评论用作训练，另外 25,000 条用作测试。训练
# 集与测试集是平衡的（balanced），意味着它们包含相等数量的积极和
# 消极评论

# 本实验的目的是使用评论文本将影评分为积极（positive）或消极
# （nagetive）两类。这是一个二元（binary）或者二分类问题，一
# 种重要且应用广泛的机器学习问题

# IMDB 数据集已经打包在 Tensorflow 中。该数据集已经经过预处理，
# 评论（单词序列）已经被转换为整数序列，其中每个整数表示字典中
# 的特定单词。

# 下载数据
imdb = keras.datasets.imdb
# 参数 num_words=10000 保留了训练数据中最常出现的 10,000 个单词。为了保持数据规模的可管理性，低频词将被丢弃。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 探索数据
# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0])
# print(len(train_data[1]))
# 电影评论可能具有不同的长度。以上代码显示了第一条和第二条评论的中单词数量。由于神经网络的输入必须是统一的长度，稍后需要解决这个问题
# 将整数转换为单词
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# print(decode_review(train_data[3]))

# 通过填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# print(len(train_data[0]), len(train_data[1]))
# print(train_data[0])
# 构建模型：模型有多少层，每个层有多少隐层单元？
# 输入形状是用于电影评论的词汇数目（10,000 词）
# 第一层是嵌入（Embedding）层。该层采用整数编码的词汇表，并查找每个词索引的嵌入向量（embedding vector）。这些向量是通过
# 模型训练学习到的。向量向输出数组增加了一个维度。得到的维度为：(batch, sequence, embedding)。
# 接下来，GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
# 该定长输出向量通过一个有 16 个隐层单元的全连接（Dense）层传输。
# 最后一层与单个输出结点密集连接。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信度。
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 创建一个验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
# 模型评估
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)
# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件
# 有四个条目：在训练和验证期间，每个条目对应一个监控指标。我们可以使用这些条目来绘制训练与验证过程的损失值（loss）和准确率（accuracy），以便进行比较。
history_dict = history.history
history_dict.keys()
# 训练集以及验证集损失值图表展示
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# 训练集以及验证集准确率图表展示
plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

