# 代码执行说明

```
模型是在colab上训练的，训练得到的模型已经复制到了checkpoints文件夹中。

执行说明以外的都放在了实验报告中。
```

1. cd 到 data 文件夹下，执行`python pre_process.py [true]`。添加 true 参数会得到含有标点的语料。否则生成的语料中含有标点。生成的文件在 dataset/out 中
2. 在 data 文件夹下，执行`python wordvector_model.py`获取。会生成 word embedding 模型。生成的模型在 checkpoints/中
3. 在项目文件夹下，执行`python main_train.py`，训练 lstm 模型，模型会存到 checkpoints/中
4. 在项目文件夹下，执行`python main_generate.py`，使用 lstm 模型生成语段。可以修改代码中的四个属性得到自己需要的结果
   - input_words:给出的前几个单词
   - generate_len:要生成的长度
   - word2vec_path:word embedding 模型路径
   - lstm_model_path:lstm 模型路径
