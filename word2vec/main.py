# 3.1文本预处理

# %%
# 3.1.1导入各种包和数据集
# 导入所需要的jieba模块、gemsim模块和其他相关模块并加载文件路径
import os
import jieba
import warnings
import gensim.models as w2v

warnings.filterwarnings('ignore')
novel_path = "D:/Code/Python/NLP/word2vec/novel/"
data_path = "D:/Code/Python/NLP/word2vec/"

# %%
# 3.1.2文本过滤
# 过滤掉文本中的标点符号和一些训练词向量时不需要，单独出现并没有什么意义的停词(stop word)。
# 停词表选用的是网上通用的中文语料停词表，我又手动在里面添加了部分武侠小说里常出现的对训练词向量没什么意义的词汇和短语。这里把标点符号当作停用词合并一起处理。

stop_words_file = open(data_path + "stop_words.txt", 'r', encoding='UTF-8')
stop_words = list()
for line in stop_words_file.readlines():
    line = line.strip()  # 去掉每行末尾的换行符
    stop_words.append(line)
stop_words_file.close()
print(len(stop_words))

# %%

# 3.1.3添加自定义词汇
# jieba的词汇表中并没有收录很多金庸的武侠小说这种特定环境下的很多专有名词，包括一些重要人物的名称，一些重要的武功等。这里整理了三份txt文本，分别记录了武侠小说中的人物名称、武功名称和门派名称，并把这些词汇添加到词汇表中。
# 人名、武功、门派txt文本参考来源：金庸网-金庸数据库。
#
# （1）将人名添加到词汇表中：
people_names_file = open(data_path + "金庸小说全人物.txt", 'r', encoding='UTF-8')
people_names = list()
for line in people_names_file.readlines():
    line = line.strip()  # 去掉每行末尾的换行符
    jieba.add_word(line)
    people_names.append(line)
stop_words_file.close()
print(len(people_names))

# %%
# （2）将武功添加到词汇表中：
people_names_file = open(data_path + "金庸小说全武功.txt", 'r', encoding='UTF-8')
people_names = list()
for line in people_names_file.readlines():
    line = line.strip()  # 去掉每行末尾的换行符
    jieba.add_word(line)
    people_names.append(line)
stop_words_file.close()
print(len(people_names))

# %%
# （3）将门派添加到词汇表中：
people_names_file = open(data_path + "金庸小说全门派.txt", 'r', encoding='UTF-8')
people_names = list()
for line in people_names_file.readlines():
    line = line.strip()  # 去掉每行末尾的换行符
    jieba.add_word(line)
    people_names.append(line)
stop_words_file.close()
print(len(people_names))

# %%
# 3.1.4分词
# 将所有文本分词并显示每一本书分词后的行数和总行数，
# 通过一个嵌套列表存储，每一句为列表中一个元素，每一句又由分好的词构成一个列表，这也是word2vec训练时需要输入的格式。

novel_names = os.listdir(novel_path)

seg_novel = []
for novel_name in novel_names:
    novel = open(novel_path + novel_name, 'r', encoding='utf-8-sig')
    print("Waiting for {}...".format(novel_name))
    line = novel.readline()
    forward_rows = len(seg_novel)
    while line:
        line_1 = line.strip()
        out_str = ''
        line_seg = jieba.cut(line_1, cut_all=False)
        for word in line_seg:
            if word not in stop_words:
                if word != '\t':
                    if word[:2] in people_names:
                        word = word[:2]
                    out_str += word
                    out_str += " "
        if len(str(out_str.strip())) != 0:
            seg_novel.append(str(out_str.strip()).split())
        line = novel.readline()
    print("{} finished，with {} Row".format(novel_name, (len(seg_novel) - forward_rows)))
    print("-" * 40)
print("-" * 40)
print("-" * 40)
print("All finished，with {} Row".format(len(seg_novel)))


# %%
# 3.2 训练词向量
# Gemsim模块是一个功能很强大的NLP处理模块，这里用到了Gemsim模块中Word2Vec函数。
model = w2v.Word2Vec(sentences=seg_novel, vector_size=200, window=5, min_count=5, sg=1)
model.save(data_path + 'all_CBOW.model')  # 保存模型
print('训练完成')


# %%
print(model.wv.similarity('杨过', '小龙女'))
print(model.wv.most_similar("杨过", topn=10))

# %%
print(model.wv.most_similar("张无忌", topn=10))


# %%
print(model.wv.similarity('段誉', '一阳指'))
print(model.wv.similarity('段誉', '六脉神剑'))
print(model.wv.similarity('段誉', '凌波微步'))
print(model.wv.similarity('段誉', '北冥神功'))

