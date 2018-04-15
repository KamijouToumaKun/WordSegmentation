# WordSegmentation
NLP课上用感知机做的自动分词器

用java写的，训练集为train.txt；
测试集与答案为test.txt, test.answer.txt；
输出的文件test_output.txt放进评测脚本的子文件夹里，用脚本评测两者之间的区别；
脚本的参数为training_words.txt, test.answer.txt, test_output.txt

最后，我设置的一些参数：
10+2 features
requirement：feature popularity >= 3
epoch = 30
0 94.2%（没什么差别，而且还整数化了，可以更快！）
0.3 94.4%

特征一开始设置少了 没有加标签
1、平均感知器、随机训练没有普通感知器好
（不过好像我写的不是真正的平均感知器……）
2、训练和预测时按照标点切开没有不切开好
（改证后发现好一点点，基本没有区别）
3、维特比没有贪心好
（用感知器根本就没有HMM和viterbi的事情，那些的概率矩阵是靠统计得到的，而不是感知器学来的）
4、初值为0没有初值为其他好
（基本没差）
5、theta写成稀疏向量没有普通的好
6、多训练几次没有少训练几次好
（基本没差）
