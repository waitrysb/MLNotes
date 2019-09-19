# Feature Processing

1.通过新增加一个或几个feature，如果cv分数上去了，就增加这个feature，如果cv分数没有上去，就舍弃这个feature，也就是相当于贪心验证。这样做的弊处在于，如果之前被舍弃的feature和之后被舍弃的feature联合在一起才会有正面影响，就相当于你错过了两个比较好的feature。因此特征的选择和联合显得非常关键。

2.**数值型feature的简单加减乘除**

这个乍一看仿佛没有道理可言，但是事实上却能挖掘出几个feature之间的内在联系，比如这场比赛中提供了bathrooms和bedrooms的数量，以及价格price，合租用户可能会更关心每个卧室的价格，即bathrooms / price，也会关心是不是每个房间都会有一个卫生间bathrooms / price ，这些数值型feature之间通过算数的手段建立了联系，从而挖掘出了feature内部的一些价值，分数也就相应地上去了。

3.**高势集类别（High Categorical）**即某个特征的类别特别多

**clustering**：依据target的值grouping成k个类(clusters)，然后再依据这个grouping的结果进行one-hot编码，这个方法最大程度的保留了原始数据的信息

**smoothing**：将特征的各个类别映射到条件概率上
$$
S_{i}=\lambda(n_{i})\frac{n_{iY}}{n_{i}}+(1-\lambda(n_{i}))\frac{n_{Y}}{n_{TR}}
$$
其中i表示类别，niY表示该类别对应的target为Y的数量，ni表示该类别数量，nTR表示训练集的总数量，lambda函数是一个在0-1间的单调递增函数，+前是后验概率，+后是先验概率，[参考](https://www.cnblogs.com/bjwu/p/9087071.html)
$$
\lambda(n)=\frac{1}{1+\exp^{-\frac{n-k}{f}}}
$$
4.**时间特征**

提取年、月、日、星期等

时间差：比如当前时间-创建时间

交叉：时间与其他特征的可视化，观察图像，考虑函数关系

5.**地理位置**

聚类特征，算中心点坐标，计算曼哈顿距离或欧氏距离

6.**文本特征**

提取关键词，情感分析，word embedding，找除停用词外高频词

7.**图片特征**

CNN提取

8.**稀疏特征集**（标签特征）

计数后使用one-hot，同时根据标签意义合并一些标签，比如cat allowed 和 dog allowed可以合并成为 pet allowed

9.**特征重要程度**

在树结构的分类器比如randomforest、xgboost中最后能够对每个特征在分类上面的重要程度进行一个评估。得出最重要的几个特征，然后进行交叉特征可视化

10.**常用特征技巧**

* 对description出现频率最高的15k单词进行一个one-hot深度xgboost训练，将这个训练出来模型的预测结果作为description的encoding。

