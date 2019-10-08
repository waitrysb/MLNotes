# Feature Processing

1.通过新增加一个或几个feature，如果cv分数上去了，就增加这个feature，如果cv分数没有上去，就舍弃这个feature，也就是相当于贪心验证。这样做的弊处在于，如果之前被舍弃的feature和之后被舍弃的feature联合在一起才会有正面影响，就相当于你错过了两个比较好的feature。因此特征的选择和联合显得非常关键。

2.**数值型feature的简单加减乘除**

这个乍一看仿佛没有道理可言，但是事实上却能挖掘出几个feature之间的内在联系，比如[这场比赛](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146)中提供了bathrooms和bedrooms的数量，以及价格price，合租用户可能会更关心每个卧室的价格，即bathrooms / price，也会关心是不是每个房间都会有一个卫生间bathrooms / price ，这些数值型feature之间通过算数的手段建立了联系，从而挖掘出了feature内部的一些价值，分数也就相应地上去了。还有与price相关的就是price%100，可以区分高价和低价

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
[例子](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32148)：对High Cardinal Categorical的特征用了一些额外的条件概率来计算其似然值，如p(y|manager_id, bathrooms)等，并且进行了点积操作来计算出一个合适的encoding值（类似于先前讨论区中出现的manager_skills

4.**时间特征**

提取年、月、日、星期等

时间差：比如当前时间-创建时间；按照id分组来求活跃时间差

交叉：时间与其他特征的可视化，观察图像，考虑函数关系

5.**地理位置**

聚类特征，算中心点坐标，计算曼哈顿距离或欧氏距离

街道地址：进行清洗，如first-》1st，west-》w，也可以进行统计；按照地址经纬度找到街道的经纬度；[将地址字符串翻转后用整数编码](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146)

6.**文本特征**

提取关键词，情感分析，word embedding，找除停用词外高频词

简单文本特征：大写小写比例，*!$等特殊字符的数量，电话号码或者邮箱的标志

7.**图片特征**

CNN提取

图像长度和宽度，面积和文件大小，分别进行聚类

像素平均大小（文件大小/面积），CRC后计数

8.**稀疏特征集**（标签特征）

计数后使用one-hot，同时根据标签意义合并一些标签，比如cat allowed 和 dog allowed可以合并成为 pet allowed，或者与$相关的标签

也可以按照另一个类别型特征分组，将组内的标签拼接为一个字符串，压缩长度或压缩比作为特征，如[description](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146)

9.**特征重要程度**

在树结构的分类器比如randomforest、xgboost中最后能够对每个特征在分类上面的重要程度进行一个评估。得出最重要的几个特征，然后进行交叉特征可视化

10.**类别型特征**

* 使用one-hot编码

* 使用类别的统计值如频率代替，转化为数值特征

* 与其他数值型特征交叉，按照类别求出数值统计值

* 与其他类别型特征合并，如no_pet和no_dog，要求这两种特征互斥

  > all[no_pet_no_dog] = all[no_pet] + (2 * all[no_dog])
  >
  > all['listingmistakescatint'] = all['buildingid_missing'] \
  > (2\*(all['photoslistlen'] < 1).astype(int)) \
  > (4\*(all['featureslistlen'] < 1).astype(int)) \
  > (8*(all['descriptionstrlen'] <= 8).astype(int))

11.**等级型特征**



12.**常用特征技巧**

* description属性是一个单词集合，对description出现频率最高的15k单词进行一个one-hot深度xgboost训练，将这个训练出来模型的预测结果作为description的encoding。

  也可以对description属性里的supermarket、bus、shopping等的经纬度进行聚类，求出这些属性的大致经纬度，进而可以估测距离

* 使用一种类别特征来给连续特征和类别特征分组，计算每组的统计值，如数量和方差等，与原始数据合并形成新的特征

* [I split the base features two classes:](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32163)

  manager:created,description,price,et

  building:bathrooms,bedrooms,latitude,longitude,display_address,featuers,photos,et

  Then I link and compare them one by one.
  
  [创建特征参考](https://github.com/plantsgo/Rental-Listing-Inquiries)
  
* 列举出有用的missing value

  > ```
  > all['listing_mistakes_2'] = all['building_id_missing'] + \
  >                            (all['photos_list_len'] < 1)
  > all['listing_mistakes_3'] = all['building_id_missing'] +  \
  >                            (all['photos_list_len'] < 1) + \
  >                            (all['features_list_len'] < 1)
  > all['listing_mistakes_4'] = all['building_id_missing'] + \
  >                            (all['photos_list_len'] < 1) + \
  >                            (all['features_list_len'] < 1) + \
  >                            (all['description_str_len'] <= 8)
  > all['listing_mistakes_cat_int'] = all['building_id_missing'] \
  >       + (2*(all['photos_list_len'] < 1))    \
  >       + (4*(all['features_list_len'] < 1))  \
  >       + (8*(all['description_str_len'] <= 8))
  > ```

* 可以使用[LightGBM](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146)来看特征的重要性

