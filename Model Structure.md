# Model Structure

1.[Two Sigma Connect: Rental Listing Inquiries第一名](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32163)

四层模型

* 第一层

  > a.最优单模型；
  >
  > b.一些不能提升最优单模型的特征，但是与其他基本特征交叉可以提升模型；
  >
  > c.gdy5的kernel加上自己的一些特征；
  >
  > d.Branden Murrayit的kernel加上自己的一些特征

* 第二层

  >①:each dateset I used [xgb,nn,gb,rf,et,lr,xgb*reg,lgb*reg,nn_reg] cv flod=5
  >
  >the reg model have a good importance in my model.
  >
  >②:and I merge high and medium level ,then used[lgb,nn,lgb*reg,nn*reg,et,rf] in my best dataset. cv flod=5
  >
  >③:[nn,nn*reg,xgb,gb,rf,et,lr,xgb*reg]@last three datasets cv flod=5
  >
  >④:[nn,nn*reg,xgb,gb,rf,et,lr,xgb*reg]add magic feature [@last](https://www.kaggle.com/last) three datasets cv flod=5
  >
  >⑤:[nn,nn*reg,xgb,knn,gb,rf,et,lr,ada*reg,rf*reg,gb*reg,et*reg,xgb*reg]@last three datasets cv flod=10

* 第三层

  > 1.use ①,②,③,④ as metefeatures with xgb,nn,et.
  >
  > with a feature from description,Classify the source by description:
  > begin with " "
  >
  > 在public上有提升但是private上反而下滑了，可以移除吧
  >
  > pre=((xgb^0.65)*(nn^0.35))*0.85+et*0.15
  > 然后使用 [@weiwei](https://www.kaggle.com/weiwei)的先验纠正.但是只提升了0.00001-0.00002
  >
  > 2.use ①,②,⑤ as metefeatures with xgb,nn,et.
  > pre=((xgb^0.65)*(nn^0.35))*0.85+et*0.15

* 第四层：50/50 average level 3

nn模型结构：

> clf = Sequential()
> clf.add(Dense(64, input*dim=tr*x.shape[1],activation="relu", W*regularizer=l2())) clf.add(Dense(64,activation="relu",W*regularizer=l2()))
> clf.add(Dense(class_num, activation="softmax"))
>
> The features are same as xgboost,actually all of my models used the same features.
> I think the improve comes from the Transform.
> When I only use Standardscaler,nn score:0.54+ in cv.
> Then I use log10(X+1) before Standardscaler,nn score:0.532 in cv.



2.[Two Sigma Connect: Rental Listing Inquiries第二名](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32148)

**Base Models**

- 32 LightGBM models
- 9 ET models (sklearn)
- 7 RF models (sklearn)
- 5 Keras models
- 3 XGBoost models
- [@KazAnova](https://www.kaggle.com/KazAnova)'s StackNet example base-level predictions

这些模型使用使用不同特征来训练，可能多类、回归或者二分类模型

Those models have been trained on different feature sets and they are either multi-class, regression or binary classification models built on
5-Fold stratified splits.

**最优模型**：lgb并使用grid search bagging进行调参，如果新模型加入后提升了整体的cv分数，则加入到bag中，最后是一个12-bagged模型，分数比一个正常的15次bag模型更好

**最终提交**：使用的是两个两层Keras-NN网络，第一个创建测试结果不断迭代交叉验证，第二个（更好）使用early stopping每折35 bags

3.[Two Sigma Connect: Rental Listing Inquiries第三名](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32123)

三层模型，比赛过程best single model --> some more base models for diversity --> ensemble of them --> improve best single model --> improve ensemble --> generate more base model to make sure the improvement flows from bottom to top --> repeat

when you find a new feature that will improves the performance, you build a new model and add it into the ensembling model sets to see if it improve the ensembling performance,like forward model selection? Will you discard some of the models in this process?Or will you reuse some old model you built?

* 第一层，lightgbm，xgboost，nn，分类和回归都训练，stacknet取第一层输出
* 第二层，lightgbm和nn，只分类，(LightGBM + NN + RF) x (features I + features II) = 6 models
* 第三层，加权平均