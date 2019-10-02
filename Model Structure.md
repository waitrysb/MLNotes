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
  > it improved at public but turn bad at pravate.Maybe can remove it.
  >
  > pre=((xgb^0.65)*(nn^0.35))*0.85+et*0.15
  > then userd [@weiwei](https://www.kaggle.com/weiwei) 's Prior correction. but only improved 0.00001-0.00002
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



2.[Two Sigma Connect: Rental Listing Inquiries第三名](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32123)

三层模型，比赛过程best single model --> some more base models for diversity --> ensemble of them --> improve best single model --> improve ensemble --> generate more base model to make sure the improvement flows from bottom to top --> repeat

when you find a new feature that will improves the performance, you build a new model and add it into the ensembling model sets to see if it improve the ensembling performance,like forward model selection? Will you discard some of the models in this process?Or will you reuse some old model you built?

* 第一层，lightgbm，xgboost，nn，分类和回归都训练，stacknet取第一层输出
* 第二层，lightgbm和nn，只分类，(LightGBM + NN + RF) x (features I + features II) = 6 models
* 第三层，加权平均