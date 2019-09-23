# Model Structure

1.[Two Sigma Connect: Rental Listing Inquiries第三名](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32123)

三层模型，比赛过程best single model --> some more base models for diversity --> ensemble of them --> improve best single model --> improve ensemble --> generate more base model to make sure the improvement flows from bottom to top --> repeat

when you find a new feature that will improves the performance, you build a new model and add it into the ensembling model sets to see if it improve the ensembling performance,like forward model selection? Will you discard some of the models in this process?Or will you reuse some old model you built?

* 第一层，lightgbm，xgboost，nn，分类和回归都训练，stacknet取第一层输出
* 第二层，lightgbm和nn，只分类，(LightGBM + NN + RF) x (features I + features II) = 6 models
* 第三层，加权平均