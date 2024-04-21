### 此资源由 58学课资源站 收集整理 ###
#		想要获取完整课件资料 请访问：58xueke.com
#		百万资源 畅享学习
#	
# -*- coding:utf-8 -*-
"""
# datetime: 20:29
# software: PyCharm
"""
import joblib

import warnings

warnings.filterwarnings("ignore")
# 存储模型
# joblib.dump(model,filename="")

###1、加载回复模型
knn = joblib.load("./knn.m")

###2、对待预测的数据进行预测 （数据处理好后的数据）

x = [[5.9, 3.0, 4.2, 1.5]]
y_hat = knn.predict(x)
y_hat_prob = knn.predict_proba(x)
print(y_hat)
print(y_hat_prob)
