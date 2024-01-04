"""
基本线性回归模型实现
"""
import numpy as np
from utils.features import prepare_for_training # 预处理模块

class LinearRegression:

    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean, 
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=True)
         
        self.data = data_processed # 使用预处理之后的数据
        self.labels = labels # 标签
        self.features_mean = features_mean # 均值
        self.features_deviation = features_deviation # 标准差
        self.polynomial_degree = polynomial_degree # 多项式次数
        self.sinusoid_degree = sinusoid_degree # 正弦函数次数
        self.normalize_data = normalize_data # 归一化
        
        num_features = self.data.shape[1] # 1表示列，即有多少个特征
        self.theta = np.zeros((num_features,1)) # 初始化参数矩阵
        
    def train(self, alpha, num_iterations = 500):
        """
                    num_iterations：迭代次数
                    训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha,num_iterations)
        
        """
        由于self.theta是类的属性，可以在整个类的实例中访问，所以它并不需要显式地在gradient_descent方法中返回。
        """
        return self.theta, cost_history
        
    def gradient_descent(self,alpha,num_iterations):
        """
                    实际迭代模块，会迭代num_iterations次
        """
        cost_history = [] # 记录每一次损失
        for _ in range(num_iterations):
            self.gradient_step(alpha) # 执行梯度下降
            cost_history.append(self.cost_function(self.data,self.labels)) # 添加损失
        return cost_history
        
        
    def gradient_step(self, alpha):    
        """
                    一次梯度下降
                    梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0] # 样本个数
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels # 真实值就是label
        theta = self.theta
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T # 使用矩阵的方法就省去了使用for循环
        self.theta = theta # 更新theta
        
        
    def cost_function(self,data,labels):
        """
                    损失计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)/num_examples # 使用均方误差进行损失计算，得出的结果是一个标量
        return cost[0][0]
        
        
        
    @staticmethod
    def hypothesis(data,theta):
        """
                    计算预测值
        """
        predictions = np.dot(data,theta)
        return predictions
        
    def get_cost(self,data,labels):
        """
        得到当前损失
        """
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
        
        return self.cost_function(data_processed,labels)
    
    def predict(self,data):
        """
                    用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
        
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        
        return predictions
        
        
        
        