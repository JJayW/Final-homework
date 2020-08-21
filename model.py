

from math import sin
import backtrader as bt
import matplotlib.pyplot as plt
import akshare as ak
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import tushare as ts
import math

from tensorflow import keras

from joblib import dump,load
from datetime import datetime

def multi_step_predict(model,data,N=1,feature_size=1):
    y_hat=[]
    x_test_last=data
    for i in range(N):
        y_hat_s1=model.predict(x_test_last.reshape(1,-1,feature_size))[0,0]
        x_test_last=np.roll(x_test_last,-1)
        x_test_last[-1]=y_hat_s1
        y_hat.append(y_hat_s1)
    return y_hat



class MyStrategy(bt.Strategy):
    """
    主策略程序
    """
    params = (("maperiod",20),)  #全局设定贸易策略的参数
    model=0
    x_scalar = 0
    y_scalar = 0
    
    def __init__(self):
        """
        初始化函数
        """
        self.data_close = self.datas[0].close  #指定价格序列
        #初始化交易指令、买卖价格和手续费
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        
        self.win_size = 50
        self.feature_size = 4
        self.x_last = np.zeros((self.win_size,self.feature_size))
        self.days_elapsed = 0
        #添加移动均线指标
        
        
    def next(self):
        """
        
        :return:
        :rtype:
        """
        
        future = 3

        #prepareData
        self.days_elapsed+=1
        self.x_last=np.roll(self.x_last,-1)
        self.x_last[-1]=np.array([self.data.open[0],self.data.high[0],self.data.low[0],self.data.close[0]])
        
        
        if self.days_elapsed<self.win_size:
            return 
        
        scaled_x=x_scalar  # .transform(self.x_last)
        y_hat=multi_step_predict(model,scaled_x,N=future,feature_size=self.feature_size)

        y=y_scalar  # .transform(np.array(self.data_close[0]).reshape(1,1))[0,0]

        buy_margin=0.03
        sell_margin=-0.03
        if self.order: #检查是否有指令等待执行
            return
        #检查是否持仓
        if not self.position:  #没有持仓
            if  y_hat[-1] > ((1+buy_margin)*y)[-1]:  #执行买入条件判断：收盘价格上涨突破20日均线
                self.order = self.buy(size=100)   #执行买入
        else:
            if y_hat[-1] < ((1+sell_margin)*y)[-1]:   #执行卖出条件判断：收盘价格跌破20日均线
                self.order = self.sell(size=100)  #执行卖出
                
         

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
x_scalar = load('X_train_single.bin')
y_scalar = load('Y_train_single.bin')

plt.rcParams["font.sans-serif"] = ["SimHei"]   #设置字体
plt.rcParams["axes.unicode_minus"] = False

stock_hfq_df = ak.stock_zh_a_daily(symbol="sh600776",adjust="hfq")   #利用AkShare 获取后复权数据

MyStrategy.model=model
MyStrategy.x_scalar=x_scalar
MyStrategy.y_scalar=y_scalar



cerebro = bt.Cerebro()   #初始化回测系统
start_date = datetime(2018,7,1)   #回测开始时间
end_date = datetime(2020,7,29)    #回测结束时间
data = bt.feeds.PandasData(dataname=stock_hfq_df,fromdate=start_date,todate=end_date)   #加载数据
cerebro.adddata(data)   #将数据传回系统
cerebro.addstrategy(MyStrategy)   #将交易策略加载到回测系统中
start_cash = 1000000
cerebro.broker.setcash(start_cash)   #设置初始资本为1000000
cerebro.broker.setcommission(commission=0.002)    #设置交易手续费为0.2%
cerebro.run()    #运行回测系统

port_value = cerebro.broker.getvalue()  #获取回测结束后的总资金
pnl = port_value - start_cash  #盈亏统计

print(f"初始资金：{start_cash}\n回测期间：{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}")
print(f"总资金：{round(port_value,2)}")
print(f"净收益：{round(pnl,2)}")





