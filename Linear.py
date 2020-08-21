# 导入必要库
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tushare as ts
from joblib import dump,load

#获取数据
ts.set_token('364c32bf12415e321deef6ba1d227de7e1e9d9de05c8aa050d35ea71')
pro = ts.pro_api('364c32bf12415e321deef6ba1d227de7e1e9d9de05c8aa050d35ea71')
df = pro.daily(ts_code='600776.SH', start_date='20150101', end_date='20200729')

#保存数据
#df.to_excel('1.xlsx')

# 读数据
# df = pd.read_excel('1.xlsx')
df = df[0:].set_index(df['trade_date'][0:])# 设置时间为索引
df = df.drop(columns=['ts_code', 'trade_date'])

# 设置训练集、验证集和测试集的比例为7：2：1
n = len(df)
train_split = int(0.7*n)
val_split = int(0.9*n)

df.head()

df_mean = df[:train_split].mean(axis=0)  # 计算每个特征在训练集的均值备用，axis=0表示计算的是每个特征而不是每日各个特征的均值
df_std = df[:train_split].std(axis=0)  # 计算每个特征在训练集的标准差备用
df = (df - df_mean) / df_std  # 标准化

df = df.values
target = df[:, 3]

print(type(df))  # values后df的类型从DataFrame变成了ndarray，没有head()方法
print(df.shape) 

def window_generator(dataset, target, start_index, end_index, 
                     history_size, target_size):
    
    """
    Generate window for training, validation and testing.
    
    Parameters:
    
        dataset:collection of features; target:collection of labels;

        start_index:beginning of the slice; end_index:end of the slice;

        history_size:input width; target_size:label width.

    """
    
    features = []
    labels = []
    
    if end_index is None:
        end_index = len(dataset) - target_size
    
    start_index += history_size
        
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        features.append(dataset[indices])
        labels.append(target[i:i+target_size])
        
    return np.array(features), np.array(labels)
def loss_curve(history):
    
    """
    Plotting the loss curve.
        
    """
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

#封装训练过程
EPOCHS = 100  # 设置最大训练轮数为100轮
EVALUATION_INTERNAL = 120  # step per epoch

# 数据增强参数备用
BATCH_SIZE = 100
BUFFER_SIZE = 2000


def compile_and_fit(model, train_data, val_data, patience=10):
    
    """
    Define the process of compling and fitting the model.
    
    """
    
    # 为防止过拟合，监视验证集上的loss值，在10个epoch内没有发生太大变化则终止训练
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=patience,
    mode='auto',
    restore_best_weights=True)  # 返回最优参数，而非训练停止时的参数
    
    # 模型编译
    model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0),  # 设置优化器
                 loss='mae')  # 设置损失函数
    
    # 模型拟合
    history = model.fit(train_data, epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERNAL,
                        validation_steps=50,
                        validation_data=val_data,
                        callbacks=[early_stopping])
    return history


#用时间窗对数据进行处理

X_train_single, y_train_single = window_generator(dataset=df, target=target, start_index=0,
                                                 end_index=train_split, history_size=1, target_size=1)

X_val_single, y_val_single = window_generator(dataset=df, target=target, start_index=train_split,
                                             end_index=val_split, history_size=1, target_size=1)

X_test_single, y_test_single = window_generator(dataset=df, target=target, start_index=val_split,
                                             end_index=n-1, history_size=1, target_size=1)

X_train_multi, y_train_multi = window_generator(dataset=df, target=target, start_index=0,
                                                 end_index=train_split, history_size=5, target_size=1)

X_val_multi, y_val_multi = window_generator(dataset=df, target=target, start_index=train_split,
                                             end_index=val_split, history_size=5, target_size=1)

X_test_multi, y_test_multi = window_generator(dataset=df, target=target, start_index=val_split,
                                              end_index=n-5, history_size=5, target_size=1)
                                              
print(X_train_multi.shape)
print(X_val_multi.shape)
print(X_test_multi.shape)

#数据增强
train_single = tf.data.Dataset.from_tensor_slices((X_train_single, y_train_single))
val_single = tf.data.Dataset.from_tensor_slices((X_val_single, y_val_single))

train_multi = tf.data.Dataset.from_tensor_slices((X_train_multi, y_train_multi))
val_multi = tf.data.Dataset.from_tensor_slices((X_val_multi, y_val_multi))

train_single = train_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()  # 数据增强，shuffle只对训练集做
val_single = val_single.cache().batch(BATCH_SIZE).repeat()  # 数据增强，shuffle只对训练集做

train_multi = train_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()  # 数据增强，shuffle只对训练集做
val_multi = val_multi.cache().batch(BATCH_SIZE).repeat()  # 数据增强，shuffle只对训练集做


#Linear Model
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

linear_history = compile_and_fit(linear, train_single, val_single)

loss_curve(linear_history)

linear_result = linear.predict(X_test_single).reshape(-1,1)

fig = plt.figure(figsize=(15, 8))
ax = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(10))

plt.plot(y_test_single, label='oringin')
plt.plot(linear_result, label='linear')

plt.legend()
plt.show()

linear.save('Linear.mod')
dump(X_train_single,'X_train_single.bin',compress=True)
dump(y_train_single,'y_train_single.bin',compress=True)