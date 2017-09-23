# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:19:06 2017

@author: gao
"""

import tensorflow as tf
import pickle
import numpy as np
import os

input_units=412
hidden_units=206
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,input_units])
W1=tf.Variable(tf.truncated_normal([input_units,hidden_units],stddev=0.1))
W2=tf.Variable(tf.zeros([hidden_units,1]))
b1=tf.Variable(tf.zeros([hidden_units]))
b2=tf.Variable(tf.zeros([1]))

h1=tf.nn.relu(tf.matmul(x,W1)+b1)
y=tf.nn.sigmoid(tf.matmul(h1,W2)+b2)

y_=tf.placeholder(tf.float32,[None,1])
#定义反向传播函数 损失函数用来刻画预测值与真实值的差距
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
mse=tf.reduce_mean(tf.square(y_-y))
#定义反向传播算法来优化神经网络中的参数
#train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(mse)


tf.global_variables_initializer().run()
filepath='/home/gaofei/vad/mix_data/'
dirname=os.listdir(filepath)#获取全部文件
for j in range(10):
    loss=0
    for i in range(0,len(dirname)):
        data=open(filepath+dirname[i],"rb")
        data_dict=pickle.load(data)
        train_data=data_dict['mix_data']
        train_label=data_dict['mix_label']
        
        data.close()
        a=len(train_label)
        train_label=np.array(train_label).reshape((a,1))
        #print(train_label.shape)
        #train_data,label_data=(data_dict['mix_data'].T ,data_dict['mix_label'])
        loss1=sess.run(mse,{x:train_data.T,y_:train_label})
        train_step.run({x:train_data.T,y_:train_label})
        loss+=loss1
    #print(sess.run(cross_entropy,{x:train_data.T,y_:train_label}))
    print(loss/len(dirname))
    
    
  

file1='/home/gaofei/vad/factory_test/'
dirname1=os.listdir(file1)
result=0
for i in range(0,len(dirname1)):
    data1=open(file1+dirname1[i],"rb")
    data_dict1=pickle.load(data1)
    test_data=data_dict1["mix_data"]
    test_label=data_dict1["mix_label"]
    a=len(test_label)
    test_label=np.array(test_label).reshape((a,1))
    data1.close()
    
    correct_prediction=tf.equal(tf.nn.relu(tf.sign(y-0.5)),test_label)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    total=accuracy.eval({x:test_data.T,y_:test_label})
    #print(total)
   
    result+=total
print('factory',result/len(dirname1))
'''
file1='/home/gaofei/vad/white_test/'
dirname1=os.listdir(file1)
result=0
for i in range(0,len(dirname1)):
    data1=open(file1+dirname1[i],"rb")
    data_dict1=pickle.load(data1)
    test_data=data_dict1["mix_data"]
    test_label=data_dict1["mix_label"]
    a=len(test_label)
    test_label=np.array(test_label).reshape((a,1))
    data1.close()
    
    correct_prediction=tf.equal(tf.nn.relu(tf.sign(y-0.5)),test_label)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    total=accuracy.eval({x:test_data.T,y_:test_label})
    #print(total)
   
    result+=total
print('white',result/len(dirname1))
'''
file1='/home/gaofei/vad/babble_test/'
dirname1=os.listdir(file1)
result=0
for i in range(0,len(dirname1)):
    data1=open(file1+dirname1[i],"rb")
    data_dict1=pickle.load(data1)
    test_data=data_dict1["mix_data"]
    test_label=data_dict1["mix_label"]
    a=len(test_label)
    test_label=np.array(test_label).reshape((a,1))
    data1.close()
    
    correct_prediction=tf.equal(tf.nn.relu(tf.sign(y-0.5)),test_label)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    total=accuracy.eval({x:test_data.T,y_:test_label})
    #print(total)
   
    result+=total
print('babble',result/len(dirname1))
'''
file1='/home/gaofei/vad/cafe_test/'
dirname1=os.listdir(file1)
result=0
for i in range(0,len(dirname1)):
    data1=open(file1+dirname1[i],"rb")
    data_dict1=pickle.load(data1)
    test_data=data_dict1["mix_data"]
    test_label=data_dict1["mix_label"]
    a=len(test_label)
    test_label=np.array(test_label).reshape((a,1))
    data1.close()
    
    correct_prediction=tf.equal(tf.nn.relu(tf.sign(y-0.5)),test_label)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    total=accuracy.eval({x:test_data.T,y_:test_label})
    #print(total)
   
    result+=total
print('cafe',result/len(dirname1))
'''
file1='/home/gaofei/vad/car_test/'
dirname1=os.listdir(file1)
result=0
for i in range(0,len(dirname1)):
    data1=open(file1+dirname1[i],"rb")
    data_dict1=pickle.load(data1)
    test_data=data_dict1["mix_data"]
    test_label=data_dict1["mix_label"]
    a=len(test_label)
    test_label=np.array(test_label).reshape((a,1))
    data1.close()
    
    correct_prediction=tf.equal(tf.nn.relu(tf.sign(y-0.5)),test_label)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    total=accuracy.eval({x:test_data.T,y_:test_label})
    #print(total)
   
    result+=total
print('car',result/len(dirname1))

'''
dic_wb={'W1':W1.eval(),
        'W2':W2.eval(),
        'b1':b1.eval(),
        'b2':b2.eval()}  
with open ('C:/Users/gao/vad/wb.pkl','wb') as f1:
    pickle.dump(dic_wb,f1) 
'''
sess.close()
