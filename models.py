import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import data_process as process
import time
import math
from PIL import Image
from matplotlib import cm
from sklearn.model_selection import KFold

def gen_heat_maps(heats,images,outputs,para):
    heat_maps = []
    for i in range(len(heats)):
        heat = heats[i]
        image = images[i]
        output = outputs[i]
        
        classify = np.argmax(output)
        cm_jet = cm.get_cmap('jet')
        sum_h = heat[:,:,0]*para[0,classify]
        
        for j in range(1,len(para)):
            sum_h += heat[:,:,j]*para[j,classify]
            
        # resize
        im = np.array(Image.fromarray(sum_h).resize((224,224)))
        im_nor = im / np.max(im)
        
        # rgba slice to rgb
        im = np.array(cm_jet(np.array(im_nor)))[:,:,:3]

        im[np.where(im_nor < 0.2)] = 0
        im = np.uint8(im*95+image*159)

        heat_maps.append(np.array(im))
    return heat_maps

def outputs_to_csv(outputs,filenames,path):
    pf = pd.DataFrame(outputs,columns=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    pf['img'] = pd.Series(filenames)
    pf = pf[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
    pf.to_csv(path,index=False)

def plot_gallery(images, titles, n_row=10, n_col=5):
    pl.figure(figsize=(2.4 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(len(images)):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i])
        pl.title(titles[i], size=10)
        pl.xticks(())
        pl.yticks(())

def test_set_predict(path,n):
    with tf.Session() as sess:
        meta_path = path + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess,path)
        graph = tf.get_default_graph()
        
        x_ = graph.get_tensor_by_name('input_x:0')
        is_train_ = graph.get_tensor_by_name('is_train:0')
        outputs_ = graph.get_tensor_by_name('outputs:0')
        
        # forward
        images_arr = []
        titles_arr = []
        count = 0
        for test_batch,filename_batch in process.get_test_images(1):
            feed = {
                x_:test_batch,
                is_train_:False,
            }
            outputs = sess.run([outputs_],feed_dict=feed)
            images_arr.extend([np.uint8((item+0.5)*255) for item in test_batch])
            titles_arr.append(process.int_txt_dict[np.argmax(outputs[0])])
            count += 1
            if count == n:
                break
        
        plot_gallery(images_arr,titles_arr)

class Vgg13_gam(object):
    def __init__(self,x_shape,class_n):
        self.width = x_shape[0]
        self.height = x_shape[1]
        self.channel = x_shape[2]
        self.class_n = class_n
        print('use default graph')
        tf.reset_default_graph()
        self.inputs()
        self.graph()
     
    def inputs(self):
        self.x = tf.placeholder(tf.float32,(None,self.width,self.height,self.channel),name='input_x')
        self.y = tf.placeholder(tf.float32,(None,self.class_n),name='input_y')
        self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.is_train = tf.placeholder(tf.bool,name='is_train')
        
    def cnn_layers(self,x,filters):
        cnn_output = tf.layers.conv2d(x,filters,3,1,padding='same',name='cnn_layer')
        relu = tf.maximum(cnn_output,0.1*cnn_output,name='leaky_relu')
#         return relu
        batch_nor = tf.layers.batch_normalization(relu,training=self.is_train,name='batch_nor')
        return batch_nor
    
    def pooling_layers(self,input):
        pool = tf.layers.max_pooling2d(input,2,2,'same')
        return pool
    
    def graph(self):
        print('init graph')
        with tf.variable_scope('layers1'):
            cnn1 = self.cnn_layers(self.x,64)
        with tf.variable_scope('layers2'):
            cnn2 = self.cnn_layers(cnn1,64)
        with tf.variable_scope('pooling1'):
            pool1 = self.pooling_layers(cnn2)
            
        with tf.variable_scope('layers3'):
            cnn3 = self.cnn_layers(pool1,128)
        with tf.variable_scope('layers4'):
            cnn4= self.cnn_layers(cnn3,128)
        with tf.variable_scope('pooling2'):
            pool2 = self.pooling_layers(cnn4)
            
        with tf.variable_scope('layers5'):
            cnn5 = self.cnn_layers(pool2,256)
        with tf.variable_scope('layers6'):
            cnn6 = self.cnn_layers(cnn5,256)
        with tf.variable_scope('layers7'):
            cnn7 = self.cnn_layers(cnn6,256)
        with tf.variable_scope('pooling3'):
            pool3 = self.pooling_layers(cnn7)
            
        with tf.variable_scope('layers8'):
            cnn8 = self.cnn_layers(pool3,512) 
        with tf.variable_scope('layers9'):
            cnn9 = self.cnn_layers(cnn8,512)
        with tf.variable_scope('layer10'):
            cnn10 = self.cnn_layers(cnn9,512)
        with tf.variable_scope('pooling4'):
            pool4 = self.pooling_layers(cnn10)
            
        with tf.variable_scope('layers11'):
            cnn11 = self.cnn_layers(pool4,512)
        with tf.variable_scope('layers12'):
            cnn12 = self.cnn_layers(cnn11,512)
        with tf.variable_scope('layers13'):
            cnn13 = self.cnn_layers(cnn12,512)
            
        with tf.variable_scope('GAP'):
            raw_cam = tf.nn.avg_pool(cnn13,[1,14,14,1],[1,14,14,1],'SAME',name='reduce_mean')
#             raw_cam = tf.layers.conv2d(cnn13,512,14,14,'same',name='reduce_mean')
            cam = tf.reshape(raw_cam,[-1,512],name='reshape')
            
        with tf.variable_scope('drop_out'):
            drop_out = tf.layers.dropout(cam,self.keep_prob,name='')
            
        with tf.variable_scope('fully_connected'):
            logits = tf.layers.dense(drop_out,self.class_n,name='fully_connected')
            outputs = tf.sigmoid(logits,name='outputs')
            
        with tf.variable_scope('error'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=self.y)
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            
        with tf.variable_scope('metrics'):
#             accuracy = tf.metrics.accuracy(tf.argmax(self.y,axis=1),tf.argmax(outputs,axis=1))[1]
            equal = tf.equal(tf.argmax(self.y,axis=1),tf.argmax(outputs,axis=1))
            accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))
              
        self.cam = tf.identity(cnn13,name='cam')
        
        # temp
        self.logits = tf.identity(logits,name='logits')
        
        self.outputs = tf.identity(outputs,name='outputs')
        self.loss = tf.identity(loss,name='loss')
        self.accuracy = tf.identity(accuracy,name='accuracy')
        
    def cv_train(self,X,Y,epoch,batch_size,save_path):
        saver = tf.train.Saver()
        star_time = time.clock()
        step = 0
        with tf.Session() as sess:
            print('init variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            writer = tf.summary.FileWriter('./graph/')
            writer.add_graph(sess.graph)
            step = 0
            start_time = time.clock()
            current_time = start_time
            print('training...')
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            for i in range(epoch):
                kf = KFold(n_splits=10,shuffle=True)
                cv_n = 0
                for train_idx,valid_idx in kf.split(X):
                    cv_n += 1
                    train_x,train_y = X[train_idx],Y[train_idx]
                    valid_x,valid_y = X[valid_idx],Y[valid_idx]
                    
                    train_accuracy = []
                    train_loss = []
                    for x_batch,y_batch in process.get_batches(train_x,train_y,batch_size):
                        step += 1
                        feed = {
                            self.x:x_batch,
                            self.y:y_batch,
                            self.is_train:True,
                            self.learning_rate:0.001,
                            self.keep_prob:0.5
                        }
                        _,__,accuracy,loss = sess.run([self.optimizer,extra_update_ops,self.accuracy,self.loss],feed_dict=feed)
                        train_accuracy.append(accuracy)
                        train_loss.append(loss)
                        
                        if step % 100 == 0:
                            train_time = time.clock() - current_time
                            current_time = time.clock()
                            valid_accuracy = []
                            valid_loss = []
                            for x_batch,y_batch in process.get_batches(valid_x,valid_y,batch_size):
                                feed = {
                                    self.x:x_batch,
                                    self.y:y_batch,
                                    self.is_train:False,
                                    self.keep_prob:1
                                }
                                accuracy,loss,cross = sess.run([self.accuracy,self.loss,self.logits],feed_dict=feed)
                                valid_accuracy.append(accuracy)
                                valid_loss.append(loss)
                            valid_time = time.clock() - current_time
                            current_time = time.clock()
                            total_time = current_time - start_time
                            train_avg_acc = np.mean(train_accuracy)
                            train_accuracy = []
                            train_avg_loss = np.mean(train_loss)
                            train_loss = []
                            valid_avg_acc = np.mean(valid_accuracy)
                            valid_accuracy = []
                            valid_avg_loss = np.mean(valid_loss)
                            valid_loss = []
                            print("Epoch:{},cv:{},trainAcc:{:.4F},validAcc:{:.4F},trainLoss:{:.6F},validLoss:{:.6F},trainTime:{:.0F},validTime:{:.0F},totalTime:{:.0F}".format(i+1,cv_n,train_avg_acc,valid_avg_acc,train_avg_loss,valid_avg_loss,train_time,valid_time,total_time))
        
            saver.save(sess,save_path)
            print('finish')

class Nin(object):
    def __init__(self,x_shape,class_n):
        self.width = x_shape[0]
        self.height = x_shape[1]
        self.channel = x_shape[2]
        self.class_n = class_n
        print('use default graph')
        tf.reset_default_graph()
        self.inputs()
        self.graph()
     
    def inputs(self):
        self.x = tf.placeholder(tf.float32,(None,self.width,self.height,self.channel),name='input_x')
        self.y = tf.placeholder(tf.float32,(None,self.class_n),name='input_y')
        self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.is_train = tf.placeholder(tf.bool,name='is_train')
        
    def graph(self):
        print('init graph')
        with tf.variable_scope('MLpconv1'):
            cnn1 = tf.layers.conv2d(self.x,128,3,1,padding='same',activation=tf.nn.relu,name='cnn_layer')
            batch_nor1 = tf.layers.batch_normalization(cnn1,training=self.is_train,name='batch_nor')
            mlp1 = tf.layers.conv2d(batch_nor1,128,1,1,padding='same',activation=tf.nn.relu,name='mlp1')
            mlp2 = tf.layers.conv2d(mlp1,128,1,1,padding='same',activation=tf.nn.relu,name='mlp2')
            pool1 = tf.layers.max_pooling2d(mlp2,2,2,'same')
            
        with tf.variable_scope('MLpconv2'):
            cnn2 = tf.layers.conv2d(pool1,256,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers')
            batch_nor2 = tf.layers.batch_normalization(cnn2,training=self.is_train,name='batch_nor')
            mlp3 = tf.layers.conv2d(cnn2,256,1,1,padding='same',activation=tf.nn.relu,name='mlp1')
            mlp4 = tf.layers.conv2d(mlp3,256,1,1,padding='same',activation=tf.nn.relu,name='mlp2')
            pool2 = tf.layers.max_pooling2d(mlp4,2,2,'same')
            
        with tf.variable_scope('MLpconv3'):
            cnn3 = tf.layers.conv2d(pool2,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers')
            batch_nor3 = tf.layers.batch_normalization(cnn3,training=self.is_train,name='batch_nor')
            mlp5 = tf.layers.conv2d(cnn3,512,1,1,padding='same',activation=tf.nn.relu,name='mlp1')
            mlp6 = tf.layers.conv2d(mlp5,512,1,1,padding='same',activation=tf.nn.relu,name='mlp2')
            pool3 = tf.layers.max_pooling2d(mlp6,2,2,'same')
            
        with tf.variable_scope('MLpconv4'):
            cnn4 = tf.layers.conv2d(pool3,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers1')
            cnn5 = tf.layers.conv2d(cnn4,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers2')
            cnn6 = tf.layers.conv2d(cnn5,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers3')
            
            
        with tf.variable_scope('GAP'):
            raw_cam = tf.nn.avg_pool(cnn6,[1,28,28,1],[1,28,28,1],'SAME',name='reduce_mean')
            cam = tf.reshape(raw_cam,[-1,512],name='reshape')
            
        with tf.variable_scope('drop_out'):
            drop_out = tf.layers.dropout(cam,self.keep_prob,name='')
            
        with tf.variable_scope('fully_connected'):
            logits = tf.layers.dense(drop_out,self.class_n,name='fully_connected')
            outputs = tf.sigmoid(logits,name='outputs')
            
        with tf.variable_scope('error'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=self.y)
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            
        with tf.variable_scope('metrics'):
#             accuracy = tf.metrics.accuracy(tf.argmax(self.y,axis=1),tf.argmax(outputs,axis=1))[1]
            equal = tf.equal(tf.argmax(self.y,axis=1),tf.argmax(outputs,axis=1))
            accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))
              
        self.cam = tf.identity(cnn6,name='cam')
        
        # temp
        self.logits = tf.identity(logits,name='logits')
        
        self.outputs = tf.identity(outputs,name='outputs')
        self.loss = tf.identity(loss,name='loss')
        self.accuracy = tf.identity(accuracy,name='accuracy')
        
    def cv_train(self,X,Y,epoch,batch_size,save_path):
        saver = tf.train.Saver()
        star_time = time.clock()
        step = 0
        with tf.Session() as sess:
            print('init variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            writer = tf.summary.FileWriter('./graph/')
            writer.add_graph(sess.graph)
            step = 0
            start_time = time.clock()
            current_time = start_time
            print('training...')
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            for i in range(epoch):
                kf = KFold(n_splits=10,shuffle=True)
                cv_n = 0
                for train_idx,valid_idx in kf.split(X):
                    cv_n += 1
                    train_x,train_y = X[train_idx],Y[train_idx]
                    valid_x,valid_y = X[valid_idx],Y[valid_idx]
                    
                    train_accuracy = []
                    train_loss = []
                    for x_batch,y_batch in process.get_batches(train_x,train_y,batch_size):
                        step += 1
                        feed = {
                            self.x:x_batch,
                            self.y:y_batch,
                            self.is_train:True,
                            self.learning_rate:0.0001,
                            self.keep_prob:0.5
                        }
                        _,__,accuracy,loss = sess.run([self.optimizer,extra_update_ops,self.accuracy,self.loss],feed_dict=feed)
                        train_accuracy.append(accuracy)
                        train_loss.append(loss)
                        if step % 100 == 0:
                            train_time = time.clock() - current_time
                            current_time = time.clock()
                            valid_accuracy = []
                            valid_loss = []
                            for x_batch,y_batch in process.get_batches(valid_x,valid_y,batch_size):
                                feed = {
                                    self.x:x_batch,
                                    self.y:y_batch,
                                    self.is_train:False,
                                    self.keep_prob:1
                                }
                                accuracy,loss,cross = sess.run([self.accuracy,self.loss,self.logits],feed_dict=feed)
                                valid_accuracy.append(accuracy)
                                valid_loss.append(loss)
                            valid_time = time.clock() - current_time
                            current_time = time.clock()
                            total_time = current_time - start_time
                            train_avg_acc = np.mean(train_accuracy)
                            train_accuracy = []
                            train_avg_loss = np.mean(train_loss)
                            train_loss = []
                            valid_avg_acc = np.mean(valid_accuracy)
                            valid_accuracy = []
                            valid_avg_loss = np.mean(valid_loss)
                            valid_loss = []
                            print("Epoch:{},cv:{},trainAcc:{:.4F},validAcc:{:.4F},trainLoss:{:.6F},validLoss:{:.6F},trainTime:{:.0F},validTime:{:.0F},totalTime:{:.0F}".format(i+1,cv_n,train_avg_acc,valid_avg_acc,train_avg_loss,valid_avg_loss,train_time,valid_time,total_time))
        
                saver.save(sess,save_path,global_step=i+1)
            print('finish')

class Deep_nin(object):
    def __init__(self,x_shape,class_n):
        self.width = x_shape[0]
        self.height = x_shape[1]
        self.channel = x_shape[2]
        self.class_n = class_n
        print('use default graph')
        tf.reset_default_graph()
        self.inputs()
        self.graph()
     
    def inputs(self):
        self.x = tf.placeholder(tf.float32,(None,self.width,self.height,self.channel),name='input_x')
        self.y = tf.placeholder(tf.float32,(None,self.class_n),name='input_y')
        self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.is_train = tf.placeholder(tf.bool,name='is_train')
        

    def graph(self):
        print('init graph')           
        with tf.variable_scope('MLpconv1'):
            cnn1 = tf.layers.conv2d(self.x,128,3,1,padding='same',activation=tf.nn.relu,name='cnn_layer1')
            batch_nor1 = tf.layers.batch_normalization(cnn1,training=self.is_train,name='batch_nor1')
            cnn1_2 = tf.layers.conv2d(batch_nor1,123,3,1,padding='same',activation=tf.nn.relu,name='cnn_layer2')
            batch_nor1_2 = tf.layers.batch_normalization(cnn1_2,training=self.is_train,name='batch_nor2')
            mlp1 = tf.layers.conv2d(batch_nor1_2,128,1,1,padding='same',activation=tf.nn.relu,name='mlp1')
            mlp1_2 = tf.layers.conv2d(mlp1,128,1,1,padding='same',activation=tf.nn.relu,name='mlp2')
            mlp1_3 = tf.layers.conv2d(mlp1_2,128,1,1,padding='same',activation=tf.nn.relu,name='mlp3')
            pool1 = tf.layers.max_pooling2d(mlp1_3,2,2,'same')
            
        with tf.variable_scope('MLpconv2'):
            cnn2 = tf.layers.conv2d(pool1,256,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers1')
            batch_nor2 = tf.layers.batch_normalization(cnn2,training=self.is_train,name='batch_nor1')
            cnn2_2 = tf.layers.conv2d(batch_nor2,256,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers2')
            batch_nor2_2 = tf.layers.batch_normalization(cnn2_2,training=self.is_train,name='batch_nor2')
            mlp2 = tf.layers.conv2d(batch_nor2_2,256,1,1,padding='same',activation=tf.nn.relu,name='mlp1')
            mlp2_2 = tf.layers.conv2d(mlp2,256,1,1,padding='same',activation=tf.nn.relu,name='mlp2')
            mlp2_3 = tf.layers.conv2d(mlp2_2,256,1,1,padding='same',activation=tf.nn.relu,name='mlp3')
            pool2 = tf.layers.max_pooling2d(mlp2_3,2,2,'same')
            
        with tf.variable_scope('MLpconv3'):
            cnn3 = tf.layers.conv2d(pool2,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers1')
            batch_nor3 = tf.layers.batch_normalization(cnn3,training=self.is_train,name='batch_nor1')
            cnn3_2 = tf.layers.conv2d(batch_nor3,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers2')
            batch_nor3_2 = tf.layers.batch_normalization(cnn3_2,training=self.is_train,name='batch_nor2')
            mlp3 = tf.layers.conv2d(batch_nor3_2,512,1,1,padding='same',activation=tf.nn.relu,name='mlp1')
            mlp3_2 = tf.layers.conv2d(mlp3,512,1,1,padding='same',activation=tf.nn.relu,name='mlp2')
            mlp3_3 = tf.layers.conv2d(mlp3_2,512,1,1,padding='same',activation=tf.nn.relu,name='mlp3')
            pool3 = tf.layers.max_pooling2d(mlp3_3,2,2,'same')
            
        with tf.variable_scope('MLpconv4'):
            cnn4 = tf.layers.conv2d(pool3,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers1')
            batch_nor4 = tf.layers.batch_normalization(cnn4,training=self.is_train,name='batch_nor1')
            cnn5 = tf.layers.conv2d(batch_nor4,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers2')
            batch_nor5 = tf.layers.batch_normalization(cnn5,training=self.is_train,name='batch_nor2')
            cnn6 = tf.layers.conv2d(batch_nor5,512,3,1,padding='same',activation=tf.nn.relu,name='cnn_layers3')
            
            
        with tf.variable_scope('GAP'):
            raw_cam = tf.nn.avg_pool(cnn6,[1,28,28,1],[1,28,28,1],'SAME',name='reduce_mean')
            cam = tf.reshape(raw_cam,[-1,512],name='reshape')
            
        with tf.variable_scope('drop_out'):
            drop_out = tf.layers.dropout(cam,self.keep_prob,name='')
            
        with tf.variable_scope('fully_connected'):
            logits = tf.layers.dense(drop_out,self.class_n,name='fully_connected')
            outputs = tf.sigmoid(logits,name='outputs')
            
        with tf.variable_scope('error'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=self.y)
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            
        with tf.variable_scope('metrics'):
#             accuracy = tf.metrics.accuracy(tf.argmax(self.y,axis=1),tf.argmax(outputs,axis=1))[1]
            equal = tf.equal(tf.argmax(self.y,axis=1),tf.argmax(outputs,axis=1))
            accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))
              
        self.cam = tf.identity(cnn6,name='cam')
        
        # temp
        self.logits = tf.identity(logits,name='logits')
        
        self.outputs = tf.identity(outputs,name='outputs')
        self.loss = tf.identity(loss,name='loss')
        self.accuracy = tf.identity(accuracy,name='accuracy')
        
    def cv_train(self,X,Y,epoch,batch_size,save_path):
        saver = tf.train.Saver()
        star_time = time.clock()
        step = 0
        with tf.Session() as sess:
            print('init variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            writer = tf.summary.FileWriter('./graph/')
            writer.add_graph(sess.graph)
            step = 0
            start_time = time.clock()
            current_time = start_time
            print('training...')
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            checkpoint_n = 1
            for i in range(epoch):
                kf = KFold(n_splits=15,shuffle=True)
                cv_n = 0
                for train_idx,valid_idx in kf.split(X):
                    cv_n += 1
                    
                    train_x,train_y = X[train_idx],Y[train_idx]
                    valid_x,valid_y = X[valid_idx],Y[valid_idx]
                    
                    train_accuracy = []
                    train_loss = []
                    for x_batch,y_batch in process.get_batches(train_x,train_y,batch_size):
                        step += 1
                        feed = {
                            self.x:x_batch,
                            self.y:y_batch,
                            self.is_train:True,
                            self.learning_rate:0.0001,
                            self.keep_prob:0.5
                        }
                        _,__,accuracy,loss = sess.run([self.optimizer,extra_update_ops,self.accuracy,self.loss],feed_dict=feed)
                        train_accuracy.append(accuracy)
                        train_loss.append(loss)
                        if step % 2000 == 0:
                            checkpoint_n += 1
                            saver.save(sess,save_path,global_step=checkpoint_n)
                            train_time = time.clock() - current_time
                            current_time = time.clock()
                            valid_accuracy = []
                            valid_loss = []
                            for x_batch,y_batch in process.get_batches(valid_x,valid_y,batch_size):
                                feed = {
                                    self.x:x_batch,
                                    self.y:y_batch,
                                    self.is_train:False,
                                    self.keep_prob:1
                                }
                                accuracy,loss,cross = sess.run([self.accuracy,self.loss,self.logits],feed_dict=feed)
                                valid_accuracy.append(accuracy)
                                valid_loss.append(loss)
                            valid_time = time.clock() - current_time
                            current_time = time.clock()
                            total_time = current_time - start_time
                            train_avg_acc = np.mean(train_accuracy)
                            train_accuracy = []
                            train_avg_loss = np.mean(train_loss)
                            train_loss = []
                            valid_avg_acc = np.mean(valid_accuracy)
                            valid_accuracy = []
                            valid_avg_loss = np.mean(valid_loss)
                            valid_loss = []
                            print("Epoch:{},cv:{},trainAcc:{:.4F},validAcc:{:.4F},trainLoss:{:.6F},validLoss:{:.6F},trainTime:{:.0F},validTime:{:.0F},totalTime:{:.0F}".format(i+1,cv_n,train_avg_acc,valid_avg_acc,train_avg_loss,valid_avg_loss,train_time,valid_time,total_time))
                
            print('finish')

def model_validation(path,x,y,csv_path=None):
    
    # load model
    with tf.Session() as sess:
        meta_path = path + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess,path)
        graph = tf.get_default_graph()
        
        x_ = graph.get_tensor_by_name('input_x:0')
        labels_ = graph.get_tensor_by_name('input_y:0')
        is_train_ = graph.get_tensor_by_name('is_train:0')
        heat_map_ = graph.get_tensor_by_name('cam:0')
        outputs_ = graph.get_tensor_by_name('outputs:0')
        
        fully_para = [v for v in tf.global_variables() if v.name == 'fully_connected/fully_connected/kernel:0'][0]
        para = sess.run(fully_para)
        # forward
        images_arr = []
        heat_maps_arr = []
        outputs_arr = []
        titles_arr = []
        for x_batch,y_batch in process.get_batches(x,y,1):
            feed = {
                x_:x_batch,
                labels_:y_batch,
                is_train_:False,
            }
            heat_maps,outputs = sess.run([heat_map_,outputs_],feed_dict=feed)
            images_arr.extend([item+0.5 for item in x_batch])
            heat_maps_arr.extend(heat_maps)
            outputs_arr.extend(outputs)
            titles_arr.append(process.int_txt_dict[np.argmax(outputs[0])])
            
        # print result
        all_heat_maps = gen_heat_maps(heat_maps_arr,images_arr,outputs_arr,para)

        
        # get error index
        equal = np.equal(np.argmax(outputs_arr,1),np.argmax(y,1))
        error_index = np.where(equal == False)
        
        # get predict correct index
        correct_index = np.where(equal == True)


        error_heat_maps = np.array(all_heat_maps)[error_index]
        error_titles = np.array(titles_arr)[error_index]

        correct_heat_maps = np.array(all_heat_maps)[correct_index]
        correct_titles = np.array(titles_arr)[correct_index]

        
        if csv_path != None:
            test_outputs = []
            test_file_names = []
            
            for test_batch,filename_batch in process.get_test_images(1):
                feed = {
                    x_:test_batch,
                    is_train_:False
                }
                test_output = sess.run(outputs_,feed_dict=feed)
                test_outputs.extend(test_output)
                test_file_names.extend(filename_batch)
            outputs_to_csv(test_outputs,test_file_names,csv_path)

        
        return error_heat_maps,error_titles,correct_heat_maps,correct_titles