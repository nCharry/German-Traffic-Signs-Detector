from __future__ import print_function
import click
import shutil
import time
import sys
import os
import zipfile
import numpy as np
import cv2
import pickle
from urllib.request import urlretrieve
sys.path.append('../')
import matplotlib.pyplot as plt
import tensorflow as tf



#Dictionay Traffic Signs

dic={
0 : "speed limit 20 (prohibitory)",1 : "speed limit 30 (prohibitory)",2 : "speed limit 50 (prohibitory)",
3 : "speed limit 60 (prohibitory)",4 : "speed limit 70 (prohibitory)",5 : "speed limit 80 (prohibitory)",
6 : "restriction ends 80 (other)",7 : "speed limit 100 (prohibitory)",8 : "speed limit 120 (prohibitory)",
9 : "no overtaking (prohibitory)",10 : "no overtaking (trucks) (prohibitory)",11 : "priority at next intersection (danger)",
12 : "priority road (other)",13 : "give way (other)",14 : "stop (other)",15 : "no traffic both ways (prohibitory)",
16 : "no trucks (prohibitory)",17 : "no entry (other)",18 : "danger (danger)",19 : "bend left (danger)",
20 : "bend right (danger)",21 : "bend (danger)",22 : "uneven road (danger)",23 : "slippery road (danger)",
24 : "road narrows (danger)",25 : "construction (danger)",26 : "traffic signal (danger)",27 : "pedestrian crossing (danger)",
28 : "school crossing (danger)",29 : "cycles crossing (danger)",30 : "snow (danger)",31 : "animals (danger)",
32 : "restriction ends (other)",33 : "go right (mandatory)",34 : "go left (mandatory)",35 : "go straight (mandatory)",
36 : "go right or straight (mandatory)",37 : "go left or straight (mandatory)",38 : "keep right (mandatory)",
39 : "keep left (mandatory)",40 : "roundabout (mandatory)",41 : "restriction ends (overtaking) (other)",
42 : "restriction ends (overtaking (trucks)) (other)"
}


#Some Functions


def infer_New_images(path,gray):

    img_size=32
    if gray==True:
        num_channels=1
    else:
        num_channels=3
    
    index=0
    image_files = os.listdir(path)
    Total=len(image_files)-1
    x=np.zeros([Total,img_size*img_size*num_channels])
    for image in image_files:
        image_file = os.path.join(path, image)
        if image[-2:] !="md": 
            img = cv2.imread(image_file)
            img_resize=cv2.resize(img ,(img_size,img_size), interpolation = cv2.INTER_AREA)
            if gray==True:
               img_resize= cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
            img_reshape=img_resize.reshape([1,img_size*img_size*num_channels])
            x[index]=img_reshape
            index+=1
      
    return x


def AppPredicted(x,predicted,gray):
    
    for i,j in enumerate(x):
    
        img=j
        fig=plt.figure(figsize=(3,3))
        if gray==True:
           img=img.reshape(32,32)
        else:
            img=img.reshape(32,32,3)
        img = img.astype(np.uint8)
        d = fig.add_subplot(1,1, 1, xticks=[], yticks=[])
        d.text(0,10,str(predicted[i]),color="red",fontsize=25)
        d.text(1, 25, dic[predicted[i]], style='oblique',fontsize=11,
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
        d.imshow(img, cmap=plt.cm.bone)
        plt.waitforbuttonpress()
        
                       
           
    plt.close('all')

def features_vectors(path,gray):
    
    if os.listdir(path)==[]:
       click.echo("The Folder is empty")
    else:
    
        img_size=32
        if gray==True:
            num_channels=1
        else:
            num_channels=3
        
        list_features=[]
        
        for folder_name,dirs,file in os.walk(path):
            list_features.append(len(file))
        Total=sum(list_features)
    
        x=np.zeros([Total,img_size*img_size*num_channels])
        target=np.zeros(Total)
        root=[]
        index=0
        for folder_name,dirs,file in os.walk(path):
            root.append(folder_name)
            if folder_name!=root[0]:
                image_files = os.listdir(folder_name)
                for image in image_files:
                    image_file = os.path.join(folder_name, image)
                    img = cv2.imread(image_file)
                    if gray==True:
                        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_reshape=img.reshape([1,img_size*img_size*num_channels])
                    x[index]=img_reshape
                    target[index]=folder_name[-2:]  
                    if index<Total:
                        index+=1
    return x,target      
def reshape_cnn(x):
    image_size=32
    num_channels=1
    Total=x.shape[0]
    
    X=np.zeros([Total,image_size,image_size,num_channels])
    for i,img in enumerate(x):
        img_reshape=img.reshape(32,32,1)
        X[i]=img_reshape
        
    return X

def preprocessing_data(x,y,one_hot):
    shape=x.shape
    if shape[1:]==(32, 32, 1):
        from sklearn.preprocessing import OneHotEncoder

        onehot_encoder = OneHotEncoder(sparse=False)  
        y= y.reshape(len(y), 1) 
        y= onehot_encoder.fit_transform(y)
    else:
               
        #Shuffle Data
        from sklearn.utils import shuffle
        
        x, y = shuffle(x, y, random_state=100)
        #Scale Features
        from sklearn.preprocessing import StandardScaler
    
        sc = StandardScaler()
        x=sc.fit_transform(x)
        
        #One Hot Encoder Target vector
        
        if one_hot==True:
        
    
            from sklearn.preprocessing import OneHotEncoder
    
            onehot_encoder = OneHotEncoder(sparse=False)  
            y= y.reshape(len(y), 1) 
            y= onehot_encoder.fit_transform(y)
    
    return x,y        


#Models
    
    
def model1(x,y):
    
    x,y=preprocessing_data(x,y,False)
    
    from sklearn.linear_model import LogisticRegression
    lm = LogisticRegression(penalty = 'l2',C =0.1)
    lm.fit(x, y)
    if os.path.isfile('models/model1/saved/LogisticRegression.sav')==False:
        filename = 'models/model1/saved/LogisticRegression.sav'
        pickle.dump(lm, open(filename, 'wb'))
    else:
        filename = 'models/model1/saved/LogisticRegression.sav'
        os.remove(filename)
        pickle.dump(lm, open(filename, 'wb'))
        
    
    
    return lm

def model2(x,y):
    
    
    batch_size = 97


    # Input data.
    tf.reset_default_graph()
    X_Train = tf.placeholder(tf.float32,shape=(None, x.shape[1]),name="x")
    Y_Train = tf.placeholder(tf.float32, shape=(None, y.shape[1]))
    #X_Test = tf.constant(x_test)
    
    # Convert the features vector from  tf.float64 to tf.float32 
    X_Train = tf.cast(X_Train, tf.float32)
    Y_Train = tf.cast(Y_Train, tf.float32)
    #X_Test = tf.cast(X_Test, tf.float32)
    
    # Logistic Regression 
    
    w = tf.Variable(tf.truncated_normal(shape=[x.shape[1], y.shape[1]]))
    b= tf.Variable(tf.zeros(shape=[y.shape[1]]))
    out = tf.matmul(X_Train, w) + b
    
    # Regularization l2
    
    logits = tf.matmul(X_Train, w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_Train))
    loss = tf.reduce_mean(loss + 0.001 * tf.nn.l2_loss(w) )
    
    
    # Optimizer Gradient Descent
    
    optimizer=tf.train.AdamOptimizer().minimize(loss)
    
    
    # Predictions 
    train_prediction = tf.nn.softmax(out)
    
    prediction = tf.argmax(train_prediction, 1, name='predict')
    #test_prediction = tf.nn.softmax(tf.matmul(X_Test, w) + b)
    saver = tf.train.Saver()

    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0]) 
    
    epoch = 8000
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(epoch):
       
            offset = (step * batch_size) % (y.shape[0] - batch_size)
    
            batch_data = x[offset:(offset + batch_size), :]
            batch_labels = y[offset:(offset + batch_size), :]
    
            feed_dict = {X_Train : batch_data, Y_Train : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("loss %d: %f" % (step, l))
                print("Accuracy batch train data: %.1f%%" % accuracy(predictions, batch_labels))
    
          
        saver.save(session, 'models/model2/saved/LogisticRegression.ckpt')
        print("Model saved")  
        
def model3(x,y):
    
    num_labels=43
    image_size=32
    batch_size = 97
    patch_size1=5
    patch_size2=5
    num_channels=1
    Num_Filters1=6
    Num_Filters2=16
    num_hidden1=120
    num_hidden2=84
    
   
    
    # Input data.
    
    tf.reset_default_graph()       
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels),name="x")
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    #tf_test_dataset = tf.constant(X_test)
    
    tf_train_dataset = tf.cast(tf_train_dataset, tf.float32)
    tf_train_labels = tf.cast(tf_train_labels, tf.float32)
    #tf_test_dataset = tf.cast(tf_test_dataset, tf.float32)
    
      
    mu = 0
    sigma = 0.1

    # L1 Convolutional. in= 32x32x1 / out = 28x28x6.
    Shape_C1=[patch_size1,patch_size1,num_channels,Num_Filters1]
    weigths_C1=tf.Variable(tf.truncated_normal(Shape_C1,mean = mu, stddev = sigma))
    bias_C1=tf.Variable(tf.zeros(Num_Filters1))

    C1=tf.nn.conv2d(tf_train_dataset,weigths_C1,strides=[1, 1, 1, 1],padding='VALID')
    C1+=bias_C1 
    C1=tf.nn.sigmoid(C1)

    #SubSampling  out 14*14*6

    S2=tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # L2: Convolutional. Output = 10x10x16.

    Shape_C2=[patch_size2,patch_size2,Num_Filters1,Num_Filters2]
    weigths_C2=tf.Variable(tf.truncated_normal(Shape_C2, mean = mu, stddev = sigma))
    bias_C2=tf.Variable(tf.zeros(Num_Filters2))

    C2=tf.nn.conv2d(S2,weigths_C2,strides=[1, 1, 1, 1],padding='VALID')
    C2+=bias_C2 
    C2=tf.nn.relu(C2)

    S4=tf.nn.max_pool(C2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

     #Flatten Layer
    from tensorflow.contrib.layers import flatten
    flatten_layer=flatten(S4)

    Shape_Fc1=[400,num_hidden1]
    weights_Fc1=tf.Variable(tf.truncated_normal(Shape_Fc1,stddev=0.05))
    bias_Fc1=tf.Variable(tf.zeros(num_hidden1))
    Fc1=tf.nn.relu(tf.matmul(flatten_layer,weights_Fc1))+bias_Fc1

    #Fully Connected Layer 2

    Shape_Fc2=[num_hidden1,num_hidden2]
    weights_Fc2=tf.Variable(tf.truncated_normal(Shape_Fc2,stddev=0.05))
    bias_Fc2=tf.Variable(tf.zeros(num_hidden2))
    Fc2=tf.nn.relu(tf.matmul(Fc1,weights_Fc2))+bias_Fc2

     #Prediction Layer
    Shape_Pl=[num_hidden2,num_labels]
    weights_Pl=tf.Variable(tf.truncated_normal(Shape_Pl,stddev=0.05))
    bias_Pl=tf.Variable(tf.zeros(num_labels))
    out=tf.matmul(Fc2,weights_Pl)+bias_Pl
  
    
      
    logits = out
    prediction=tf.nn.softmax(logits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    prediction = tf.argmax(train_prediction, 1, name='predict')
    #test_prediction=model(tf_test_dataset)
    saver = tf.train.Saver()
        
        
        
    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])
    
    
    
    
    num_steps = 3000
    
    with tf.Session() as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          for step in range(num_steps):
            offset = (step * batch_size) % (y.shape[0] - batch_size)
            batch_data =x[offset:(offset + batch_size), :, :, :]
            batch_labels = y[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 1000 == 0):
              print('Minibatch loss at step %d: %f' % (step, l))
              print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    
          saver.save(session, 'models/model3/saved/LeNet.ckpt')
          print("Model saved")       
    
    

@click.group()
def cli():
    pass

@cli.command()
def download():

    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    click.echo('Wait a few seconds while loading the libraries')
    if os.path.isfile(os.path.join(os.getcwd(),"images","dataset.zip")):
       click.echo('The dataset has already been downloaded')
    else:
        url = "http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip"
        dst = "images/dataset.zip"
        click.echo('Start Download Dataset')
        urlretrieve(url, dst, reporthook)
        click.echo('The dataset download has finished')

        click.echo('Move files to train and test folders')

        def move_folders(path,data,target):         
            for j,i in enumerate(target):
                img=data[j]
                for folder_name,dirs,file in os.walk("images/FullIJCNN2013"):
                    if os.path.isdir(os.path.join(path,'0'+ str(int(i)))):
                        cv2.imwrite(os.path.join(path,'0'+str(int(i)),str(int(j))+'.ppm'),img)
                    else:
                        os.mkdir(os.path.join(path, '0'+str(int(i))))
                        cv2.imwrite(os.path.join(path,'0'+str(int(i)),str(int(j))+'.ppm'),img)
         


        archive = zipfile.ZipFile('images/dataset.zip')
        path=os.path.join(os.getcwd(),"images")
        files=[]
        for file in archive.namelist():
            files.append(file)
            if (file[-3:]=="ppm")and(file!="FullIJCNN2013/")and(len(file)!=23):
                archive.extract(file, path)

        #Data Explore

        folders=[]
        files=[]

        for folder_name,dirs,file in os.walk("images/FullIJCNN2013"):
            folders.append(folder_name)
            files.append(len(file))
        Total=sum(files)     
          
        #Create Matrix X and targets, these contain the images in format RGB in X and targets, resized 32x32 

        img_size=32
        num_channels=3
        X=np.zeros([Total,img_size,img_size,num_channels])
        target=np.zeros(Total)
        index=0

        for folder_name,dirs,file in os.walk("images/FullIJCNN2013"):
            
            if folder_name!=("images/FullIJCNN2013"):
               image_files = os.listdir(folder_name)
               for  image in image_files:
                    image_file = os.path.join(folder_name, image)
                    img = cv2.imread(image_file)
                    resized = cv2.resize(img ,(32,32), interpolation = cv2.INTER_AREA)
                    X[index]=resized[np.newaxis]
                    target[index]=folder_name[-2:]  
                    if index<Total:
                       index+=1
                       
        # Split Data in train and test data                
                       
        from sklearn.model_selection import train_test_split 

        X_train, x_test, y_train, y_test = train_test_split(X, target,test_size=0.2,random_state=100)

        # Move to folders

        #Move to train folder

        move_folders('images/train',X_train,y_train)

        # Move to test folder

        move_folders('images/test',x_test,y_test)
          
                 
        shutil.rmtree('images/FullIJCNN2013')
        
        
        
        click.echo('The train and test folders contain the resized 32x32 images with which the feature vectors will be formed') 


MODEL = ['model1', 'model2', 'model3']
DATA = ['images/train','images/test','images/user']

@cli.command()
@click.option('-m', type=click.Choice(MODEL), help='Select a model')
@click.option('-d', type=click.Choice(DATA), default="train", help='Select train o test data')

def train(m,d):
    
    """Train a Model Selected"""
    if (m=="model1"):       
        
        click.echo("Logistic Regression Sklearn-Data Train")
        
        if d=="images/train":
           path=d
           x_train,y_train=features_vectors(path,False)
           click.echo("Training logistic regression model")
           lm=model1(x_train,y_train)
           predic= lm.predict(x_train)
           matches_train = (predic == y_train)
           print("Accuracy = ",matches_train.sum()/ float(len(matches_train))*100)
           
        else:
             click.echo("Enter -d images/train for train model and train accuracy")
           
           
           
           
       
    elif (m=="model2"):
        
        click.echo("Logistic Regression Tensorflow-Data Train")
        
        if d=="images/train":
           path=d
           x_train,y_trainx=features_vectors(path,True)
           x_train,y_train=preprocessing_data(x_train,y_trainx,True)
           
           click.echo("Training logistic regression model")
           
           model2(x_train,y_train)

            
           sess = tf.Session()
           saver = tf.train.import_meta_graph('models/model2/saved/LogisticRegression.ckpt.meta')
           saver.restore(sess, tf.train.latest_checkpoint('models/model2/saved'))
           graph = tf.get_default_graph()
           x = graph.get_tensor_by_name("x:0")
           prediction = graph.get_tensor_by_name("predict:0")
           feed_dict = {x: x_train}
           Prediction_restored=sess.run([prediction],feed_dict=feed_dict)
           matches_train=(Prediction_restored[0]==np.argmax(y_train,1))
           accuracy_train=(sum(matches_train)/y_train.shape[0])*100
           print("Accuracy = ",accuracy_train)
           
        else:
             click.echo("Enter -d images/train for train model and train accuracy")
           
          
            
    elif (m=="model3"):
        
        click.echo("LeNet-Data Train")
        
        if d=="images/train":
           path=d
           x_train,y_train=features_vectors(path,True)
           x_train=reshape_cnn(x_train)
           x_train,y_train=preprocessing_data(x_train,y_train,True)
           from sklearn.utils import shuffle
            
           x_train, y_train = shuffle(x_train, y_train, random_state=100)           
           
           click.echo("Training LeNet model")
           
           model3(x_train,y_train)
           
           
           sess = tf.Session()
           saver = tf.train.import_meta_graph('models/model3/saved/LeNet.ckpt.meta')
           saver.restore(sess, tf.train.latest_checkpoint('models/model3/saved'))
           graph = tf.get_default_graph()
           x = graph.get_tensor_by_name("x:0")
           prediction = graph.get_tensor_by_name("predict:0")
           feed_dict = {x: x_train}
           Prediction_restored=sess.run([prediction],feed_dict=feed_dict)
           matches_train=(Prediction_restored[0]==np.argmax(y_train,1))
           accuracy_train=(sum(matches_train)/y_train.shape[0])*100
           print("Accuracy = ",accuracy_train)
           
        else:
            click.echo("Enter -d images/train for train model and train accuracy")
        
            
@cli.command()
@click.option('-m', type=click.Choice(MODEL), help='Select a model')
@click.option('-d', type=click.Choice(DATA), default="train", help='Select train o test data')

def test(m,d):
    
    """Test the accuracy a model selected"""
    if (m=="model1"):      
              
        click.echo("Logistic Regression Sklearn-Data Test")
        
        if d=="images/test":
           path=d
           x_test,y_test=features_vectors(path,False)
           
           
           filename = 'models/model1/saved/LogisticRegression.sav'
           loaded_model = pickle.load(open(filename, 'rb'))
           click.echo("Regression Logictic model has been loaded")
           score = loaded_model.score(x_test, y_test)
           print("Accuracy = ",score)
           
                        
           
        else:
             click.echo("Enter -d images/test for test accuracy")
             
           
           
       
            
    elif (m=="model2"):
        
        click.echo("Logistic Regression Tensorflow-Data Test")
        
        if d=="images/test":
            path=d
            x_test,y_test=features_vectors(path,True)

            x_test,y_test=preprocessing_data(x_test,y_test,True)
            
            sess = tf.Session()
            saver = tf.train.import_meta_graph('models/model2/saved/LogisticRegression.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('models/model2/saved'))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            prediction = graph.get_tensor_by_name("predict:0")
            feed_dict = {x: x_test}
            Prediction_restored=sess.run([prediction],feed_dict=feed_dict)
            matches_test=(Prediction_restored[0]==np.argmax(y_test,1))
            accuracy_test=(sum(matches_test)/y_test.shape[0])*100
            print("Accuracy =",accuracy_test)
               
        else:
            click.echo("Enter -d images/test for test accuracy")
            
    elif (m=="model3"):
        
        click.echo("LeNet Tensorflow-Data Test")
        
        if d=="images/test":
            path=d

            x_test,y_test=features_vectors(path,True)
            x_test=reshape_cnn(x_test)
            x_test,y_test=preprocessing_data(x_test,y_test,True)
            from sklearn.utils import shuffle
            
            x_test, y_test = shuffle(x_test, y_test, random_state=100)
            
            
            sess = tf.Session()
            saver = tf.train.import_meta_graph('models/model3/saved/LeNet.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('models/model3/saved'))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            prediction = graph.get_tensor_by_name("predict:0")
            feed_dict = {x: x_test}
            Prediction_restored=sess.run([prediction],feed_dict=feed_dict)
            matches_test=(Prediction_restored[0]==np.argmax(y_test,1))
            accuracy_test=(sum(matches_test)/y_test.shape[0])*100
            print("Accuracy =",accuracy_test)


        else:
            click.echo("Enter -d images/test for test accuracy")
            
            
@cli.command()
@click.option('-m', type=click.Choice(MODEL), help='Select a model')
@click.option('-d', type=click.Choice(DATA), default="train", help='Select train o test data')

def infer(m,d):
    
    """Aplication Image with any model"""
    if (m=="model1"):       
        
        click.echo("Logistic Regression Sklearn-Data Test")
        
        if d=="images/user":
           click.echo("Infer with User data")
           path=d
           x_user=infer_New_images(path,False)
              
           filename = 'models/model1/saved/LogisticRegression.sav'
           loaded_model = pickle.load(open(filename, 'rb'))
           click.echo("Regression Logictic model has been loaded")
           predic= loaded_model.predict(x_user)
          
           click.echo("Press any key to continue showing the predictions") 
           AppPredicted(x_user,predic,False) 
           
        else:
            click.echo("Enter -d images/user for infer app")
            
    elif (m=="model2"):
        
        click.echo("Logistic Regression Sklearn-Tensorflow")
        
        
        if d=="images/user":
           click.echo("Infer with User data")
           path=d
           x_user=infer_New_images(path,True)
            
           sess = tf.Session()
           saver = tf.train.import_meta_graph('models/model2/saved/LogisticRegression.ckpt.meta')
           saver.restore(sess, tf.train.latest_checkpoint('models/model2/saved'))
           graph = tf.get_default_graph()
           x = graph.get_tensor_by_name("x:0")
           prediction = graph.get_tensor_by_name("predict:0")
           feed_dict = {x: x_user}
           Prediction_restored=sess.run([prediction],feed_dict=feed_dict)
           predic=Prediction_restored[0]
           AppPredicted(x_user,predic,True)           

        else:
            click.echo("Enter -d images/user for infer app")
             
            
    elif (m=="model3"):
        
        click.echo("LeNet-Tensorflow") 
        
        if d=="images/user":
            click.echo("Infer with User data")
            path=d
            x_user=infer_New_images(path,True)
            x_user=reshape_cnn(x_user)
            sess = tf.Session()
            saver = tf.train.import_meta_graph('models/model3/saved/LeNet.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('models/model3/saved'))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            prediction = graph.get_tensor_by_name("predict:0")
            feed_dict = {x: x_user}
            Prediction_restored=sess.run([prediction],feed_dict=feed_dict)
            predic=Prediction_restored[0]
            AppPredicted(x_user,predic,True) 
           
        else:
            click.echo("Enter -d images/user for infer app")
       
   
        

    

if __name__ == '__main__':
    cli()




