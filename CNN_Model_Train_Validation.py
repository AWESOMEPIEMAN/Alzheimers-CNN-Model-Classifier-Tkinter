import tensorflow as tf
from tensorflow import keras 
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
from tkinter import *
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Input
from tensorflow.python.ops.gen_math_ops import Sub


#Function called after Epoch value is entered through Entry box abd button is pressed    
def StartBuildingModel():
    input_valy = int(button_entry.get())
    print(input_valy)
    Local_dir = "alzdataset/"

    train_set = ImageDataGenerator(rescale=1./255)
    test_set =  ImageDataGenerator(rescale=1./255)
    validation_set =  ImageDataGenerator(rescale=1./255)

    train_set=Local_dir+ 'train/'

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_set,
        validation_split=0.2,
        image_size=(224,224),
        batch_size=32,
        subset='training',
        seed=1000 )

    validation_set=Local_dir+ 'train/'

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        validation_set,
        validation_split=0.2,
        image_size=(224,224),
        batch_size=32,
        subset='validation',
        seed=1000
        )
    test_set=Local_dir +'test/'

    test_data=tf.keras.preprocessing.image_dataset_from_directory(
        test_set,
        image_size=(224,224),
        batch_size=32,
        seed=1000
        )
    class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
    train_data.class_names = class_names
    val_data.class_names = class_names
    print(val_data)

    model=Sequential()

    model.add(Conv2D(16,(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(4,activation='softmax'))


    model.summary()

    model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, validation_data=val_data, epochs=input_valy)
    evaluation = model.evaluate(test_data)
    evaluation = str(evaluation[1])
    
    

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    plt2.plot(loss_train, 'g', label='Training loss')
    plt2.plot(loss_val, 'b', label='validation loss')
    plt2.title('Training and Validation loss')
    plt2.xlabel('Epochs')
    plt2.ylabel('Loss')
    plt2.legend()
    plt2.show()

    accuracy_train = history.history['accuracy']
    accuracy_val = history.history['val_accuracy']
    plt3.plot(accuracy_train, 'g', label='Training accuracy')
    plt3.plot(accuracy_val, 'b', label='Validation accuracy')
    plt3.title('Training and Validation accuracy')
    plt3.xlabel('Epochs')
    plt3.ylabel('Accuracy')
    plt3.legend()
    plt3.show()

    class_names={0:"Mild Dementia", 1:"Moderate Dementia", 2:"Non Dementia", 3:"Very Mild Dementia"}

    plt4.figure(figsize=(18,8))
    for images, labels in val_data.take(1):
        for i in range(10):
            plt4.subplot(2,5,i+1)
            plt4.imshow(images[i]/255)
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            x = image.img_to_array(images[i])
            x = np.expand_dims(x, axis=0)
            p=np.argmax(model.predict(x))    
            plt4.title("Prediction: {}".format(class_names[p]))
            plt4.xlabel("True value: {}".format(val_data.class_names[labels[i]]))
    


###########################################################################    
#Sub window code for the project to display accuracy and model prediction.#    
###########################################################################
    SubWindow = Tk()
    SubWindow.title("Results of the model")

    SubWindow.minsize(500,200)
    SubWindow.maxsize(500,200)

    bg = ImageTk.PhotoImage(Image.open("assets/bg1.jpg"),master=SubWindow)
        #Creating label
    label1 = Label(SubWindow, image = bg)
    label1.place(x=-500, y=0)
    label2 = Label(SubWindow,text ="The Accuracy of this model is {}".format(evaluation),font="none 11 bold",bg="white")
    label2.place(x=110,y=20)
    label3 = Label(SubWindow,text ="View final predictions alongside actual state of the image ",font="none 11 bold",bg="white")
    label3.place(x=50,y=100)
    Button(SubWindow,text="View",font="none 10 bold",bg="white",width="5",command=lambda:plt4.show()).place(relx=0.5,rely=0.75,anchor="center")
    SubWindow.mainloop()



####################################################
#This is the main window code for the project.#    
####################################################

#Setting up the window to be used for the main screen 
MainWindow = Tk()
MainWindow.geometry("700x500")
MainWindow.minsize(700,500)
MainWindow.maxsize(700,500)
MainWindow.title("Alzheimers Classifier based on CNN model by Usman Hamid")
bg = ImageTk.PhotoImage(Image.open("assets/bg1.jpg"))

#Creating the label ,Button and entry widgets for the Epochs value
label1 = Label(MainWindow, image = bg)
label1.place(x=-390, y=0)
label_intro = Label (MainWindow,text="Welcome to the prototype for the CNN based Alzheimer's classifier",font="none 15 bold",bg="white")
label_intro.place(relx=0.5,rely=0.2,anchor="center")
label_info = Label (MainWindow,text="This prototype is still in development so please be patient when the model is being built. The program hasn't crashed :)",
    font="none 15 bold",bg="white",wraplength=600, justify="center")
label_info.place(relx=0.5,rely=0.3,anchor="center")
label_input = Label (MainWindow,text="Since this project is in its bare intitial phases, it will only accept Epoch value for your dataset (You can use your own dataset by adding files in the directory with the same name)",
    font="none 15 bold",bg="white",wraplength=600, justify="center")
label_input.place(relx=0.5,rely=0.45,anchor="center")
label_button = Label (MainWindow,text="Press the button below to start building the model. Check the terminal window to see your model being built",
    font="none 15 bold",bg="white",wraplength=600, justify="center")
label_button.place(relx=0.5,rely=0.6,anchor="center")
label_eppoch = Label (MainWindow,text="Enter Epochs value : ",
    font="none 15 bold",bg="white",wraplength=600, justify="center")
label_eppoch.place(relx=0.2,rely=0.75,anchor="center")

button_entry = Entry (MainWindow,width=20,bg="white")
button_entry.place(relx=0.41,rely=0.73)


Button (MainWindow,text="Start",font="none 15 bold",bg="white",width="6",command=lambda:threading.Thread(target=StartBuildingModel).start()).place(relx=0.5,rely=0.9,anchor="center")

MainWindow.mainloop()
