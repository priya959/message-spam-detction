from tkinter import *
import tkinter
import pyttsx3
import tkinter.messagebox as tm
from tkinter import messagebox
import tkinter as tk 
from tkinter import Message, Text 
from PIL import Image, ImageTk 
import tkinter.ttk as ttk 
import tkinter.font as font
import datetime 
import time 
engine=pyttsx3.init()
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def module3():
    root=Tk()
    speak("welcome")
    root.title("spam detection")
    speak("start spam detection")
    #root.geometry("1350x760")
    def delete():
        user1.delete(0,tk.END)
    def module5():
        win=Tk()
        #win.after(2000,win.destroy)
        win.geometry("350x200")
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        import matplotlib.pyplot as plt
        #from wordcloud import WordCloud
        from math import log, sqrt
        import pandas as pd
        import numpy as np
        import re

        mails = pd.read_csv('spam1.csv', encoding = 'latin-1')
        mails.head()

        mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
        mails.head()

        mails.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
        mails.head()


        mails['labels'].value_counts()


        mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})
        mails.head()

        mails.drop(['labels'], axis = 1, inplace = True)
        mails.head()

        totalMails = 4825 + 747
        trainIndex, testIndex = list(), list()
        for i in range(mails.shape[0]):
            if np.random.uniform(0, 1) < 0.75:
                trainIndex += [i]
            else:
                testIndex += [i]
        trainData = mails.loc[trainIndex]
        testData = mails.loc[testIndex]


        trainData.reset_index(inplace = True)
        trainData.drop(['index'], axis = 1, inplace = True)
        trainData.head()

        testData.reset_index(inplace = True)
        testData.drop(['index'], axis = 1, inplace = True)
        testData.head()

        trainData['label'].value_counts()

        testData['label'].value_counts()

        trainData.head()


        trainData['label'].value_counts()

        testData.head()

        testData['label'].value_counts()

        def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
            if lower_case:
                message = message.lower()
            words = word_tokenize(message)
            words = [w for w in words if len(w) > 2]
            if gram > 1:
                w = []
                for i in range(len(words) - gram + 1):
                    w += [' '.join(words[i:i + gram])]
                return w
            if stop_words:
                sw = stopwords.words('english')
                words = [word for word in words if word not in sw]
            if stem:
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]   
            return words
        class SpamClassifier(object):
            def __init__(self, trainData, method = 'tf-idf'):
                self.mails, self.labels = trainData['message'], trainData['label']
                self.method = method

            def train(self):
                self.calc_TF_and_IDF()
                if self.method == 'tf-idf':
                    self.calc_TF_IDF()
                else:
                    self.calc_prob()

            def calc_prob(self):
                self.prob_spam = dict()
                self.prob_ham = dict()
                for word in self.tf_spam:
                    self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + \
                                                                        len(list(self.tf_spam.keys())))
                for word in self.tf_ham:
                    self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + \
                                                                        len(list(self.tf_ham.keys())))
                self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 


            def calc_TF_and_IDF(self):
                noOfMessages = self.mails.shape[0]
                self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
                self.total_mails = self.spam_mails + self.ham_mails

                self.spam_words = 0
                self.ham_words = 0
                self.tf_spam = dict()
                self.tf_ham = dict()
                self.idf_spam = dict()
                self.idf_ham = dict()
                for i in range(noOfMessages):
                    message_processed = process_message(self.mails[i])
                    count = list() #To keep track of whether the word has ocured in the message or not.
                                   #For IDF
                    for word in message_processed:
                        if self.labels[i]:
                            self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                            self.spam_words += 1
                        else:
                            self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                            self.ham_words += 1
                        if word not in count:
                            count += [word]
                    for word in count:
                        if self.labels[i]:
                            self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                        else:
                            self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

            def calc_TF_IDF(self):
                self.prob_spam = dict()
                self.prob_ham = dict()
                self.sum_tf_idf_spam = 0
                self.sum_tf_idf_ham = 0
                for word in self.tf_spam:
                    self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails) \
                                                                  / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
                    self.sum_tf_idf_spam += self.prob_spam[word]
                for word in self.tf_spam:
                    self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

                for word in self.tf_ham:
                    self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails) \
                                                                  / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
                    self.sum_tf_idf_ham += self.prob_ham[word]
                for word in self.tf_ham:
                    self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
                    
            
                self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 
                            
            def classify(self, processed_message):
                pSpam, pHam = 0, 0
                for word in processed_message:                
                    if word in self.prob_spam:
                        pSpam += log(self.prob_spam[word])
                    else:
                        if self.method == 'tf-idf':
                            pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                        else:
                            pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
                    if word in self.prob_ham:
                        pHam += log(self.prob_ham[word])
                    else:
                        if self.method == 'tf-idf':
                            pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys()))) 
                        else:
                            pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
                    pSpam += log(self.prob_spam_mail)
                    pHam += log(self.prob_ham_mail)
                return pSpam >= pHam
            
            def predict(self, testData):
                result = dict()
                for (i, message) in enumerate(testData):
                    processed_message = process_message(message)
                    result[i] = int(self.classify(processed_message))
                return result

        def metrics(labels, predictions):
            true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
            for i in range(len(labels)):
                true_pos += int(labels[i] == 1 and predictions[i] == 1)
                true_neg += int(labels[i] == 0 and predictions[i] == 0)
                false_pos += int(labels[i] == 0 and predictions[i] == 1)
                false_neg += int(labels[i] == 1 and predictions[i] == 0)
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            Fscore = 2 * precision * recall / (precision + recall)
            accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

            #print("Precision: ", precision)
            #print("Recall: ", recall)
            #print("F-score: ", Fscore)
            #print("Accuracy: ", accuracy)

        sc_tf_idf = SpamClassifier(trainData, 'tf-idf')
        sc_tf_idf.train()
        preds_tf_idf = sc_tf_idf.predict(testData['message'])
        metrics(testData['label'], preds_tf_idf)

        sc_bow = SpamClassifier(trainData, 'bow')
        sc_bow.train()
        preds_bow = sc_bow.predict(testData['message'])
        metrics(testData['label'], preds_bow)

        '''
        pm = process_message('I cant pick the phone right now. Pls send a message')
        print(sc_tf_idf.classify(pm))
        spam=sc_tf_idf.classify(pm);
        print(spam)
        if(spam):
            print("message is spam");
        else:
            print("message is ham");
        pm1 = process_message('Congratulations ur awarded $500 ')
        print(sc_tf_idf.classify(pm1))
        spam1=sc_tf_idf.classify(pm1);
        print(spam1)
        if(spam1):
            print("message is spam");'''
        message=var1.get()
        pm = process_message(message)
        #print(sc_tf_idf.classify(pm))
        spam=sc_tf_idf.classify(pm);
        #print(spam)
        if(spam):
            #print("message is spam");
            speak("message is spam")
            user=Label(win,text="message is spam",font=("times new roman",30,"bold")).grid(row=0,column=0,padx=10,pady=10)
            win.after(3000,win.destroy)
        else:
            #print("message is ham");
            speak("message is ham")
            user=Label(win,text="message is ham",font=("times new roman",30,"bold")).grid(row=0,column=0,padx=10,pady=10)
            win.after(3000,win.destroy)
        win.mainloop()
    def module4():
        win=Tk()
        win.after(2000,win.destroy)
        win.geometry("200x160")
        message=var1.get()
        user=Label(win,text=message,font=("times new roman",20,"bold"),bg="skyblue").grid(row=0,column=0,padx=10,pady=10)
        win.mainloop()
    img=ImageTk.PhotoImage(Image.open ("C:\\Users\\Priyanka\\Desktop\\final_year_project\\image4.jpg"))
    lab=Label(image=img).pack()
    Login_Frame=Frame(root,bg="skyblue")
    Login_Frame.place(x=135,y=150)
    var1=StringVar()
    #btn_log=Button(Login_Frame,text=" Proccessed..",width=15,command=root.destroy,font=("times new roman",20,"bold"),bg="skyblue",fg="black").grid(row=3,column=0,pady=10)
    u1=Label(Login_Frame,font=("times new roman",20,"bold"),bg="skyblue").grid(row=0,column=0,padx=10,pady=10)
    user=Label(Login_Frame,text=" SPAM DETECTION",font=("times new roman",40,"bold"),bg="skyblue").grid(row=1,column=1,padx=10,pady=10)
    user1=Entry(Login_Frame,width="40",font=("times new roman",20,"bold"),bg="white",textvariable=var1)
    user1.grid(row=2,column=1,padx=25,pady=25,ipadx=25,ipady=100)
    user2=Button(Login_Frame,text="PREDICT",font=("times new roman",15,"bold"),bg="grey",command=module5).grid(row=3,column=1,padx=25,pady=25,ipadx=10,ipady=10)
    user3=Button(Login_Frame,text="CLEAR",font=("times new roman",15,"bold"),bg="grey",command=delete).grid(row=3,column=0,padx=25,pady=25,ipadx=10,ipady=10)
    user4=Button(Login_Frame,text="CANCEL",font=("times new roman",15,"bold"),bg="grey",command=root.destroy).grid(row=3,column=2,padx=25,pady=25,ipadx=10,ipady=10)
    root.mainloop()
class Login_System1:
    def __init__(self,root):
        speak("welcome admin please enter your username and passward")
        self.root=root
        self.root.title("Admin")
        self.root.geometry("1350x740")
        self.bg_icon=ImageTk.PhotoImage(file="C:\\Users\\Priyanka\\Desktop\\final_year_project\\image4.jpg")
        self.username=StringVar()
        self.pass_=StringVar()
        bg_lbl=Label(self.root,image=self.bg_icon).pack()
        Login_Frame=Frame(self.root,bg="skyblue")
        Login_Frame.place(x=450,y=300)
        wel=Label(Login_Frame,text=" ",compound=LEFT,font=("times new roman",20,"bold")
                       ,bg="skyblue").grid(row=1,column=1,padx=20,pady=10)
        username=Label(Login_Frame,text="Username :",compound=LEFT,font=("times new roman",20,"bold")
                       ,bg="skyblue").grid(row=2,column=0,padx=20,pady=10)
        username1=Entry(Login_Frame,textvariable=self.username,bd=5
                        ,relief=GROOVE,font=("",15)).grid(row=2,column=1,padx=20)
        pass1=Label(Login_Frame,text="Password  :",compound=LEFT,font=("times new roman",20,"bold"),
                    bg="skyblue").grid(row=6,column=0,padx=20,pady=10)
        pass2=Entry(Login_Frame,textvariable=self.pass_,bd=5,relief=GROOVE
                    ,show="*",font=("",15)).grid(row=6,column=1,padx=20)
        btn_log=Button(Login_Frame,text="Login",width=15,command=self.login,font=("times new roman",15,"bold")
                      ,bg="skyblue",fg="black").grid(row=8,column=1,pady=10)
    def login(self):
        if self.username.get()=="" or self.pass_.get()=="":
            messagebox.showerror("Error","All field are required!!")
        elif self.username.get()=="priyanka" and self.pass_.get()=="priya":
            messagebox.showinfo("Successfull",f"welcome  {self.username.get()}")
            speak("Thanku so much")
            self.root.after(1000,self.root.destroy)
        else:
            messagebox.showerror("Error","Invalid Username or Password!")
def module1():
    root=Tk()
    speak("welcome")
    root.title("spam detection")
    speak("my project on spam detection")
    root.geometry("1350x740")
    img=ImageTk.PhotoImage(Image.open ("C:\\Users\\Priyanka\\Desktop\\final_year_project\\image4.jpg"))
    lab=Label(image=img).pack()
    Login_Frame=Frame(root,bg="skyblue")
    Login_Frame.place(x=415,y=400)
    user=Label(Login_Frame,text="   SPAM DETECTION ",compound=LEFT,
                   font=("times new roman",45,"bold"),bg="skyblue").grid(row=1,column=0,padx=10,pady=10)
    btn_log=Button(Login_Frame,text=" Proccessed..",width=15,command=root.destroy,
                       font=("times new roman",20,"bold"),bg="skyblue",fg="black").grid(row=3,column=0,pady=10)
    root.mainloop()
#starting of the project :
module1()
root2=Tk()
obj=Login_System1(root2)
root2.mainloop()
module3()
