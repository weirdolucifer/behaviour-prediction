from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("final.csv", delimiter=",",skiprows=1)
# split into input (X) and output (Y) variables
X = dataset[0:,0:14]
Y = to_categorical(dataset[0:,14], num_classes=4)

# create model
model = Sequential()
model.add(Dense(14, input_dim=14, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X, Y, epochs=675, batch_size=10, verbose=1)

#PREDICTION FOR EACH USER
k = 0
ans = [0, 0, 0, 0]
h = 1
while k < 169:
    A = []
    Xnew1 = numpy.array([X[k]])
    Xnew2 = numpy.array([X[k+1]])
    Xnew3 = numpy.array([X[k+2]])
    Xnew4 = numpy.array([X[k+3]])
    Xnew5 = numpy.array([X[k+4]])
    ynew1 = model.predict_classes(Xnew1)
    A.append(ynew1[0])
    ynew2 = model.predict_classes(Xnew2)
    A.append(ynew2[0])
    ynew3 = model.predict_classes(Xnew3)
    A.append(ynew3[0])
    ynew4 = model.predict_classes(Xnew4)
    A.append(ynew4[0])
    ynew5 = model.predict_classes(Xnew5)
    A.append(ynew5[0])
    print("Prediction : %s %s %s %s %s" % (ynew1[0],ynew2[0] ,ynew3[0] ,ynew4[0] ,ynew5[0]))

    counter = 0
    num = A[0] 

    for i in A: 
        curr_frequency = A.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i
    print("Overall Prediction for user %s is : %s" % (h,num))
    ans[num]+= 1
    h = h + 1
    k = k + 5
    
#Histogram of predicted data
import matplotlib.pyplot as plt
clas = ['happy', 'sad', 'angry', 'fear']
plt.bar(clas, ans)
plt.title('histogram')
plt.xlabel('class')
plt.ylabel('test result count')
plt.show('histogram.png')    
 
labels =  ['Relax','Stressed','Partially Stressed','Happy']
colors = ['red','lightsalmon', 'tomato', 'lightcoral']
# Plot
plt.pie(ans, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show('a1.png')