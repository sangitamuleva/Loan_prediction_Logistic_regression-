
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#load csv file
train_file=pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\loan_pred\Train.csv')

test_file=pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\loan_pred\Test.csv')
Loan_ID=test_file.Loan_ID
#Preprocessing

print(train_file.head())
print(train_file.shape)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
print(train_file.describe(include='all'))
#

#Check for a missing values
train_file.isnull().sum()
test_file.isnull().sum()

#Nummerical Variable

#   distribution analyasis of ApplicantIncome has no missing value

#box plot for all numerical data
train_file.boxplot(column='ApplicantIncome')#has outliars and extream values
plt.show()

train_file.boxplot(column='CoapplicantIncome')#has outliars and extream values
plt.show()


train_file.boxplot(column='LoanAmount')#has and outliers
plt.show()

#has missing value
train_file['LoanAmount'].fillna(train_file['LoanAmount'].mean(), inplace=True)

train_file['LoanAmount'].hist(bins=40)
train_file['logLoanAmount']=np.log(train_file['LoanAmount'])

train_file['logLoanAmount'].hist(bins=40)

#histogram for all numerical data

train_file['ApplicantIncome'].hist(bins=40)
train_file['log_ApplicantIncome']= np.log(train_file['ApplicantIncome'])
train_file['log_ApplicantIncome'].hist(bins=40)
train_file['TotalIncome']= train_file['ApplicantIncome']+train_file['CoapplicantIncome']
print(train_file['TotalIncome'])
train_file['log_TotalIncome']= np.log(train_file['TotalIncome'])
train_file['log_TotalIncome'].hist(bins=40)


#Categorical Variable


#male prob high and 13 missing values
train_file['Gender'].value_counts()
train_file['Gender'].fillna('Male',inplace=True)

#married Yes max and 3 missing values
train_file['Married'].value_counts()
train_file['Married'].fillna('Yes',inplace=True)

#self_employed NO max 32 missing value
train_file['Self_Employed'].value_counts()
train_file['Self_Employed'].fillna('No',inplace=True)

#credit history 1 max 50 missing value
train_file['Credit_History'].value_counts()
train_file['Credit_History'].fillna(1.0,inplace=True)

#360  14 missing value
train_file['Loan_Amount_Term'].value_counts()
train_file['Loan_Amount_Term'].fillna(360.0,inplace=True)

#0 max and 15 max values
train_file['Dependents'].value_counts()
train_file['Dependents'].fillna('0',inplace=True)


train_file.isnull().sum()
#label ENcoding

from sklearn import preprocessing

le = {}

for x in ['Loan_ID','Gender', 'Married','Education','Dependents','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']:
    le[x] = preprocessing.LabelEncoder()

for x in ['Loan_ID','Gender','Married','Education','Dependents','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']:
    train_file[x] = le[x].fit_transform(train_file[x])



train_file.dtypes
print(train_file.head(20))

# 0--> No
# 1--> Yes






 #          Test file


#Nummerical Variable

#   distribution analyasis of ApplicantIncome has no missing value

#box plot for all numerical data
test_file.boxplot(column='ApplicantIncome')#has outliars and extream values
plt.show()

test_file.boxplot(column='CoapplicantIncome')#has outliars and extream values
plt.show()


test_file.boxplot(column='LoanAmount')#has and outliers
plt.show()

#has missing value
test_file['LoanAmount'].fillna(test_file['LoanAmount'].mean(), inplace=True)

test_file['LoanAmount'].hist(bins=40)
test_file['logLoanAmount']=np.log(test_file['LoanAmount'])

test_file['logLoanAmount'].hist(bins=40)

#histogram for all numerical data

test_file['ApplicantIncome'].hist(bins=40)
test_file['log_ApplicantIncome']= np.log(test_file['ApplicantIncome'])
print(test_file['log_ApplicantIncome'])
test_file['TotalIncome']= test_file['ApplicantIncome']+test_file['CoapplicantIncome']
print(test_file['TotalIncome'])
test_file['log_TotalIncome']= np.log(test_file['TotalIncome'])
test_file['log_TotalIncome'].hist(bins=40)


#Categorical Variable


#male prob high and 13 missing values
test_file['Gender'].value_counts()
test_file['Gender'].fillna('Male',inplace=True)

#married Yes max and 3 missing values
test_file['Married'].value_counts()
test_file['Married'].fillna('Yes',inplace=True)

#self_employed NO max 32 missing value
test_file['Self_Employed'].value_counts()
test_file['Self_Employed'].fillna('No',inplace=True)

#credit history 1 max 50 missing value
test_file['Credit_History'].value_counts()
test_file['Credit_History'].fillna(1.0,inplace=True)

#360  14 missing value
test_file['Loan_Amount_Term'].value_counts()
test_file['Loan_Amount_Term'].fillna(360.0,inplace=True)

#0 max and 15 max values
test_file['Dependents'].value_counts()
test_file['Dependents'].fillna('0',inplace=True)


test_file.isnull().sum()
#label ENcoding

from sklearn import preprocessing

le = {}

for x in ['Loan_ID','Gender', 'Married','Education','Dependents','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area']:
    le[x] = preprocessing.LabelEncoder()

for x in ['Loan_ID','Gender','Married','Education','Dependents','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area']:
    test_file[x] = le[x].fit_transform(test_file[x])



test_file.dtypes
print(train_file.head(20))

# 0--> No
# 1--> Yes



colname=['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','log_TotalIncome','logLoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
train=train_file[colname]
target=train_file.Loan_Status

test=test_file[colname]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train)
train= scaler.transform(train)
print(train)


scaler.fit(test)
test= scaler.transform(test)


from sklearn.model_selection import train_test_split
# Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(train,target, test_size=0.3,
                                                    random_state=10)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_test_predict=lr.predict(X_test)
print(list(zip(Y_test, Y_test_predict)))


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(Y_test, Y_test_predict)

print("Confusion matrix")
print(cfm)

print("Classification report: ")
print(classification_report(Y_test, Y_test_predict))

acc = accuracy_score(Y_test, Y_test_predict)
print("Accuracy of the model: ", acc)



from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

r2score=r2_score(Y_test,Y_test_predict)
print('r square score:',r2score)

rmse=np.sqrt(mean_squared_error(Y_test,Y_test_predict))
print('rmse:',rmse)


target_predict=lr.predict(test)
print(target_predict.dtype)
df=pd.DataFrame(target_predict,index=test_file.Loan_ID,columns=['Loan_Status'])
print(df)
df.to_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\loan_pred\OutputFile.csv')
