import pandas as pd
import numpy as np



#load csv file
data=pd.read_csv(r'C:\Users\Admin\Contacts\Desktop\sangita\loan_pred\Train.csv')
print(data.head())
print(data.shape)


#614 rows  and 13 column
#predict Loan Status
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
print(data.describe(include='all'))

#Check for a missing values

#print all missing values count
data.isnull().sum()

#copy data frame
New_data=pd.DataFrame.copy(data)
New_data.describe(include='all')

#Preprocessing


#distribution analysis
New_data.hist(bins=50)

#box plot for all numerical data
import matplotlib.pyplot as plt
New_data.boxplot() #for plotting boxplots for all the numerical columns
plt.show()

#has outliars
New_data.boxplot(column='ApplicantIncome')
plt.show()

#has missing values and outliers
New_data.boxplot(column='LoanAmount')
plt.show()


#impute outliars for ApplicantIncome Variable
#for defining acceptable range
q1 = New_data['ApplicantIncome'].quantile(0.25) #first quartile value
q3 = New_data['ApplicantIncome'].quantile(0.75) # third quartile value
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range

df_appincom_include = New_data.loc[(New_data['ApplicantIncome'] >= low) & (New_data['ApplicantIncome'] <= high)] # meeting the acceptable range
df_appincom_exclude = New_data.loc[(New_data['ApplicantIncome'] < low) | (New_data['ApplicantIncome'] > high)] #not meeting the acceptable range

#outliars
print(df_appincom_exclude.shape)

print(df_appincom_include.shape)

#mean of acceptable range
Applicant_Income_mean=int(df_appincom_include.ApplicantIncome.mean()) #finding the mean of the acceptable range
print(Applicant_Income_mean)

#imputing outlier values with mean value
df_appincom_exclude.ApplicantIncome=Applicant_Income_mean
print(df_appincom_exclude.shape)


#getting back the original shape of df
New_df_rev=pd.concat([df_appincom_include,df_appincom_exclude]) #concatenating both dfs to get the original shape
print(New_df_rev.shape)
print(New_df_rev.head(10))


New_df_rev.boxplot(column='ApplicantIncome')
plt.show()








#add Missing values

New_data['LoanAmount'].fillna(New_data['LoanAmount'].mean(), inplace=True)
New_data.isnull().sum()

#impute outliars for LoanAmount Variable
#for defining acceptable range

q1 = New_data['LoanAmount'].quantile(0.25) #first quartile value
q3 = New_data['LoanAmount'].quantile(0.75) # third quartile value
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range

df_loanAmt_include = New_data.loc[(New_data['LoanAmount'] >= low) & (New_data['LoanAmount'] <= high)] # meeting the acceptable range
df_loanAmt_exclude = New_data.loc[(New_data['LoanAmount'] < low) | (New_data['LoanAmount'] > high)] #not meeting the acceptable range

#39 outliars
print(df_loanAmt_exclude.shape)
print(df_loanAmt_include.shape)

#mean of acceptable range
Loan_Amount_mean=int(df_loanAmt_include.LoanAmount.mean()) #finding the mean of the acceptable range
print(Loan_Amount_mean)

#imputing outlier values with mean value
df_loanAmt_exclude.LoanAmount
df_loanAmt_exclude.LoanAmount=Loan_Amount_mean
print(df_loanAmt_exclude.shape)


#getting back the original shape of df
New_df_rev=pd.concat([df_loanAmt_include,df_loanAmt_exclude]) #concatenating both dfs to get the original shape
print(New_df_rev.shape)
print(New_df_rev.head(10))


New_df_rev.boxplot(column='LoanAmount')
plt.show()


#Numerical Variable

#depending on the amount of missing values and the expected importance of variables.





#Categorical Variable

#male prob high
New_df_rev['Gender'].value_counts()

#maried Yes max
New_df_rev['Married'].value_counts()

#self_employed NO max
New_df_rev['Self_Employed'].value_counts()

#credit history 1 max
New_df_rev['Credit_History'].value_counts()

#semiurban max
New_df_rev['Property_Area'].value_counts()


New_df_rev['Loan_Amount_Term'].value_counts()

New_df_rev.isnull().sum()
# Categorical Variable replace the missing values with values in the top row of each column
for value in ['Gender', 'Married',
              'Dependents','Self_Employed','Loan_Amount_Term','Credit_History']:
    New_df_rev[value].fillna(New_df_rev[value].mode()[0], inplace=True)
    print(New_df_rev.head(20))




# For preprocessing the data
from sklearn import preprocessing

le = {}

for x in ['Loan_ID','Gender', 'Married','Education',
              'Dependents','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']:
    le[x] = preprocessing.LabelEncoder()

for x in ['Loan_ID','Gender', 'Married','Education',
              'Dependents','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']:
    New_df_rev[x] = le[x].fit_transform(New_df_rev[x])


# 0--> No
# 1--> Yes


New_df_rev.dtypes
New_df_rev.isnull().sum()


X = New_df_rev.values[:, :-1]
print(X)
Y = New_df_rev.values[:, -1]
print(Y)


from sklearn.model_selection import train_test_split

# Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=10)


from sklearn.linear_model import LogisticRegression

# create a model
lr = LogisticRegression()
# fitting training data to the model
lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)
print(list(zip(Y_test, Y_pred)))


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test, Y_pred))

acc = accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ", acc)


#using Decision Tree classfier
from sklearn.tree import DecisionTreeClassifier
dr = DecisionTreeClassifier(criterion='gini',random_state=10, max_depth = 6)
dr.fit(X_train,Y_train)
Y_pred = dr.predict(X_test)
print(list(zip(Y_test,Y_pred)))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)



#running the random forest model
from sklearn.ensemble import RandomForestClassifier
dr = RandomForestClassifier(501,random_state=2)
dr.fit(X_train,Y_train)
Y_pred = dr.predict(X_test)
print(list(zip(Y_test,Y_pred)))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)


