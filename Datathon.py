import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

path = kagglehub.dataset_download("adelanseur/insurance-company-complaints")
data = pd.read_csv("Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv")

data = data.fillna("<None>")

unique_subreasons = data['SubReason'].unique()

data['Opened'] = pd.to_datetime(data['Opened'])
data['Closed'] = pd.to_datetime(data['Closed'])
data['duration'] = (data['Closed'] - data['Opened']).dt.total_seconds() / 60


data = pd.get_dummies(data, columns=["Coverage", "SubReason", "Company"]) # SubReason is a very important variable for the accuracy score
X = data[['duration'] + [col for col in data.columns if col.startswith('Coverage_') or col.startswith('SubReason_') or col.startswith("Comapany_")]]
data["Approved"] = data["Recovery"] > 0
y = data['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# model = LogisticRegression(max_iter=10000)
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
probability_approved = model.predict_proba(X_test)[:,1].mean()
probability_denied = model.predict_proba(X_test)[:,0].mean()
print(probability_approved)
print(probability_denied)

sum = 0
count = 0
for subreason in unique_subreasons:
    count += 1
    temp_X_test = X_test.copy()
    for col in temp_X_test.columns:
        if col.startswith('SubReason_'):
            temp_X_test[col] = 0
    temp_X_test[f'SubReason_{subreason}'] = 1
    probability_approved = model.predict_proba(temp_X_test)[:, 1].mean()
    sum += probability_approved
print(sum/count)

