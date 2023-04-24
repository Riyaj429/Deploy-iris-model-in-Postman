import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
import json

auth = HTTPBasicAuth(scheme='Basic')
app = Flask(__name__)

df = pd.read_csv("iris.csv")

x = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

model_file = 'model.pkl'
pickle.dump(classifier, open(model_file, 'wb'))

users = {
    "user1": "password1",
    "user2": "password2"
}

@auth.verify_password
def verify_password(username, password):
    if username in users and password == users[username]:
        return username

@app.route("/predict", methods=["POST"])
@auth.login_required
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    
    prediction = classifier.predict(query_df)
    
    return jsonify({"Prediction": list(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

