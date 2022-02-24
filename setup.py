from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('MODELS/model123.pkl','rb'))
model2 = pickle.load(open('MODELS/modelfinale.pkl','rb'))
model3 = pickle.load(open('MODELS/cool.pkl','rb'))
model4 = pickle.load(open('MODELS/cool2.pkl','rb'))
tr = pickle.load(open('MODELS/tr.pkl','rb'))
df = pickle.load(open('MODELS/df.pkl','rb'))
gnb = pickle.load(open('MODELS/gnd.pkl','rb'))
disease = pickle.load(open('MODELS/disease.pkl','rb'))


l2=[]
for x in range(0,len(model3)):
    l2.append(0)
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[model3]
y_test = tr[["prognosis"]]
np.ravel(y_test)

X= df[model3]

y = df[["prognosis"]]
np.ravel(y)

gnb=gnb.fit(X,np.ravel(y))
from sklearn.metrics import accuracy_score
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred, normalize=False))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    Symptom1 = request.form.get('Symptom1')
    Symptom2 = request.form.get('Symptom2')
    Symptom3 = request.form.get('Symptom3')
    Symptom4 = request.form.get('Symptom4')
    Symptom5 = request.form.get('Symptom5')
    input_query = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
    model.append(input_query)
    for k in range(0,len(model3)):
        for z in input_query:
            if(z==model3[k]):
                l2[k]=1
    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]
    h='no'
    for a in range(0,len(disease)):
        if(disease[predicted] == disease[a]):
            h='yes'
            break
    return jsonify({'disease':str(disease[a])})

if __name__ == '__main__':
    app.run(debug=True)