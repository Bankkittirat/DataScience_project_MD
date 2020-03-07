from django.shortcuts import render
from joblib import load
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)

# Create your views here.
def index(req):
    model = load('./chatgroup/static/chatgroup.model')
    label = ""
    chat  = ""
    if req.method == 'POST':
        print("POST IN")
        chat = str(req.POST['chat'])
        print(chat)
        pred = model.predict([chat])
        label = train.target_names[pred[0]]
    return render(req, 'chatgroup/index.html' ,{
            'label':label,
    })