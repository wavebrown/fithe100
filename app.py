from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField, StringField
from wtforms.validators import NumberRange
import pandas as pd
from tensorflow.keras.models import load_model
import joblib



def return_prediction(model, scaler, sample_json):
    test_sex = sample_json['test_sex']
    height = sample_json['height']
    weight = sample_json['weight']
    fat = sample_json['fat']
    core_cm = sample_json['core_cm']
    situp = sample_json['situp']
    sitflex = sample_json['sitflex']
    longrun = sample_json['longrun']
    run10m = sample_json['run10m']
    longjump = sample_json['longjump']

    test_data = [[test_sex, height, weight, fat, core_cm, situp, sitflex, longrun, run10m, longjump]]
    # scaler를 불러와서 scaling
    test_data_scaled = scaler.transform(test_data)

    # 예측값 생성
    predict = model.predict(test_data_scaled)

    return str(round(predict[0][0], 3))

app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'someRandomKey'
F_model = load_model("model/F_model_1208.h5")
M_model = load_model("model/M_model_1208.h5")
F_scaler = joblib.load("model/F_scaler_1208.pkl")
M_scaler = joblib.load("model/M_scaler_1208.pkl")
group = pd.read_pickle("model/group.pkl")
# Loading the model and scaler
# adult_model = load_model("F_model_1208.h5")
# adult_scaler = joblib.load("F_scaler_1208.pkl")

class GreetUserForm(FlaskForm):
    test_sex = TextField('성별')
    height = TextField('키')
    weight = TextField('몸무게')
    fat = TextField('체지방율')
    core_cm = TextField('허리둘레')
    situp = TextField('교차윗몸일으키기')
    sitflex = TextField('앉아윗몸앞으로굽히기')
    longrun = TextField('20m왕복오래달리기')
    run10m = TextField('10M4회왕복달리기')
    longjump = TextField('제자리멀리뛰기')

    submit = SubmitField("분석하기")

@app.route('/', methods=('GET', 'POST'))
def index():
    form = GreetUserForm()
    if form.validate_on_submit():
        session['test_sex'] = form.test_sex.data
        session['height'] = form.height.data
        session['weight'] = form.weight.data
        session['fat'] = form.fat.data
        session['core_cm'] = form.core_cm.data
        session['situp'] = form.situp.data
        session['sitflex'] = form.sitflex.data
        session['longrun'] = form.longrun.data
        session['run10m'] = form.run10m.data
        session['longjump'] = form.longjump.data
        print(form.test_sex.data)
        print(form.height.data)
        print(form.weight.data)
        print(form.fat.data)
        print(form.core_cm.data)

        return redirect(url_for("prediction"))
    return render_template('index.html', form=form)

@app.route('/prediction')
def prediction():

    content = {}

    content['test_sex'] = float(session['test_sex'])
    content['height'] = float(session['height'])
    content['weight'] = float(session['weight'])
    content['fat'] = float(session['fat'])
    content['core_cm'] = float(session['core_cm'])
    content['situp'] = float(session['situp'])
    content['sitflex'] = float(session['sitflex'])
    content['longrun'] = float(session['longrun'])
    content['run10m'] = float(session['run10m'])
    content['longjump'] = float(session['longjump'])
    print(content)
    if(content['test_sex'] == 0):
        results = return_prediction(model=F_model,scaler=F_scaler,sample_json=content)
    elif (content['test_sex'] == 1):
        results = return_prediction(model=M_model, scaler=M_scaler, sample_json=content)

    # pivot data select
    group_label = str(session['test_sex'])
    # pivot_select = pd.DataFrame(group.loc[group_label]).T
    pivot_select = group
    if float(session['test_sex']) > 0.5:
        pivot_col = [pivot_select.columns[0][0], pivot_select.columns[1][0], pivot_select.columns[2][0],
                     pivot_select.columns[3][0],pivot_select.columns[4][0]]
        print("pivot_col",pivot_col)
        pivot_data = [pivot_select.iloc[0, 0], pivot_select.iloc[0, 1], pivot_select.iloc[0, 2],
                      pivot_select.iloc[0, 3],pivot_select.iloc[0, 4]]
        print("pivot_data",pivot_data)
    else:
        pivot_col = [pivot_select.columns[0][0], pivot_select.columns[1][0], pivot_select.columns[2][0],
                     pivot_select.columns[3][0], pivot_select.columns[4][0]]
        pivot_data = [pivot_select.iloc[1, 0], pivot_select.iloc[2, 0], pivot_select.iloc[3, 0],
                      pivot_select.iloc[4, 0],pivot_select.iloc[5, 0]]

    # 예측 결과 리턴
    return render_template('prediction.html', results=results, pivot_col=pivot_col, pivot_data=pivot_data)



if __name__ == '__main__':
    app.run(debug=True)