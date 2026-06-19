from flask import Flask, request, send_file
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import plotly
import plotly.express as px
import json


app = Flask(__name__)


# =========================
# AI CAREER DATASET
# =========================


data = pd.DataFrame({

    "Career":[
        "AI Engineer",
        "Software Engineer",
        "Data Scientist",
        "UI/UX Designer",
        "Cyber Security Analyst"
    ],


    "Math":[
        5,4,5,2,4
    ],


    "Coding":[
        5,5,5,2,4
    ],


    "Creativity":[
        3,3,2,5,3
    ],


    "Communication":[
        4,4,3,5,4
    ]

})



encoder = LabelEncoder()


data["Career_ID"] = encoder.fit_transform(
    data["Career"]
)



X = data[
    [
        "Math",
        "Coding",
        "Creativity",
        "Communication"
    ]
]


y = data["Career_ID"]




model = RandomForestClassifier(

    n_estimators=200,

    random_state=42

)


model.fit(
    X,
    y
)



# Store user result

user_result = {

    "career":"Not Generated",

    "confidence":0,


    "skills":{

        "Math":3,

        "Coding":3,

        "Creativity":3,

        "Communication":3

    }

}





# =========================
# HOME
# =========================


@app.route("/")

def home():

    return open(

        "index.html",

        encoding="utf-8"

    ).read()





# =========================
# PREDICTION
# =========================


@app.route(

    "/predict",

    methods=["POST"]

)

def predict():



    skills = {


        "Math":

        int(request.form["math"]),


        "Coding":

        int(request.form["coding"]),


        "Creativity":

        int(request.form["creativity"]),


        "Communication":

        int(request.form["communication"])

    }



    input_data = np.array([

        list(skills.values())

    ])



    prediction = model.predict(

        input_data

    )[0]



    confidence = max(

        model.predict_proba(input_data)[0]

    ) * 100





    user_result["career"] = encoder.inverse_transform(

        [prediction]

    )[0]



    user_result["confidence"] = round(

        confidence,

        2

    )



    user_result["skills"] = skills




    return dashboard()





# =========================
# DASHBOARD
# =========================


@app.route("/dashboard")

def dashboard():


    career_chart = px.bar(

        data,

        x="Career",

        y=[

            "Math",

            "Coding",

            "Creativity",

            "Communication"

        ],

        barmode="group",

        title="Career Skill Comparison"

    )




    skill_df = pd.DataFrame({

        "Skill":

        list(user_result["skills"].keys()),


        "Score":

        list(user_result["skills"].values())

    })




    skill_chart = px.pie(

        skill_df,

        names="Skill",

        values="Score",

        title="Your Skill Profile"

    )





    html=open(

        "dashboard.html",

        encoding="utf-8"

    ).read()





    html=html.replace(

        "{{career}}",

        user_result["career"]

    )


    html=html.replace(

        "{{confidence}}",

        str(user_result["confidence"])

    )



    html=html.replace(

        "{{career_chart}}",

        json.dumps(

            career_chart,

            cls=plotly.utils.PlotlyJSONEncoder

        )

    )



    html=html.replace(

        "{{skill_chart}}",

        json.dumps(

            skill_chart,

            cls=plotly.utils.PlotlyJSONEncoder

        )

    )



    return html





# =========================
# EXTRA FEATURES
# =========================


@app.route("/skill-analysis")

def skill_analysis():

    return open(

        "skill_analysis.html",

        encoding="utf-8"

    ).read()





@app.route("/prediction")

def prediction():

    html=open(

        "prediction.html",

        encoding="utf-8"

    ).read()



    html=html.replace(

        "{{career}}",

        user_result["career"]

    )


    html=html.replace(

        "{{confidence}}",

        str(user_result["confidence"])

    )


    return html





# CSS

@app.route("/style.css")

def css():

    return send_file(

        "style.css"

    )





application = app


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
