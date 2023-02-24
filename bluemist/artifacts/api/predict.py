import nest_asyncio
import pandas as pd
import joblib
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from pyngrok import ngrok
import os
import numpy as np


class request_body(BaseModel):
    age: float
    sex: str
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    

app = FastAPI(debug=True)

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
preprocessor = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/preprocessor/preprocessor.joblib')
pipeline = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/models/LarsCV.joblib')


@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    input_data = [[
        data.age,
        data.sex,
        data.bmi,
        data.bp,
        data.s1,
        data.s2,
        data.s3,
        data.s4,
        data.s5,
        data.s6,
        ]]

    input_df = pd.DataFrame(input_data, columns=[
        'age',
        'sex',
        'bmi',
        'bp',
        's1',
        's2',
        's3',
        's4',
        's5',
        's6',
        ])

    df_to_predict = pd.DataFrame(preprocessor.transform(input_df), columns=[
        'age',
        'bmi',
        'bp',
        's1',
        's2',
        's3',
        's4',
        's5',
        's6',
        'sex',
        ])

    # Predicting the Class
    prediction = pipeline.predict(df_to_predict)

    # Return the Result
    return {'predicted_target': prediction[0]}


def start_api_server(host='localhost', port=8000):
    ngrok_tunnel = ngrok.connect(port)
    ngrok_tunnel
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)
