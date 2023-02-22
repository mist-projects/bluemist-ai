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
    AGE: int
    SEX: str
    BMI: float
    BP: float
    S1: int
    S2: float
    S3: float
    S4: float
    S5: float
    S6: int
    

app = FastAPI(debug=True)

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
preprocessor = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/preprocessor/preprocessor.joblib')
pipeline = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/models/LarsCV.joblib')


@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    input_data = [[
        data.AGE,
        data.SEX,
        data.BMI,
        data.BP,
        data.S1,
        data.S2,
        data.S3,
        data.S4,
        data.S5,
        data.S6,
        ]]

    input_df = pd.DataFrame(input_data, columns=[
        'AGE',
        'SEX',
        'BMI',
        'BP',
        'S1',
        'S2',
        'S3',
        'S4',
        'S5',
        'S6',
        ])

    df_to_predict = pd.DataFrame(preprocessor.transform(input_df), columns=[
        'AGE',
        'BMI',
        'BP',
        'S1',
        'S2',
        'S3',
        'S4',
        'S5',
        'S6',
        'SEX_2',
        ])

    # Predicting the Class
    prediction = pipeline.predict(df_to_predict)

    # Return the Result
    return {'predicted_Y': prediction[0]}


def start_api_server(host='localhost', port=8000):
    ngrok_tunnel = ngrok.connect(port)
    ngrok_tunnel
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)
