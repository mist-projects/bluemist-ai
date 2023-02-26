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
    cylinders: int
    displacement: float
    horsepower: float
    weight: int
    acceleration: float
    model_year: str
    

app = FastAPI(debug=True)

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
preprocessor = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/preprocessor/preprocessor.joblib')
pipeline = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/models/LarsCV.joblib')


@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    input_data = [[
        data.cylinders,
        data.displacement,
        data.horsepower,
        data.weight,
        data.acceleration,
        data.model_year,
        ]]

    input_df = pd.DataFrame(input_data, columns=[
        'cylinders',
        'displacement',
        'horsepower',
        'weight',
        'acceleration',
        'model_year',
        ])

    df_to_predict = pd.DataFrame(preprocessor.transform(input_df), columns=[
        'cylinders',
        'displacement',
        'horsepower',
        'weight',
        'acceleration',
        'model_year_71',
        'model_year_72',
        'model_year_73',
        'model_year_74',
        'model_year_75',
        'model_year_76',
        'model_year_77',
        'model_year_78',
        'model_year_79',
        'model_year_80',
        'model_year_81',
        'model_year_82',
        ])

    # Predicting the Class
    prediction = pipeline.predict(df_to_predict)

    # Return the Result
    return {'predicted_mpg': prediction[0]}


def start_api_server(host='localhost', port=8000):
    ngrok_tunnel = ngrok.connect(port)
    ngrok_tunnel
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)
