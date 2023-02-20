import os
from jinja2 import Template

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]

class_template = """import nest_asyncio
import pandas as pd
import joblib
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from pyngrok import ngrok
import os
import numpy as np


class request_body(BaseModel):
    {%+ for column, data_type in initial_column_metadata -%}
        {{ column }}: np.{{ data_type.name }}
    {%+ endfor -%}
"""

func_template = """

app = FastAPI(debug=True)

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
preprocessor = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/preprocessor/preprocessor.joblib')
pipeline = joblib.load(BLUEMIST_PATH + '/' + 'artifacts/models/LarsCV.joblib')


@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    input_data = [[
        {%+ for column, _ in initial_column_metadata -%}
            data.{{ column }},
        {%+ endfor -%}
        ]]

    input_df = pd.DataFrame(input_data, columns=[
        {%+ for column, _ in initial_column_metadata -%}
            '{{ column }}',
        {%+ endfor -%}
        ])

    df_to_predict = pd.DataFrame(preprocessor.transform(input_df), columns=[
        {%+ for column in encoded_column_metadata -%}
            '{{ column }}',
        {%+ endfor -%}
        ])

    # Predicting the Class
    prediction = pipeline.predict(df_to_predict)

    # Return the Result
    return {'predicted_{{ target_variable }}': prediction[0]}


def start_api_server(host='localhost', port=8000):
    ngrok_tunnel = ngrok.connect(port)
    ngrok_tunnel
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)

"""


def generate_api_code(estimator_name, initial_column_metadata, encoded_column_metadata, target_variable):
    template = Template(class_template)
    class_code = template.render(initial_column_metadata=initial_column_metadata)

    template = Template(func_template)
    func_code = template.render(initial_column_metadata=initial_column_metadata,
                                encoded_column_metadata=encoded_column_metadata, estimator_name=estimator_name,
                                target_variable=target_variable)

    with open(BLUEMIST_PATH + '/' + 'artifacts/api/predict.py', 'w') as f:
        f.truncate()
        f.write(class_code)
        f.write(func_code)



