from jinja2 import Template

column_metadata = [
    ('cylinders', int),
    ('displacement', int),
    ('weight', int),
    ('acceleration', int),
    ('model_year', int),
    ('horsepower', int)]

class_template = """import nest_asyncio
import pandas as pd
import joblib
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from pyngrok import ngrok


class request_body(BaseModel):
    {%+ for column, data_type in column_metadata -%}
        {{ column }}: {{ data_type.__name__ }}
    {%+ endfor -%}
"""

func_template = """

app = FastAPI(debug=True)

preprocessor = joblib.load('artifcats/preprocessor/preprocessor.joblib')
pipeline = joblib.load('artifcats/models/{{ estimator_name }}.joblib')


@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    input_data = [[
        {%+ for column, _ in column_metadata -%}
            data.{{ column }},
        {%+ endfor -%}
        ]]

    input_df = pd.DataFrame(input_data, columns=[
        {%+ for column, _ in column_metadata -%}
            '{{ column }}',
        {%+ endfor -%}
        ])

    df_to_predict = pd.DataFrame(preprocessor.transform(input_df), columns=[
        {%+ for column, _ in column_metadata -%}
            '{{ column }}',
        {%+ endfor -%}
        ])

    # Predicting the Class
    prediction = pipeline.predict(df_to_predict)

    # Return the Result
    return {'class': prediction[0]}


def start_api_server(host='localhost', port=8000):
    ngrok_tunnel = ngrok.connect(port)
    ngrok_tunnel
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)

"""


def generate_api_code(estimator_name):
    template = Template(class_template)
    class_code = template.render(column_metadata=column_metadata)

    template = Template(func_template)
    func_code = template.render(column_metadata=column_metadata, estimator_name=estimator_name)

    # Saving the class and method in a separate file
    with open('model_api.py', 'w') as f:
        f.write(class_code)
        f.write(func_code)


