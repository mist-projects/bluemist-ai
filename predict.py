import joblib
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd

# Creating FastAPI instance
app = FastAPI()

preprocessor = joblib.load('artifcats/preprocessor/preprocessor.joblib')
pipeline = joblib.load('artifcats/models/LinearRegression.joblib')


# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    cylinders: int
    displacement: float
    weight: int
    acceleration: int
    model_year: int
    horsepower: int


# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    input_data = [[
        data.cylinders,
        data.displacement,
        data.weight,
        data.acceleration,
        data.model_year,
        data.horsepower
    ]]

    input_df= pd.DataFrame(input_data, columns=['cylinders', 'displacement', 'weight', 'acceleration', 'model_year',
                                                  'horsepower'])

    df_to_predict = pd.DataFrame(preprocessor.transform(input_df), columns=['cylinders', 'displacement', 'weight',
                                                                             'acceleration', 'model_year',
                                                                             'horsepower'])

    # Predicting the Class
    prediction = pipeline.predict(df_to_predict)

    # Return the Result
    return {'class': prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
