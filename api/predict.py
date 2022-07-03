import joblib
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Creating FastAPI instance
app = FastAPI()

pipeline = joblib.load('models/LinearRegression.joblib')


# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    cylinders: int
    displacement: int
    horsepower: int
    weight: int
    acceleration: int
    model_year: int


# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
        data.cylinders,
        data.displacement,
        data.horsepower,
        data.weight,
        data.acceleration,
        data.model_year,
    ]]

    # Predicting the Class

    class_idx = pipeline.predict(test_data)[0]
    # Return the Result
    return {'class': class_idx}
