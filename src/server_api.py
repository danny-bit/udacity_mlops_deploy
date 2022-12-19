# Put the code for your API here. (API Code)
#

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal  # Literal=Symbol?

app = FastAPI()


class CensusModelInputs(BaseModel):
    workclass: Literal['Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Federal-gov',
                       'State-gov', 'Without-pay']
    marital_status: Literal['Married-civ-spouse', 'Never-married', 'Married-spouse-absent', 'Widowed',
                            'Divorced', 'Separated', 'Married-AF-spouse']
    occupation: Literal['Protective-serv', 'Machine-op-inspct', 'Other-service', 'Prof-specialty',
                        'Adm-clerical', 'Craft-repair', 'Sales', 'Exec-managerial',
                        'Transport-moving', 'Tech-support', 'Farming-fishing', 'Handlers-cleaners',
                        'Priv-house-serv', 'Armed-Forces']
    relationship: Literal['Husband', 'Unmarried',
                          'Own-child', 'Not-in-family', 'Other-relative', 'Wife']
    race: Literal['White', 'Asian-Pac-Islander',
                  'Other', 'Black', 'Amer-Indian-Eskimo']
    sex: Literal['Male', 'Female']
    native_country: Literal['United-States', 'Thailand', 'Canada', 'Cuba', 'Nicaragua', 'Mexico', '?',
                            'Peru' 'China', 'Iran', 'Greece', 'Ireland', 'Philippines', 'England',
                            'Outlying-US(Guam-USVI-etc)', 'Jamaica', 'Columbia', 'South', 'Italy',
                            'El-Salvador', 'Puerto-Rico', 'France', 'Dominican-Republic', 'Vietnam',
                            'Haiti', 'Taiwan', 'India', 'Poland', 'Japan', 'Hong', 'Germany',
                            'Trinadad&Tobago', 'Guatemala', 'Hungary', 'Portugal', 'Yugoslavia', 'Ecuador',
                            'Laos', 'Scotland', 'Cambodia', 'Honduras']
    age: int
    education_num: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
                "workclass": "Private",
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "native_country": "United-States",
                "age": 32,
                "education_num": 13,
                "hours_per_week": 60
            }
        }

# /...../....?var1=1&var2=2
# /{path}/...?{query}


@app.get("/")
async def welcome():
    return {"message": "Yo, nice to see ya!"}  # dict is mapped to json format?


# Use POST action to send data to the server
@app.post("/census_model/")
async def exercise_function(body: CensusModelInputs):
    data_dict = dict(body)
    df_query = pd.DataFrame.from_dict([data_dict])
    model = joblib.load('model/census_model.pkl')
    salary_above_50k = int(model.predict(df_query))
    return salary_above_50k
