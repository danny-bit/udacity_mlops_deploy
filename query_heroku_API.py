import requests

body_data = {
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

r = requests.post('https://udacity-ml-devops-ged.herokuapp.com/census_model/',
                  json=body_data)

print("response code: %s" % r.status_code)
print("response content: %s" % r.json())
