from fastapi.testclient import TestClient
from src.server_api import app

client = TestClient(app)


def test_welcomepost():
    r = client.get("/")
    assert r.status_code == 200
    assert 'Yo' in r.json()["message"]


def test_model_query_low_salary():
    """
    test for census model API - test prediction=0
    """
    r = client.post("/census_model/?", json={
        "workclass": "Private",
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "Black",
        "sex": "Male",
        "native_country": "United-States",
        "hours_per_week": 30,
        "education_num": 1,
        "age": 20
    })
    assert r.status_code == 200
    assert r.json() == 0


def test_model_query_high_salary():
    """
    test for census model API - test prediction=1
    """
    r = client.post("/census_model/?", json={
        "workclass": "Private",
        "marital_status": "Married-civ-spouse",
        "occupation": "Armed-Forces",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
        "hours_per_week": 56,
        "education_num": 13,
        "age": 35
    })
    assert r.status_code == 200
    assert r.json() == 1


def test_model_query_invalid_value():
    r = client.post("/census_model/?", json={
        "workclass": "Hero",  # erroneus value
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
        "hours_per_week": 60,
        "education_num": 1,
        "age": 50
    })
    assert r.status_code == 422
