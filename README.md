# Deploy a ML model to production
Udacity demo project with a pipeline to train a model based on the census dataset.
The main focus is on the continuous integration and continous deployment (CI/CD) process.

## ML pipeline
Model pipeline can be run useing the main.py script in the src folder.
```
src/main.py
```
The pipeline performs:
- Basic cleaning
- Training a random forest classifier
- Deployment

## Rest API
For the model a Rest API is provided that was implemented using fast API. To test the server locally run
```
uvicorn --app-dir=src/ server_api:app --reload
```
then the documentation of the API can be accessed by 
```
127.0.0.1:8000/docs/
```

### Continous integration
Continous integration is done with GitHub actions. See the following file for details.
```
.github/workflows/python-package.yml
```
As basic steps pytest and flake8 linter are run if a new commit is pushed.

### Continous deployment
The heroku app is configured to deploy the model to production if the continous integration succeedes.
You can see the app in action here: [Heroku app](https://udacity-ml-devops-ged.herokuapp.com/)

### Rubric files

- [Screenshots](https://github.com/danny-bit/udacity_mlops_deploy/tree/master/screenshots)
- [Model Card](https://github.com/danny-bit/udacity_mlops_deploy/tree/master/model_card.md)
- [Slices Output](https://github.com/danny-bit/udacity_mlops_deploy/tree/master/model/slice_output.txt)
- [Heroku App](https://udacity-ml-devops-ged.herokuapp.com/)