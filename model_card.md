# Model Card
- Model predicts if the income exceeds $50k/yr based on census information.
- Developed as a demo project for udacity MLDevOps degree

## Model Details
- Random forest classifier

Model inputs:
- workclass 
- marital_status
- occupation
- relationship
- race
- sex
- native_country
- age
- education_num
- hours_per_week

Model outpus:
- 0/1 (0 is below $50k)

## Intended Use
Model shall be used for demo purposes of CI/CD pipeline - not to show off good modelling knowledge. 

## Training Data
- Model was trained on a cleaned census dataset (https://archive.ics.uci.edu/ml/datasets/census+income)
- Total number of samples in cleaned dataset: 32561
- Training data size 85%.

## Evaluation Data
- Model was evaluated on cleaned census dataset (https://archive.ics.uci.edu/ml/datasets/census+income)
- Model was tested on 15 percent of data not used for training.

## Metrics
- As metric precision, recall and the F-beta score were evaluated.
- Performance on training data: precision 0.95, recall: 0.93, fbeta: 0.94
- Performance on test data: precision 0.66, recall: 0.57, fbeta: 0.61

## Ethical Considerations
Some model features are gender and race. This model may be biased - further investigation into this topic is needed.

## Caveats and Recommendations
Recommendation: Dont trust the predictions of this model too much. ;-)