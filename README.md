# DataOrbit 2025 -- CashGPT

![Model Diagram](model_diagram.png "Model Diagram")

## Synopsis

This project was inspired by our observation of the substantial time and financial resources expended by individuals pursuing insurance claim complaints.

Our project revolutionizes auto insurance claims processing with AI-driven predictive analytics. By analyzing claims data, regulatory complaints, and resolution patterns, we provide: Claim Decision Predictions – Instantly assess the likelihood of claim approval or denial. Dispute Outcome Forecasting – Estimate the probability of approval after a dispute. Data-Driven Insights – Help insurers reduce losses, prevent disputes, and enhance customer satisfaction. With an interactive dashboard for insurers and potential consumer-facing tools, our solution streamlines claims management, minimizes costs, and improves industry transparency. Faster Claims. Fewer Disputes. Smarter Decisions.

We built this project using predictive modeling. We combined the results of two independent models trained on two separate datasets, an initial claims model (Logistic Regression, ~80% accuracy) and a complaints model (Random Forest, ~90% accuracy), using marginal probabilities to predict the probability of an initial claim, or an initial claim + complaint being approved. We also built a REST API to provide easy access to the model.

Complaints dataset: https://www.kaggle.com/datasets/adelanseur/insurance-company-complaints

Initial claims dataset: https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data

We spent a lot of time looking for good datasets for health and life insurance, but eventually we realized there's a lot more data available for auto insurance, so we pivoted. We also had some trouble with the combination of the results of the two models, and optimizing the transformations applied to both datasets we used.

We are very proud of what we accomplished because this was the first time we have ever worked together and we were very productive in such little amount of time.

Given that it was all our first datathon, we all learned so much about how these competitions go. We learned how to work very efficiently and the ins and outs of predictive modeling.

## Video Summary

https://github.com/user-attachments/assets/31d4214a-b239-4b23-a067-94633fb4784f

## How to run

1) cd into main project directory
2) Run the local backend: 
```
cd backend && python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt && python backend_api.py
```
3) Run the local frontend (different terminal): 
```
cd frontend && npm i && npm start
```
