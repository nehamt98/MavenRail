# Maven rail

This project builds a logistic regression-based machine learning pipeline to predict refund requests for UK rail journeys using the MavenRail dataset. It includes data preprocessing, model training, and inference, with a FastAPI service for serving the model via API endpoints.

## Dataset Overview

The MavenRail dataset originates from a fictional dataset developed for a data visualization competition hosted by Maven Analytics. It covers UK rail journeys from January 1 to April 30, 2024.

**Dataset Source:** [Kaggle - MavenRail](https://www.kaggle.com/datasets/helddata/uk-train-rides-maven-rail-challenge)

### Key Features
- **Purchase Type:** Online or Station
- **Payment Method:** Contactless, Credit Card, or Debit Card.
- **Railcard:** Adult, Disabled, Senior, or empty if not used.
- **Ticket Class:** Standard or First Class.
- **Ticket Type:** Advance, Off-Peak, or Anytime.
- **Price:** The cost of the ticket in pounds (£).
- **Departure Station:** One of the following major UK train stations:
  *London Paddington, London Kings Cross, Liverpool Lime Street, London Euston, York, Manchester Piccadilly, Birmingham New Street, London St Pancras, Oxford, Reading, Edinburgh Waverley, Bristol Temple Meads.*
- **Arrival Destination:** One of the following stations:
  *Liverpool Lime Street, York, Manchester Piccadilly, Reading, London Euston, Oxford, Durham, London St Pancras, Birmingham New Street, London Paddington, Bristol Temple Meads, Tamworth, London Waterloo, Sheffield, Wolverhampton, Leeds, Stafford, Doncaster, Swindon, Nottingham, Peterborough, Edinburgh, Crewe, London Kings Cross, Leicester, Nuneaton, Didcot, Edinburgh Waverley, Coventry, Wakefield, Cardiff Central, Warrington.*
- **Arrival Time:** The scheduled arrival time, formatted as `HH:mm:ss`.
- **Actual Arrival Time:** The actual arrival time, also formatted as `HH:mm:ss`.
- **Journey Status:** On Time, Delayed, or Cancelled.
- **Reason for Delay:** If delayed, the reason is specified as one of:
  *Signal Failure, Technical Issue, Weather Conditions, Weather, Staffing, Staff Shortage, Signal failure, Traffic.*
- **Refund Request:** Whether a refund was requested (Yes/No).

## Setup

### Create & Activate Conda Environment
```bash
conda create --name mavenrail_venv python=3.12
conda activate mavenrail_venv
```

### To update the environment.yml
```bash
conda env export | grep -v "^prefix: " > environment.yml
```

## Running ML pipeline

### Preprocess data
```
python -m src.preprocess --filename TrainRides.csv --test_size 0.1 --random_state 42
```
If you want to specify which columns to use for training, you can pass them using the --columns argument:
```bash
python -m src.preprocess --filename TrainRides.csv --test_size 0.1 --random_state 42 --columns col1 col2 col3
```
If --columns is not specified, the model will be trained on all the features mentioned above.

### Train model
```
python -m src.train
```

### Run inference
```
python -m src.inference
```

## Code quality
### Run pre-commits hooks
```
pre-commit run --all-files
```

## Running the FastAPI Server
### Start FastAPI
```
uvicorn main:app --reload
```

### Open Swagger
Open your browser and go to: http://127.0.0.1:8000/

### Sending a Prediction Request
Use this JSON payload in Swagger or a tool like Postman:
```
{
  "Purchase Type": "Online",
  "Payment Method": "Contactless",
  "Railcard": "Adult",
  "Arrival Time": "13:30:00",
  "Actual Arrival Time": "13:30:00",
  "Ticket Class": "Standard",
  "Ticket Type": "Advance",
  "Price": 43,
  "Departure Station": "London Paddington",
  "Arrival Destination": "Liverpool Lime Street",
  "Journey Status": "On Time",
  "Reason for Delay": "Not Delayed"
}
```
## Limitations & Future Scope
This project provides a functional machine learning pipeline for predicting refund requests using logistic regression. It includes data preprocessing, training, and API-based inference with the flexibility to choose input features. However, there are areas where it can be further enhanced:
- **Expanding Model Options:** Currently, the model uses logistic regression. Exploring additional models like Random Forest, XGBoost, or Neural Networks could improve performance.
- **Advanced Feature Selection:** The pipeline allows selecting input features, but further refinement using techniques like correlation analysis, feature importance scores, or PCA could enhance model efficiency.
- **Dynamic Input Validation:** FastAPI currently treats all features as mandatory. A more sophisticated approach could dynamically enforce mandatory and optional fields based on model training insights.
- **Hyperparameter Optimization:** The model is trained with default parameters. Tuning hyperparameters using methods like Grid Search or Bayesian Optimization could yield better results.

This project serves as a foundation for a more advanced refund prediction system, and future improvements can significantly enhance its accuracy and usability.

## Docker

To run the FAST Api locally.

```
docker build -f Dockerfile.local -t mavenrail-app-local .
docker run -d -p 8000:8000 mavenrail-app-local
```

To view the swagger page, click on http://localhost:8000/docs.

### Build and Deploy to AWS ECR

# Login to ECR
```
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 850995548171.dkr.ecr.eu-north-1.amazonaws.com

# Build and push
docker buildx build --platform linux/amd64 -f Dockerfile.lambda -t 850995548171.dkr.ecr.eu-north-1.amazonaws.com/mavenrail-repo:latest --push .

# Check pushed image (optional)
aws ecr describe-images --repository-name mavenrail-repo --region eu-north-1
```

Once the image is deployed on AWS ECR, create an AWS function using the deployed container image.
