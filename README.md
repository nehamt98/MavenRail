# mavenrail

## Install enviornment

```
conda env create -f environment.yml
```

## to update the environment.yml

```
conda env export | grep -v "^prefix: " > environment.yml
```

## Setup environment
```
conda create --name mavenrail_venv python=3.12
```

## Activate environment
```
conda activate mavenrail_venv
```

## Preprocess data
```
python -m pipeline.preprocess --filename TrainRidesCleaned.csv --test_size 0.1 --random_state 42
```

## Train model
```
python -m pipeline.train
```

## Inference
```
python -m pipeline.inference
```
## run pre-commits
```
pre-commit run --all-files
```

## to run the fast api
```
uvicorn main:app --reload
```

## Open Swagger
by opening browser with this link http://127.0.0.1:8000/

## send payload
sample payload to send in swagger to router /infernce/
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
