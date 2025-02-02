# mavenrail

## Install enviornment

```
conda env create -f environment.yml
```

to update the environment.yml

```
conda env export | grep -v "^prefix: " > environment.yml
```

## Setup environment
```
conda create --name mavenrail python=3.12
```

## Preprocess data
```
python preprocess.py --filename TrainRidesCleaned.csv --test_size 0.1 --random_state 42
```

## Train model
```
python train.py
```

## Inference
```
python inference.py
```
