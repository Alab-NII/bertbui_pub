# BERT BUI

This repository consists of two parts:
- tasksvr
- model training \& evaluation

## Development Environment  

- Ubuntu 18.4
- Python 3.6 with venv
- Firefox 87.0
- geckodriver 0.29.0

Note that we developed this repository in a venv environment.

```
cd bertbui_pub
python3 -m venv venv
source venv/bin/activate
```

We assume that we execute the commands in the following sections in the environment. 

## Tasksvr

Tasksvr provides tasks as web pages, mainly depending on datasets and tornado.

### Install and Boot

Install

```
cd bertbui_pub/tasksvr
pip install -e .
python -m tasksvr download
```

Download command will make tasksvr/cache directory and download dataset there from GLUE (via huggingface's datasets), VQA, and coco.
Please see license for those datasets or packages

- GLUE: https://gluebenchmark.com/tasks (links to each dataset)
- Huggingface's datasets: https://github.com/huggingface/datasets
- VQA: https://visualqa.org/terms.html
- coco: https://cocodataset.org/#termsofuse

Boot

```
python -m tasksvr run
```

## Model Training \& Evaluation

A set of scripts for model training and evaluation, mainly depending on selenium, firefox, pytorch and transformers.

### Install

Download the Firefox binary and geckodriver into the venv/bin directory.

- Firefox: https://www.mozilla.org/en-US/firefox/
- geckodriver: https://github.com/mozilla/geckodriver/releases

```
cd bertbui_pub/bertbui
pip install -e .
```

This command outputs some view images for normality checking.

```
python bertbui/src/check_env
```

## BUI Models

### Gold Sequences Recording

```
# Make sure that tasksvr is running
python bertbui/src/record
```

### Training

```
# Make sure that tasksvr is running
python bertbui/src/train_bui.py
```

### Prediction

```
# Make sure that tasksvr is running
python bertbui/src/predict_bui.py
```

### Evaluation

```
python tasksvr/src/evaluate.py models/model_x/predictions/cola.dev
```

### Check Model Actions

See this notebook.

## Non-BUI Models

### Training & Prediction

#### Task head models
Train 
```
python bertbui/src/train.py
```

Predict
```
python bertbui/src/train.py
```

#### Sequence to sequence models.
Train 
```
python bertbui/src/train_s2s.py
```

Predict
Predict answer with sequence to sequence models. 
```
python bertbui/src/train_s2s.py
```

### Evaluation

```
python tasksvr/src/evaluate.py models/model_x/predictions/cola.dev
```
