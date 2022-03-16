# BERT BUI

Work In Progress

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

This will take a long time because it will download zip files that contain many images. 

Boot

```
python -m tasksvr run --use cola,mnli,mrpc,stsb,qnli,qqp,rte,sst2,wnli,squad_v2,pta,vqa_v2,sa
```

## Model Training \& Evaluation

A set of scripts for model training and evaluation, mainly depending on selenium, firefox, pytorch and transformers.

### Install

Download the Firefox binary and geckodriver.

- Firefox: https://www.mozilla.org/en-US/firefox/ -> venv/lib
- geckodriver: https://github.com/mozilla/geckodriver/releases -> venv/bin

Note that the path to Firefox can be changed from venv/lib.

In addition, we need to install pytorch and torchvision manually.
We do not include those packages in requirement.txt because the optimal versions of those packages depend on the environment.

After that, install the bertbui package.

```
cd bertbui_pub/bertbui
pip install -e .
```

If successful, the command below outputs some gif images for normality checking.

```
python -m bertbui check_env --firefox path_to_firefox_binary
```

## BUI Models

### Gold Sequences Recording

We need to record gold sequences before bui model training.
The command below will record training and validation splits of wnli and outputs some json files in the direcotry named static. 

```
# Make sure that tasksvr is running
python -m bertbui record --binary_location /Applications/Firefox.app/Contents/MacOS/firefox --targets /train/wnli,/valid/wnli
```

### Training

To train model, we use src/train_bui.py script.
This time, tasksvr is not required because model will be trained on the recorded files.

```
# Make sure that tasksvr is running
python src/train_bui.py --model_path models/test
```

### Prediction

After training, we can use the trainded model to predict answers for tasks from tasksvr. 
This script make prediction files in the model directory.

```
# Make sure that tasksvr is running
python src/predict_bui.py --task_path /valid/wnli --model_path path_to_model --firefix path_to_firefox_binary
```

### Evaluation

We can evaluate the prediction with tasksvr.

```
python -m tasksvr evaluate path_to_prediction_file
```
