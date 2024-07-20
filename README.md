
# Lunar Lander

## How to run

### Prerequisites

- Pytorch https://pytorch.org/
- Swig (for building box2d) https://swig.org/


### Running
1.) First install the requirements.txt

```bash
pip install -r requirements.txt
```

2.) Run main.py

```bash
python src/main.py
```


### How to use

To run in training mode, simply add --training when running. By default it will be in evaluate mode 


```bash
python src/main.py --training
```

To load a model, use the --model and pass in the file path to the model. By default it will create a blank model to evaluate so remember to load the model if you want to evalutate it

```bash
python src/main.py --model models/something.pth
```

or 
with training 

```bash
python src/main.py --training --model models/something.pth
```

To set your own hyperparameters you can use --parameters and pass in the your own file and your own name of which hyperparameters to load. By default it will load *test1* from *hyperparameters.yaml*

```
python src/main.py --training --parameters hyperparamters.yaml test1
````

