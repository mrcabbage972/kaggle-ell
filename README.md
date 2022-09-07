# About The Project
This is a template for a Kaggle project, adapted to the Feedback 3 Prize competition.

The goal is for a Kaggle competitor to be able to:
1. Develop their solution in an IDE (PyCharm, VSCode, etc) rather than in a Jupyter-style notebook
2. Easily train a model in different environments (Kaggle, Google Colab, Jarvislabs, etc)
3. Submit a solution without copy-pasting code to an inference notebook

# Usage Guide
The project has a single entry point, which is the script `run_solution`. It can be used to run all three typical Kaggle workflows: train a model, create a submission and make predictions for ensembling.
## Local Environment Setup
1. Create an IDE project from the repository root
2. Create a virtual environment for the project (using either virtualenv, conda, or a similar tool)
3. Install the project in development mode: `cd src && pip install -e . `
## Train a Model
1. In the training environment, install the project: `cd src && pip install .` 
2. Run the script: ```run_solution env=[ENVIRONMENT_NAME] solution=[SOLUTION_NAME] solution.do_train=true ```
3. Export the project to a wheel and export it to a Kaggle dataset
## Create a Submission
1. In a Kaggle notebook, install the project from a dataset: `cd ../input/[DATASET NAME] && pip install .` 
2. Run the script: ```run_solution wandb.enabled=false env=[ENVIRONMENT_NAME] solution=[SOLUTION_NAME] solution.do_create_submission=true ```


## Configuration
The configuration of the `run_solution` script is done with the `hydra-core` package, which allows specifying any attribute via either the config yaml files or the CLI.
The top-level configs are `env` and `solution`. Their respective yaml's are located in: 

    .
    └── src                   
        └── kaggle_ell
            └── config
                ├── env
                │    ├── colab.yaml
                │    ├── kaggle.yaml
                │    ├── local.yaml
                │    └── [YOUR ENV NAME].yaml
                └── solution 
                    ├── dummy_solution.yaml
                    └── [YOUR SOLUTION NAME].yaml
Toggling between different environments and solutions is as simple as calling `run_solution env=[ENV NAME] solution=[SOLUTION NAME]`.

It's also possible to override specific parameter values from the command-line (or notebook). Example: 
```
run_solution \
env=kaggle \
solution=transformer_finetune \
solution.do_train=true \
solution.args.train.epochs=3 \
solution.args.train.trn_fold=[0,1,2,3] \
solution.args.model.backbone=microsoft/deberta-v3-base
```