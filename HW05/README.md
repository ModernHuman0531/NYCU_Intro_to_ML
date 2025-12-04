# HW5. Build & Compare NN vs CNN (Image Classification)
## Files structure
```
.
|- data/
| |- sample_submission.csv
| |- test4students.csv
| └─ train.csv
|- data_loader.py           # Load data and pre-processing
|- models.py                # Define models(NN,CNN,Residual Blocks .etc)
|- param_counter.py         # Count the amounts of parameters of model
|- train.py                 # Training model, validating model and record the result
|- evaluation.py            # Execute Abliation experiment, final test evaluation
|- main.py                  # Colaborate with the whole process, to process different experiments
└─ README.md
```