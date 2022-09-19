# concept_xai
Code to create results described in the paper "Example or Prototype? Learning Concept-Based Explanations in Time-Series"

## Environment
The requirements.txt file may be used to create an environment with miniconda using:
```
$ conda create --name <env> --file requirements.txt
```

## Execution
To recreate time series concept reconstruction from the paper execute:
```
python3 main1d.py
```

To recreate mnist 3 concept reconstruction from the paper execute:
```
python3 main2d.py
```

## Survey
Participants of the survey filled in one of the forms bellow.

* Explanation-by-example survey: https://forms.gle/J3EAnAqN99mpw6P39
* Model-specific-prototypes survey: https://forms.gle/rsRzHcXyurPi6LQA9
* Model-agnostic-prototype survey: https://forms.gle/tSZRXbuZUraKW7cz8

## Repository references
Code from the following repositories is used in this directory. 

* https://github.com/nesl/ExMatchina
* https://github.com/OscarcarLi/PrototypeDL
* https://github.com/chihkuanyeh/concept_exp
* https://github.com/hfawaz/dl-4-tsc