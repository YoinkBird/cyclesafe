# Cycle Safe

Cycle Safe is a navigation system that recommends safer cycling routes by evaluating the risk of routes suggested by Google Maps.

Systems:
* Frontend: Google Maps and very simple JavaScript
  * Doc: [User Application (Frontend)](docs/report/report.md#user-application)
  * Src: https://github.com/YoinkBird/cyclesafe_server
* Backend: Python REST-API
  * Doc: [Architecture](/docs/report/report.md#architecture)
  * Src: https://github.com/YoinkBird/cyclesafe_server 
* Machine Learning Framework: Classifier Model Framework based on python, pandas, scikit-learn, etc
  * Doc: [Machine Learning Framework for Model Lifecycle Management based on CRISP-DM](/docs/report/report.md#framework-for-crisp-dm)
  * Src: [modelmanager](/modelmanager/model.py)


Model:
* Risk Assessment Model: 
  * Design: [Predictions based on Route Data](/docs/report/report.md#predicting-using-route-data)

# Cycle Safe Model Generation Prototype

Model generation, system documentation, and assorted helper utilities for the [Probabilistic Routing-Based Injury Avoidance Navigation Framework for Pedalcyclists](./docs/report/report.md) Project.

This is a rudimentary custom machine learning framework using scikit-learn and pandas.

This code (mostly automatically) manages model generation and validation/evaluation (train+test on TxDoT Crash Data), as well as scoring navigation routes (based on Geo-JSON).


Modules:

| Filename | Purpose |
|---|---|
| [model.py](/modelmanager/model.py)                             | Model Build, Optimise, Predict Route-Score |
| [txdot_parse.py](/modelmanager/txdot_parse.py)                 | Prepare data as outlined under Data Preparation  |
| [feature_definitions.py](/modelmanager/feature_definitions.py) | Track features and their purpose  |
| [mapgen.py](/modelmanager/mapgen.py)                           | Generate maps for static heatmap visualisation  |
| [helpers.py](/modelmanager/helpers.py)                         | Useful functions |

[![container test](https://github.com/YoinkBird/cyclesafe/actions/workflows/github-actions.yml/badge.svg?branch=main)](https://github.com/YoinkBird/cyclesafe/actions?query=branch%3Amain)

## Usage

Start with the [cyclesafe server](https://github.com/YoinkBird/cyclesafe_server) project, which will manage all dependencies.

# Architecture

This repo corresponds to the `Scoring Application` and `Modeling Application` in the [Architecture overview documentation](./docs/report/report.md#architecture).

# Contributing

See the [Contribution Guidlines](./CONTRIBUTING.md) .

# Further Documentation

The [docs directory](./docs/) contains:

* [articles](./docs/articles/): Developer writeups on interesting decisions and operations.
* [system documentation](./docs/report/report.md) : Overall systems documentation, technically originated as a masters report, hence the name.
