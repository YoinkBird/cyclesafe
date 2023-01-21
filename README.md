# Cycle Safe Model Generation Prototype

Model generation, system documentation (i.e. the report), and assorted helper utilities for the [Probabilistic Routing-Based Injury Avoidance Navigation Framework for Pedalcyclists](./docs/report/report.md) Project.

This is a rudimentary custom machine learning framework using scikit-learn and pandas.

This code (mostly automatically) manages model generation and validation/evaluation (train+test on TxDoT Crash Data), as well as scoring navigation routes (based on Geo-JSON).


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
