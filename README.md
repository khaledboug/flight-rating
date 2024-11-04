# Flight Rating RAMP Challenge

This repository contains the **Flight Rating** RAMP challenge, designed as a structured competition to evaluate predictive modeling solutions for a specific dataset. Created for use with **RAMP** (Rapid Analytics and Model Prototyping), this challenge framework is ready to be deployed on RAMP Studio, where participants can collaborate on predictive modeling tasks and benchmark their solutions.

### Project Structure

The project is organized across two main directories:
- **ramp-data/flight-rating**: Contains data preparation scripts and dependencies to set up and clean the dataset.
    - `data/`: Directory to store train/test data files.
    - `prepare_data.py`: Script for data preparation.
    - `requirements.txt`: Lists all necessary dependencies.

- **ramp-kits/flight-rating**: The primary kit directory, containing files needed to configure and launch the challenge.
    - `data/`: Placeholder for challenge data.
    - `submissions/starting_kit/`: Includes a baseline submission to help participants get started.
    - `flight-rating_starting_kit.ipynb`: Jupyter notebook with an overview of the problem, exploratory analysis, and example solution.
    - `problem.py`: Defines the predictive task using RAMP workflow specifications.
    - `environment.yml` and `requirements.txt`: Dependency files for conda and pip setup.

This structure is intended to provide a clear, organized approach for participants, ensuring they can easily prepare, understand, and participate in the challenge on RAMP Studio.
