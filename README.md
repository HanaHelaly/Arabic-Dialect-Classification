# Arabic Dialect Classification

Many countries speak Arabic; however, each country has its own dialect. The aim of this project is to build a model that predicts the dialect given the text.

## Overview

In this project, we have explored various machine learning models such as Support Vector Machine (SVM), XGBoost, and Multinomial Naive Bayes (MultinomialNB). After experimentation, we found that the MultinomialNB model achieved the highest accuracy of 79%.

Additionally, we utilized ARABERT, a BERT-based model from Hugging Face, to further improve the accuracy of our predictions. With ARABERT, we achieved an accuracy of 82%.

## Project Structure

- `data/`: Contains database and cleaned data used for training and evaluation, as well as the data fetching script (`fetch_data.py`) for easy access to the data.
- `Models/`: Contains saved model parameters.
- `Notebooks/`: Jupyter notebooks used for data exploration, model training, and evaluation.
- `Preprocessing/`: Jupyter notebooks for the data cleaning process.
- `Web App/`: Contains the web app script for deployment.

## WebApp

![WebApp Video](https://drive.google.com/file/d/1wdt8hxwRWwiEsAmhwFH25xBUNe8bNZPE/view?usp=drive_link)
