# Fraud Case Study

## Step 1: EDA

#### Defining the Problem

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We labeled account types fraudster, fraudster_event, and fraudster_att as fraud. While spammers and terms of service violations also warrant investigation, that is not what our model is intended to do.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The class imbalance is large with only 8.6% of events being fraud. This makes accuracy a poor metric to go by as a model could easily get over 90% accuracy predicting only non-fraud.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Since our system is intended to flag potential fraud for investigation, we would rather have False Positives (spend resources investigating non-fraud) than False Negatives (letting someone get away with fraud). While we ideally want high values for all metrics, maximizing recall is the best course of action for this business case.

#### Features of Interest

Put EDA graphs here
 - payment type
 - body length, description
 - ignore num payouts as leakage
 - timestamps/seasonailty

## Step 2: Model Building

#### Basic Model

Created first model using features with greatest differences in distributions between the two classes.

Features:
- body_length
- channels
- delivery_method
- fb_published
- gts
- org_facebook
- org_twitter
- user_age

Tried various classifier models and Random Forest performed the best at 72% recall.

Decided to use more features and and Gradient Boosting did the best at 86% recall.

Put our model into src/model.py which saves a trained model as data/model.pkl.

#### Ideas

Perform NLP on body text.
- Scrapped due to very poor initial accuracy and being too time intensive.

## Step 3: Prediction Script

The model does all the work so the script just reads the data and loads the pickled model to run a predict.

## Step 4: Database

Set up a MongoDB database to store new data and predictions.

## Step 5: Web App

Booted up a simple flask server with a hello world and a test Tableau dashboard. Began working on a homepage by customizing a template.

## Step 6: Live Data

Modified the client to run and save predictions itself instead of calling on the prediction script. The init loads the model and connects to the database, so these tasks are not repeated every time a prediction is needed.

## Step 7: Dashboard

Used Tableau connected to MongoDB to create an interactive dashboard.

## Step 8: Deploy

Hosted static website on Amazon S3.
