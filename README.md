# Disaster Response Pipeline Project
### Table of Contents

1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions <a name="instructions"></a>

1) Run the following commands in the project's root directory to set up your database and model.

 To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponses.db models/classifier.pkl

2)Run the following command in the app's directory to run your web app. python run.py

3)Go to http://0.0.0.0:3001/ 

## Project Motivation<a name="motivation"></a>

For this project, I was interested in creating a Disaster Response WebApp using Flask where an emergency worker can input a new message and get classification results in several categories. This project was aimed to aid the Disaster Response organizations when they have the least capacity to tackle situations.


## File Descriptions <a name="files"></a>

There are 2 datasets used here provided by Figure Eight.  The raw files for each of the 2 datasets is provided inside the 'data' folder. These data were collected from social media and direct communication sources.  

The files are to be organized the way they are currently to run properly. The data folder contains the 2 datasets along with the ETL Pipeline.

The model folder contains Machine Learning pipeline. This folder should also contain the classifier.pkl model pipeline. However, the file size was too large to upload.

The whole folder structure of this Respository is as below:

###### app
| - template \n
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

###### data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to

###### models
|- train_classifier.py
|- classifier.pkl # saved model

## Results<a name="results"></a>

The main findings of the code can be found when the website is loaded. It contains 3 visualization and a message search bar.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for the project learning.  Otherwise, feel free to use the code here as you would like! 
