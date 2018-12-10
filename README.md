# Disaster Response Pipeline Project
### Introductions:
This project is designed by Udacity's Data Scientiest Nanodegree and completed by me. This project builds a pipeple that analyzes the text 
messages which is collected from various source and tries to classify the messages into different categories. In this way, people in the
disaster area could respond quicklly and more lives could be saved.

There are three sequential steps which also serve three different goals in this repository. The first one is the famous ETL (Extract,Transform,Load) 
pipeline which extract data from different sources and transform and formalize them into a coherrent form and save them as .db file for future use.
The second step is using machine learning model (decision tree in this application) to learning the relation between plain text and their actual meaning.
The third and final step is building a Web App in which some basic visualizations of the data are posted and model is embedded into the website so online
testing is possible. These three steps could be used independently.

### Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important files
- 'data/process_data.py': This file is used to perform ETL pipeline and store formalized data into database file.
- 'models/train_classifier.py' this file is used to develop a model on given datasets. Data preprocessing, grid search, model evaluation,
are conducted, and model is stored in a pickle file finally.
- 'run.py': This file defines what and how to visualize the datasets on a web page, it also enable use to do online query.

### useful resource
1. the datasets are provided by figure eight company. (https://www.figure-eight.com/)

2. Udacity's data scientist nanodegree. (https://www.udacity.com/course/data-scientist-nanodegree--nd025)
