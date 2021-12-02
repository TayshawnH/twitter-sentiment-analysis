# Twitter Sentiment Analysis 

This program runs in Python 3.9, and uses packages that will NOT install or work properly in Python 3.10. Ensure you have Python 3.9 installed or create a separate virtual environment. For help configuring a virtual environment, go to https://docs.python.org/3/library/venv.html.

## Step 1:  Install requirements
```bash
Run the command [pip3 install -r requirements.txt] in your command prompt.
```

## Step 2: Download training data from Kaggle

https://www.kaggle.com/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv

In your downloaded project location, navigate to folder "dataset" and add Kaggle training data.

Rename training data file to "processed_tweet_dataset.csv".

## Step 3: Create Model

Run "training_model_LR.py".

## Step 4: Generate Reports

Run "sentimentTest.py". This will prompt you to choose a report to generate. Reports are stored in the general "twitter-sentiment-analysis" folder.

## Step 5: View Reports!

Open any generated HTML file in your preferred web browser. 

## NOTE FOR WINDOWS USERS
If you are on a windows machine, running the command from step 1 might not work for you. In that case, open the requirements file and manually install the required packages listed. 

