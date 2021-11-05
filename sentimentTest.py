import pickle
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt


def load_models():
    # Load the vectoriser.
    file = open('models/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()

    # Load the LR Model.
    file = open('models/LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel


def predict(vectoriser, model, text, file_names=None):
    # Predict the sentiment
    textdata = vectoriser.transform(text)
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])

    # Creating a new column for file names here.
    # it will take a list of files names
    df.insert(2, "file", file_names, True)
    df = df.replace([0, 1], ["Negative", "Positive"])


    return df

def report(df):


    # write tweet html to file
    path = df['file'].values[0]
    name = path[path.rindex('/')+1:]
    title = name.removesuffix('.csv')
    text_file = open(title+".html", "w")

    html_header = '''
              <link rel="stylesheet" type="text/css" href="df_style.css"/>
                {header}
            '''
    text_file.write(
        html_header.format(header=('<h1>'+title+' Sentiment Analysis Report</h1> '), classes='h1'))

    # write sentiment frequency html to file
    count = df.groupby(['sentiment']).count()
    count = count.drop('file', 1)
    amount = count.to_string()
    amount = amount.replace('sentiment', '')
    amount = amount.replace('text', '')
    amount = amount.replace('Negative', '<b> Negative:  </b> ')
    amount = amount.replace('Positive', '<b> Positive: </b>')

    html_column = '''
      <div class="row">
       <div class="column">
      <h2><u>Sentiment Frequency</u></h2>
      <p>{Column1}</p>
        </div>

        <div class="column" >
        <h2><u>Sentiment Count</u></h2>
          <p>{Column2Row1}</p>
           <br> 
      <h2><u> 2 Random Tweets with a Positive Sentiment: </u></h2
          <p>{Column2Row2}</p>
          <p>{Column2Row3}</p>
           <br> <br>
          <h2> <u>2 Random Tweets with a Negative Sentiment: </u></h2>
          <p>{Column2Row4}</p>
           <p>{Column2Row5}</p>

        </div>
      </div>
      '''

    postiveCount = df.loc[df.sentiment == "Positive", ['text']].sample(2).values
    negativeCount = df.loc[df.sentiment == "Negative", ['text']].sample(2).values

    # create figure
    fig, ax = plt.subplots()
    df.groupby('sentiment').size().plot(kind='pie', explode=(0, 0.1),
                                        colors=['red','green'],labels= ['Negative', 'Positive'],
                                        shadow = True,
                                        autopct='%.1f%%')
    ax.set_ylabel('')
    fig.savefig('graphs/'+title+'.png')
    plt.close(fig)

    text_file.write(
        html_column.format(Column1=(" <img  src = " + '"graphs/' + title + '.png"''"> <br> '),
                           Column2Row1=(amount + ' <br> <br> '), Column2Row2=(postiveCount[0]), Column2Row3=(postiveCount[1]),
                           Column2Row4=(negativeCount[0]), Column2Row5=(negativeCount[1])
                           , classes='img'))

    text_file.write('<h2> <u>50 Random Tweets</u> </h2>')

    html_table = '''
      <body>
        {table}
      </body>

    '''
    text_file.write(html_table.format(table=df.sample(50).drop('file', 1).to_html(classes='mystyle')))

    text_file.close()

if __name__ == "__main__":
    # Loading the models.
    vectoriser, LRmodel = load_models()

    path = './game_data/*.csv'
    lists_from_csv = []
    file_names = []
    for f in glob.glob(path):
        file = open(f, "r")
        csv_reader = csv.reader(file)
        for row in csv_reader:
            file_names.append(f)
            lists_from_csv.append(row[0])

    df = predict(vectoriser, LRmodel, lists_from_csv, file_names)
    cl = df.loc[df.file == './game_data/Minecraft.csv']
    report(cl)
    print(df)