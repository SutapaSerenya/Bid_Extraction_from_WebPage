# Bid_Extraction_from_WebPage
Extraction of bid numbers from web page

The model consists of a different layers , first an embedding layer for word embeddings.
Then a dropout Layer to overcome overfitting of dataset.
A Concolutional 1D layer to extract features followed by a max pooling layer.
An LSTM layer for building up long and short term dependencies.
Finally a dense layer with softmax activation to predict the probability of a bid number.

There is a pickle file (tokenizer.pickle)consisting of tokens to be loaded for the models to make predictions. 

There is a weights file(checkpoints) to be loaded on the model for it to give predictions as output.

The BeautifulSoup module is used to open the link provided in the link variable and parsed using the lxml parser.
The desired tags to be checked are mentioned in the desired_elements.

The tags extracted from the web page is then formatted as in to remove extra spaces , tab space and newline characters.
The text collected from these tags are then added to a dictionary and finally converted into dataframe.
Then the texts in the dataframe is tokenized and finally fed into the model for prediction
which outputs the probabitlity of each line being a bid number.

