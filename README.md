# Abstractive-text-summarization
Using a deep learning model that takes advantage of LSTM and a custom Attention layer, we create an algorithm that is able to train on news dataset and existent summaries to generate brand new summaries of its own.

Model description:
![image](https://user-images.githubusercontent.com/46419383/128472337-4c15ede8-12c9-4546-af6a-0d2483299585.png)
Output:
![image](https://user-images.githubusercontent.com/46419383/128472558-48f773d1-3448-4d5d-96da-a0c92dfa3b59.png)

Objective:
To generate summaries of news.

Methodology:
For this case, I wanted to have the machine learning model to learn the context of the text itself. In order for this to be done, some form of training had to be executed; unlike the extractive summarisation algorithm, which did not require a training set whatsoever.
Hence, a training set had to be curated. For this project, the training set would be reviews from Amazon followed by their summaries. In a nutshell, this model trains by feeding the full text review into the system and adjusting the weights by backpropogation using the summary as the response value.
There are 3 parts to the entire model: The Encoder, The Decoder, The Attention Layer.

The Bidirectional Encoder - Is an LSTM model that takes in the full text of the review and run in both forward and reverse direction to generate a context vector which is then feed to attention layer and decoder.
The Decoder - Is an LSTM model that takes in the learned context vector from The Encoder to generate the summary.
The Attention Layer - Adjusts the attention of The Decoder based on the contextual understanding of both the full text review and the full text summary. The intuition behind the Attention Layer is basically finding and focusing on the essence of the question or text. For example, if the question was, "What animal do you like?", simply focusing on the word 'animal' would get you to consider all animals for this context. Thereafter, focusing on the word 'like' would get you to answer with your favorite animal straightaway. Hence, instead of fully considering all 5 words, by focusing on less than half the question and 'blurring' the rest, we are able to generate a response already.

overview:
![MODEL (4)](https://user-images.githubusercontent.com/46419383/128473547-6fcc98df-eebb-4750-87af-1eed3bbaa458.png)
