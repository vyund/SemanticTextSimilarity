# CSDS497_SemanticTextSimilarity
### Comparing four model architectures on the Semantic Text Similarity task

### Background:
Semantic text similarity is a metric defined to measure the distance between sentences, documents, terms, etc. This distance is based on the semantic content or meaning of the items. This is estimated through statistical methods, and the idea of semantic relatedness has been approached through deep learning models. This project serves to implement and compare four deep learning model architectures on the semantic text similarity task.

I investigate the performance of deep learning on the semantic text similarity task by exploring threepublished model architectures (Manhattan LSTM, Bidirectional LSTM, and CNN-LSTM), and extend my project by proposing my own modified model architecture (Bi-CNN-LSTM) that utilizes techniques taken from the three investigated models.

### Requirements:
* numpy 1.19.2
* matplotlib 3.3.2
* pandas 1.1.5
* nltk 3.5
* gensim 3.8.3
* scikit-learn 0.23.2
* scipy 1.5.2
* keras 2.4.3

### Data:
Two sentence relatedness datasets are used to train and evaluate model performance. Sentence relatedness is the task of determining how semantically similar two sentences are. 

#### SICK [1]:
The SICK dataset consists of 19,996 sentence pairs split into a 10,000/9,996 training/test split. Each sentence pair is annotated with a relatedness score, which is on a scale of 1-5. This value corresponds to sentence relatedness, and is determined by averaging scores given by a panel of 10 different individuals. This score is normalized to be between 0-1 for evaluation.

#### Sem-Eval [2]:
The Sem-Eval dataset consists of 3094 sentence pairs split into a 1485/1609 training/test split. Similarly to the SICK dataset, the sentence pairs are annotated with a relatedness score of the same scale. The Sem-Eval dataset relatedness scores are determined by averaging scores from a panel of 5 different individuals. The same normalization is applied for model evaluation purposes.

### Model Architectures:
Details for each model architecture can be found in the original publications (references below).

* Manhattan LSTM (MaLSTM) [3]
* Bidirectional LSTM (BiLSTM) [4]
* CNN-LSTM [5]

* Bidirectional CNN-LSTM (Bi-CNN-LSTM)

My extension of the 3 published architectures involves utilizing the siamese CNN structure of the CNN-LSTM to first train the model on the local context of the input sentence pair. Then, I attach the four bidirectional LSTM layers found in BiLSTM. Finally, I use a dense feed forward layer to output the predicted sentence relatedness. I utilize early stopping and dropout layers to prevent overfitting of the model, as this architecture is much more complex than the previous 3.

### Results:
Results were gathered and averaged over 5 runs for each model architecture. Data is randomly split into  training/validation sets prior to training, and the test set remains constant. The reported results are the models predictive performance on the test set, based on the MSE, Pearson, and Spearman metrics. Optimal hyper-parameters are first found for each model, and are stored as default parameters in the code base. Parameters are found using grid search, and are deemed optimal if training loss plots do not show indication of over-fitting. These hyper-parameters are the same as what was given in model implementation sections.

![Results](/results.PNG)

### References:
[1] Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. The SICK (Sentences Involving Compositional Knowledge) dataset for relatedness and entailment. Zenodo, May 2014.

[2] Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. SemEval-2012 task 6: A pilot on semantic textual similarity. In *SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 385–393, Montréal, Canada, 7-8 June 2012. Association for Computational Linguistics.

[3] Jonas Mueller and Aditya Thyagarajan. Siamese recurrent architectures for learning sentence similarity. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI’16, page 2786–2792. AAAI Press, 2016.

[4] Paul Neculoiu, Maarten Versteegh, and Mihai Rotaru. Learning text similarity with Siamese recurrent networks. In Proceedings of the 1st Workshop on Representation Learning for NLP, pages 148–157, Berlin, Germany, August 2016. Association for Computational Linguistics.

[5] Elvys Linhares Pontes, Stéphane Huet, Andréa Carneiro Linhares, and Juan-Manuel Torres-Moreno. Predicting the semantic textual similarity with siamese CNN and LSTM. CoRR, abs/1810.10641, 2018.
