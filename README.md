# Comparing approaches to analyze sentiment :memo:
This repository contains the implementation of the related article that was written as a part of Machine learning course at the Faculty of Computer and Information Science at the University of Ljubljana.

Related article: [Comparing approaches to analyze sentiment](https://github.com/lzontar/Sentiment-Analysis-ML/blob/master/article/comparing-approaches-to-analyze-sentiment.pdf)

# Repo structure :blue_book:
This repository contains folders:
* ```data/``` - contains data necessary for execution and results. Root contains the data from Kaggle.
  * ```data/preprocessing``` - contains all the generated preprocessed data files. These are generated during execution of `evaluation.py` to enhance following executions during development.
  * ```data/glove``` - contains GloVe embeddings that you should download from [GloVe](https://nlp.stanford.edu/projects/glove/)
  * ```data/results/``` - contains the ```*.pkl``` files, where we exported data results.
  * ```data/results/fig/``` - contains all the visualizations and results of our work.
  * ```data/results/models/``` - contains all the primary models that were generated during execution of ```evaluation.py```.  
* ```util/``` - contains Python helper files that are used in the main Python script ```evaluation.py```.
* ```notebooks/``` - contains Jupyter notebooks. 
* ```article/``` - contains the article: [Comparing approaches to analyze sentiment](https://github.com/lzontar/Sentiment-Analysis-ML/blob/master/article/comparing-approaches-to-analyze-sentiment.pdf).
* ```scripts/``` - contains ```evaluation.py```, the main Python script used to generate results. 

# Reproducing results :snake:
To reproduce my results, you will have to download the dataset I used: [Sentiment140](https://www.kaggle.com/kazanova/sentiment140). Unzip it to folder ```data/``` in the root of the repository.

After you successfully forked this repo and downloaded the dataset, you will have to install Python dependencies using `conda`:
```
conda env create -f environment.yml
```
Or `pip`:
```
pip install -r requirements.txt
```
To install ```en_core_web_sm``` execute the following command:
```
python -m spacy download en_core_web_sm
```

Now your environment is ready to go. Executing ```python scripts/evaluation.py``` produces some interesting results. :partying_face: :clinking_glasses:


