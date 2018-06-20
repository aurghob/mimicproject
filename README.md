# mimicproject

This repository is for our Natural Language Processing Course Project.
It was initially worked on by 5 team members.
The dataset used here were the Electronic Health Records Datasets of MIMIC and I2B2.
We cleaned the data using Clamp.
We also wrote scripts to perform further cleaning.
The free text present in the dataset was used and we modelled the paragraphs of text using Gensim's doc2vec.
We used Deep Learning to perform sequence classification of the section text headers.
We used a hybrid model of Conditional Random Fields(CRF) and BiLSTM.