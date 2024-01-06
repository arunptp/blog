---
layout: post
title: Identifying Thematic Clusters using Latent Dirichlet Allocation 
subtitle: Explorative Survey in Energy Domain considering 5 premier journals over a period of 25 years
cover-img: /assets/img/LDA.png
thumbnail-img: /assets/img/LDA.png
share-img: /assets/img/LDA.png
tags: [Unsupervised Machine Learning, Latent Dirichlet Allocation, Topic Modelling, Text Analytics]
author: Arun Abraham Thomas
---

<br>

# Introduction

<br>

<div align="justify">

Identifying thematic clusters helps in reviewing literature in a given area of research.
Typically in traditional manual analysis, a researcher starts by defining topics of interest through
identifying a list of keywords. These keywords are used in online search engines for potential
relevant literature for the topic of interest. This traditional method of manual review has a few
shortcomings like choosing non-precise keywords, missing journals, and errors due to the large
volume of literature that a reseacher has to skim through (Delen & Crossland, 2008) .

</div>

<br>

<div align="justify">

Thus, if such thematic clusters are developed and made available for multiple journals, it
will aid and support in the traditional manual analysis during the exploratory phase in research. It
helps in providing an overview of the research field and is particularly useful for novice
academics and reseachers.

</div>


<div align="justify">

Clustering of large number of documents is a non-trivial problem. Since any textual
documents can contain words from a vocabulary, representation of textual documents is high-
dimensional. High dimensionality of NLP problems leads to difficulty in clustering and failure of
“distance-metric” based algorithms.

</div>

<br>

<div align="justify">

The very basic step of NLP clustering algorithms is to reduce the dimensionality of the
problem, using bag of words, removing stop words etc. Even after such methods, the document
space remains very high dimensional. David M. Blei, Andrew Y. Ng and Michael I. Jordan, 2003 
in their seminal paper, introduced the Latent Dirichlet Algorithm (LDA), which thereafter has
been used to provide interpretable lower dimensional representation of documents.

</div>


<br>

# Topic Modelling

<br>

<div align="justify">
  
Topic Modelling has been used for exploratory analysis of large number of papers
(Jelodar, et al., 2019). It has gained much prominence in public policy, political science
and rhetoric analysis (J & BM, 2013) , (Debnath &amp; Bardhan, 2020) , finance (Feuerriegel, et al.,
2016) , (Shirota, et al., 2014) , biomedical research (HJ, et al., 2019) . Topic based clustering
model was used to group Indian legal documents into various clusters (Kumar & Raghuveer,
2012) . In the Energy domain, Topic Modelling has been used for tracking the evolution of the
policy of New Energy Vehicles (NEVs) in China (Jia & Wu, 2018) , sustainability reporting
(Székely &  Brocke, 2017)

  
  </div>

<br>
  
# Latent Dirichlet Algorithm (LDA)

<br>

<div align="justify">

Latent Dirichlet Algorithm (LDA) is a probabilistic generative algorithm that extracts the
thematic structure in a large corpus. The model considers that a topic is a distribution of words in
a vocabulary space and every document (described over the same vocabulary) is a distribution of
a small subset of those topics. These latent topics that the models learn are highly interpretable
and provide insights and qualitative understanding of the text corpora. For example, in our case,
we would expect the model to learn “renewable” as one of the topics and cluster all papers
related to renewable energy under the topic “renewable”. Once LDA generates a sparse
representation in much lower dimensions (1000 or lower), it is easily amenable for standard clustering algorithms. We have used the pyLDAvis package for interpreting the topics/clusters
and make interactive web-based visualizations of the journals (Sievert & Shirley, 2014) .
  
</div>

<br>


<div align="justify">

LDA calculates the joint probability distribution between the observed words in a paper
and the unobserved (the hidden structure). This method evaluates the frequency of words and the
semantics are ignored (Asmussen & Møller , 2019) . LDA is an unsupervised algorithm and a key
feature of such unsupervised machine learning methods is hyperparameters like the number of
topics/clusters. Generally, cross validation using perplexity score is used to find the optimal
number of topics/clusters. A larger number of topics/clusters provide a more detailed clustering
while a low number of topics/clusters provide a general overview. Depending on the research
questions trying to be answered, the number of topics can differ.
  
 </div>

<br>

# The Results

<br>

<div align="justify">

It is interesting to note that the methodology and algorithms described above
automatically identifies and clusters the broad areas of research like “Electricity Market”.
“Climate Change”, “Renewables” and groups various papers into the identified thematic groups
. The broad themes remain more or less the same for the three
niche, focused journals, i.e. Energy Economics, Energy Policy and Resource &amp; Energy
Economics. For the journals with broader scope, i.e. Applied Energy and Energy, technical
thematic clusters like “Power Systems”, “Thermal Storage”, “Fuel Cells” are automatically
identified in addition to other themes like “Renewables”, “Climate Change”. Thus, the algorithm
and methodology


</div>

---



> ## # An  interactive demo is explored <a href="https://arun-thomas.in/Explorative-Survey-of-Papers-in-Energy/">here!!!</a>



---
