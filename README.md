# EnCOT: Enhancing Global Clustering with Optimal Transport in Topic Modeling

This repository contains the official implementation of our paper:  
**"Topic Modeling for Short Texts via Optimal Transport-Based Clustering"**, accepted at **ACL 2025**.

[Conference Website (ACL 2025)](https://2025.aclweb.org/)

## Overview

Discovering topics and learning document representations in topic space are two crucial aspects of topic modeling, particularly in the short-text setting, where inferring topic proportions for individual documents is highly challenging. Despite significant progress in neural topic modeling, effectively distinguishing document representations as well as topic embeddings remains an open problem.

We propose **EnCOT**, a novel approach that introduces *abstract global clusters* to capture global semantic structures. Using the **Optimal Transport** framework, our method aligns:
- Document representations with global clusters.
- Global clusters with latent topics.

This dual alignment improves both the coherence of learned topics and the separation of document embeddings in topic space. Through extensive experiments, we demonstrate that **EnCOT** outperforms state-of-the-art methods on widely used benchmarks for short-text topic modeling.

## Setup Instructions

### 1. Environment Preparation

- Python version: `3.10.14`
- Required libraries:
    ```bash
    pip install numpy==1.26.3
    pip install scipy==1.10.1
    pip install sentence-transformers==2.7.0
    pip install torchvision==0.19.1
    pip install gensim==4.3.3
    pip install scikit-learn==1.5.1
    pip install tqdm==4.66.5
    pip install wandb==0.18.1
    pip install topmost==0.0.5
    ```

### 2. Java Dependencies

- Ensure Java is installed and available in your system's PATH.

- Download the [Palmetto JAR](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) and place it in:

  ```
  ./topmost/evaluations/palmetto.jar
  ```

### 3. Wikipedia Reference Corpus

- Download and extract the [Wikipedia_bd corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to the path:

  ```
  ./topmost/evaluations/wiki_data/
  ```

- Final folder structure should be:
    ```
    |- topmost
        |- evaluations
            |- palmetto.jar
            |- wiki_data
                |- wikipedia_bd/
                |- wikipedia_bd.histogram
    ```

## Running the Model

To train and evaluate EnCOT, execute the following script:

```bash
bash bash/NewMethod/top100/NewMethod_Biomedical_top100_cluster50.sh
```
