# GradPath Readme

This repository gives details for the paper "Generating and Evaluating Plausible Explanations for Knowledge Graph Completion", ACL 2024.

## Datasets
We use the following two datasets:
* [Kinship](https://github.com/ZhenfengLei/KGDatasets/tree/master/Kinship) from [Hinton 1990](https://archive.ics.uci.edu/dataset/55/kinship)
* [CoDEx](https://github.com/tsafavi/codex) from [Safavi & Koutra EMNLP 2020](https://aclanthology.org/2020.emnlp-main.669.pdf)

## Explanation Methods

### Baseline
The baseline explanation method that we use is [Gradient Rollback](https://github.com/carolinlawrence/gradient-rollback).
Please follow the instructions in this repository to generate explanation with this method.

### GradPath
Follow step 1 in [Gradient Rollback](https://github.com/carolinlawrence/gradient-rollback) to create a model and corresponding influence map.
GradPath provides an alternative method to compute step 2 of [Gradient Rollback](https://github.com/carolinlawrence/gradient-rollback).

## GradPath Explanations
### Hyperparameter Settings
To train the Kinship model following step 1 in [Gradient Rollback](https://github.com/carolinlawrence/gradient-rollback):

```
--epochs 100 \
--latent_expert_embedding_dim 10 \
--batch_size 1 \
--num_negative 13 \
--top_k 1 \
--optimizer 'adam' \
--learning_rate 0.001 \
--train_with_softmax True \
--expert 'ComplEx' \
```

To train the CoDEx model following step 1 in [Gradient Rollback](https://github.com/carolinlawrence/gradient-rollback):

```
--epochs 500 \
--latent_expert_embedding_dim 1024 \
--batch_size 1 \
--num_negative 20 \
--top_k 1 \
--optimizer 'adam' \
--learning_rate 0.3 \
--train_with_softmax True \
--expert 'DistMult' \
```

### Explanation files
Instead of generating gradient rollback explanations, we want to generate GradPath explanations.
This can be done by following step 2 in [Gradient Rollback](https://github.com/carolinlawrence/gradient-rollback),
Please replace [the for loop](https://github.com/carolinlawrence/gradient-rollback/blob/master/gr/xai/explanation_generator.py#L252)
with the below code and make sure that gradpath_explanations.py is available.

```
    prediction_triple = predictions[test_triple_idx][0]
    prediction_str = ' '.join(prediction_triple[0:3])

    current_prediction = {}
    current_prediction["correct"] = one_if_top1_correct[test_triple_idx]
    current_prediction["probability"] = predictions[test_triple_idx][0][3]

    topN_explanations = {}
    paths_all = get_all_paths(lookup_table, prediction_triple[0], prediction_triple[2], nodelimit)
    for ell in range(1,nodelimit):
        explanations = get_explanations(model_holder, influence_map, prediction_triple, paths_all, length=ell)
        if explanations is not None and len(explanations)>TopN:
            tem = sorted(explanations.items(), key=lambda item: item[1],reverse=True)
            for tem_i in range(TopN):
                k,v = tem[tem_i]
                topN_explanations[k] = v
        elif explanations is not None and len(explanations)>0:
            topN_explanations.update(explanations)
    current_prediction["explanations"] = topN_explanations

    explanation_dictionary [prediction_str] = current_prediction
```
