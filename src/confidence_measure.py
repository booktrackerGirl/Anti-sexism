## LOOK INTO IT

import argparse
import os
import pickle
import random

import config
import numpy as np
import torch
import pandas as pd
import numpy as np
import pdb

### Modified from Kuhn et al, Semantic Uncertainty ICLR 2023: https://github.com/lorenzkuhn/semantic_uncertainty & DynamicQA: https://github.com/copenlu/dynamicqa


def get_overall_log_likelihoods(list_of_results):
    """Compute log likelihood of all generations under their given context.
    
    list_of_results: list of dictionaries with keys:
    
    returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
             that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
    """

    result_dict = {}

    list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods', 'sequence_embeddings',\
                    'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                    'neg_log_likelihood_of_most_likely_gen']

    for key in list_of_keys:
        list_of_ids = []
        overall_results = []
        for model_size, result in list_of_results:
            results_per_model = []
            for sample in result:
                average_neg_log_likelihoods = sample[key]
                list_of_ids.append(sample['id'])
                if torch.is_tensor(average_neg_log_likelihoods):
                    results_per_model.append(average_neg_log_likelihoods)
                # else:
                #     type(average_neg_log_likelihoods[0]) == str:
                #         results_per_model.append(torch.Tensor)
                    
                    
            # breakpoint()
            if key != 'group_ids':
                results_per_model = torch.stack(results_per_model)

            overall_results.append(results_per_model)

        if key != 'sequence_embeddings' and key != 'group_ids':
            overall_results = torch.stack(overall_results)

        result_dict[key] = overall_results

    result_dict['ids'] = list_of_ids
    return result_dict


def get_log_likelihood_variance(neg_log_likelihoods):
    """Compute log likelihood variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

    return variance_of_neg_log_likelihoods