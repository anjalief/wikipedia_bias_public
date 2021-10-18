# This file is mostly depreciated, use tfidf_matching
import pickle
import random
from collections import Counter, defaultdict
from utils import load_people, load_filtered_categories, get_people_sample, get_filtered_people_with_topics, kl_divergence, get_cohen_d, load_categories
import math
import numpy as np
from numpy import mean, var, std
import math
import time
import multiprocessing
from basic_log_odds import write_log_odds
import os
import argparse

# import spacy
# nlp = spacy.load("en_core_web_sm")

"The standardized bias is calculated by taking the difference in means"
"for a given covariate between the treatment and control groups and dividing"
"by the standard deviation in the treatment group"
# Categories that we don't care about should already be filtered out
def get_standardized_bias(treatment_people, control_people, cats, return_all=False):
    treatment_cats = Counter()
    for _,t in treatment_people.items():
        treatment_cats.update(t['categories'])

    control_cats = Counter()
    for _,t in control_people.items():
        control_cats.update(t['categories'])

    num_control = len(control_people)
    num_treatment = len(treatment_people)

    cat_to_std_bias = defaultdict(float)
    for c in cats:
            treatment_mean = treatment_cats[c] / num_treatment
            control_mean = control_cats[c] / num_control
            treatment_std = math.sqrt((treatment_mean) * (1 - treatment_mean))
            control_std = math.sqrt((control_mean) * (1 - control_mean))
            pooled_std = math.sqrt(((treatment_std * treatment_std) + (control_std * control_std))/ 2)
            if pooled_std == 0 and treatment_mean == 0 and control_mean == 0:
                cat_to_std_bias[c] = (0, 0, 0)
            else:
                cat_to_std_bias[c] = ((abs(treatment_mean - control_mean) / pooled_std), treatment_mean, control_mean)

    # print_count = 0
    # for k,v in sorted(cat_to_std_bias.items(), key=lambda item: item[1][0], reverse=True):
    #     if print_count > 10:
    #         break
    #     print(k, v)
    #     print_count += 1

    all_bias = [v[0] for k,v in cat_to_std_bias.items()]

    if return_all:
        return all_bias

    avg_bias = sum(all_bias) / len(all_bias)
    num_over25 = len([a for a in all_bias if a > 0.25])
    num_over1 = len([a for a in all_bias if a > 0.1 and a <= 0.25])
    num_over01 = len([a for a in all_bias if a > 0.01 and a <= 0.1])
    num_over001 = len([a for a in all_bias if a > 0.001 and a <= 0.01])
    num_over0001 = len([a for a in all_bias if a > 0.0001 and a <= 0.001])
    num_under0001 = len([a for a in all_bias if a <= 0.0001])
    return avg_bias, num_over25, num_over1, num_over01, num_over001, num_over0001, num_under0001

# Weighting options are:
    # prioritize matching on categories that are frequent in the treatment data
    # prioritize matching on categories that are very different between treatment and control
    # prioritize matching on categories that are rare in total data, since they are more specific
def get_category_weights(cat_info, treatment_people, control_people, weight_type):
    if weight_type == "inverse_by_freq": 
        total_num_people = sum([i for c,i in cat_info.items()])
        return {c:math.log(total_num_people / cat_info[c]) for c in cat_info}
    # if match_method == "disparity":
    #     pass

def get_text_lengths(people_dict):
    return [len((t["text"])) for i,t in people_dict.items()]


def get_num_categories(people_dict):
    return [len((t["categories"])) for i,t in people_dict.items()]

def get_avg_log_odds_scores(sample1, sample2, vocab):
    count1 = Counter()
    count2 = Counter()
    for _,info in sample1.items():
        count1.update(info["text"])
    for _,info in sample2.items():
        count2.update(info["text"])
    prior = Counter()
    prior.update(count1)
    prior.update(count2)

    delta = write_log_odds(count1, count2, prior)

    scores = [s for w,s in delta.items() if w in vocab]
    normal_mean = np.mean(scores)
    normal_stdev = np.std(scores)
    abs_scores = [abs(s) for s in scores]

    # What if we only take the most polar words?
    abs_scores = sorted(abs_scores, reverse=True)[:200]

    polar_mean = np.mean(abs_scores)
    polar_std = np.std(abs_scores)
    return normal_mean, normal_stdev, polar_mean, polar_std

def get_kl_divergence(sample1, sample2):
    # we fixed lda at 100 topics
    def get_distribution(sample):
        # TODO: this is pretty sketchy. We're just doing additive smoothing
        # Is there a better way??
        counts = np.ones(100) / 1000
        for _,info in sample.items():
            for x in info["lda"]:
                counts[x[0]] += x[1]
        return counts / np.sum(counts)

    counts1 = get_distribution(sample1)
    counts2 = get_distribution(sample2)
    return kl_divergence(counts1, counts2), kl_divergence(counts2, counts1)


def assess_match(sample1, sample2, vocab, invalid_cats = None):
    valid_cats = load_filtered_categories()
    if invalid_cats is not None:
        valid_cats = set([v for v in valid_cats if not v in invalid_cats])

    avg_bias = get_standardized_bias(sample1, sample2, valid_cats)

    length1, length2 = get_text_lengths(sample1), get_text_lengths(sample2)
    text_length_d = get_cohen_d(length1, length2)
    length1 = sum(length1) / len(length1)
    length2 = sum(length2) / len(length2)

    cats1, cats2 = get_num_categories(sample1), get_num_categories(sample2)
    num_cats_d = get_cohen_d(cats1, cats2)
    cats1 = sum(cats1) / len(cats1)
    cats2 = sum(cats2) / len(cats2)

    mean, stdev, polar_mean, polar_std = get_avg_log_odds_scores(sample1, sample2, vocab)
    kl1, kl2 = get_kl_divergence(sample1, sample2)

    return list(avg_bias) + [num_cats_d, cats1, cats2, text_length_d, length1, length2, mean, stdev, polar_mean, polar_std, kl1, kl2]

def print_eval(eval):
    for e in eval:
        print(e, end=" ")
    print("")
