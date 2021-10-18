# Helper funtions and data loacers
import pickle
import random
import os
from collections import Counter
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel, ttest_ind
from pandas import crosstab, Categorical
import numpy as np
from math import sqrt

CACHE_PATH = '/projects/tir3/users/anjalief/wikipedia_bias_public/cache'
VIZ_CACHE_PATH = '/projects/tir3/users/anjalief/wikipedia_bias_public/viz_cache'
MATCHED_CACHE_PATH = '/projects/tir3/users/anjalief/wikipedia_bias_public/matched_cache'

DROP_CATS =  ["Wiki", "mdy_dates", "dmy_dates", "All_stub_articles", "Living_people", "Articles", "template", "lacking_sources","articles", "language_sources", "CS1", "birth_missing_", "Infobox", "infobox", "containing_links", "lacking_titles", "Pages_with_", "Pages_using", "All_pages_", "Use_Indian_English", "Use_Australian_English", "Use_British_English", "_stubs"]

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# Adapted from https://machinelearningmastery.com/effect-size-measures-in-python/
# samples should be independent
def get_cohen_d(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
    d = (u1 - u2) / s
    return d

# count is an array-like value of the count of binary things
# nobs is the denomiators (total value in corpus) corresponding to count
# Mostly not using this because we're using methods for paired data instead
def get_ztest(count, nobs):
    stat, pval = proportions_ztest(count, nobs)
    return stat, pval

# This uses benjamini hochberg correction. Bonferroni or Holm are more stringent than needed
def get_mulitple_test(pvals):
    is_reject, corrected_pvals, _, _ = multipletests(pvals, alpha=0.1, method='fdr_bh')
    return is_reject, corrected_pvals

# Paper that recommends using paired test metrics: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3110307/

# "McNemar's test is a statistical test used on paired nominal data"

# Paired t-test should be fine with discrete data, as long as you have a large
# enough data sample. If your data set is small (<25) you can use Wilcoxon sign-ranked
# test. It looks like scipy version of Wilcoxon test defaults to a normal approximation
# anyway if your data size is > 25
# If you have nominal data (as opposoed to ordinal data), then can use McNemar's test
# https://stats.stackexchange.com/questions/303024/appropriate-tests-on-discrete-and-paired-data
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
# Both inputs must be array-like with the same shape
def paired_ttest(paired_data_1, paired_data_2):
    # Other value returned is test statistic
    _, pvalue = ttest_rel(paired_data_1, paired_data_2)
    return sum(paired_data_1) / len(paired_data_1), sum(paired_data_2) / len(paired_data_2), pvalue

def ttest(data_1, data_2):
    _, pvalue = ttest_ind(data_1, data_2)
    return sum(data_1) / len(data_1), sum(data_2) / len(data_2), pvalue

# Values 1 and 2 should be parallel arrays with binary 1/0 values for each
# https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
def binary_mcnemar_test(values_1, values_2):
    values_1_cat = Categorical(values_1, categories=[0, 1])
    values_2_cat = Categorical(values_2, categories=[0, 1])
    table = crosstab(values_1_cat, values_2_cat)
    # These settings use a Chi-squared approximation, which
    # is appropriate when all values in contingency table are
    # >25
    stats = mcnemar(table, exact=False, correction=True)
    # the other return value is "statistic" (test statistic), which I don't think is useful
    return sum(values_1) / len(values_1), sum(values_2) / len(values_2), stats.pvalue


def load_categories():
    cache_name = os.path.join(CACHE_PATH, "category_people.pkl")

    if not os.path.exists(cache_name):
        print("Please create the category cache. See the README file for where to download cached files")
        exit()

    with open(os.path.join(CACHE_PATH, "category_people.pkl"),"rb") as file:
        cat = pickle.load(file)
    return cat

# This is primarly used for getting categories that we can match on
def load_filtered_categories():
    cat = load_categories()
    return {n:c for n, c in cat.items() if len(c)>1 and all([w not in n for w in DROP_CATS])}

def load_people():
    cache_name = os.path.join(CACHE_PATH, 'tokenized_people.pickle')
    if os.path.exists(cache_name):
        with open(cache_name, "rb") as file:
            people = pickle.load(file)
        return people
    else:
        print("Please create the tokenized people cache. See the README file for where to download cached files")
        exit()

def get_people_sample(people, size = 100, seed = None, fixed = False):
    if fixed:
        sample_pairs = ['T-Pain', 'Barack_Obama', 'Meryl_Streep', 'Yuna_Kim', 'Amitabh_Bachchan', 'Tim_Cook', 'Ron_Berger_(professor)', 'Kevin_Barnes_(American_football)']
        return {s:people[s] for s in sample_pairs}
    names = people.keys()

    random.seed(seed)
    sample = random.sample(names, size)
    return {s:people[s] for s in sample}

def print_lda_topics(path_to_lda_model):
    lda = LdaModel.load(path_to_lda_model)
    people, _ = get_filtered_people()
    texts = [info["text"] for p,info in people.items()]
    dictionary = Dictionary(texts)

    print(dictionary.id2token)
    for x in lda.show_topics(formatted=False):
        words = " ".join([str(dictionary.id2token[int(w[0])]) for w in x[1]])
        print(x[0], words)

def get_filtered_people_with_topics():
    cache_name = os.path.join(CACHE_PATH, "filtered_people_topics.pickle")
    if os.path.exists(cache_name):
        with open(cache_name,"rb") as file:
            people, vocab = pickle.load(file)
            return people, vocab

    # they should already be filtered and tokenized
    people, vocab = get_filtered_people()
    texts = [info["text"] for p,info in people.items()]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    for tok,i in dictionary.token2id.items():
        dictionary.id2token[i] = tok

    print("Starting LDA training")
    lda = LdaModel(corpus, num_topics=100)
    for x in lda.show_topics(formatted=False, num_topics=100):
        words = " ".join([str(dictionary.id2token[int(w[0])]) for w in x[1]])
        print(x[0], words)
    lda_model_path = os.path.join(CACHE_PATH, "filtered_people.lda")
    lda.save(lda_model_path)
    print("Done LDA training. Saved to", lda_model_path)

    for p,info in people.items():
        bow = dictionary.doc2bow(info["text"])
        people[p]["lda"] = lda[bow]

    with open(cache_name,"wb") as file:
        pickle.dump((people, vocab), file, protocol=4)
    return people, vocab


def get_filtered_people():
    cache_name = os.path.join(CACHE_PATH, "filtered_people.pickle")
    if os.path.exists(cache_name):
        with open(cache_name,"rb") as file:
            people, vocab = pickle.load(file)
            return people, vocab

    people = load_people()
    # Tokenize everything, and count vocab
    # Drop people that we think are junk
    vocab = Counter()
    new_people = {}
    print("Number of people before", len(people))
    for p,info in people.items():
        cats = set(info["categories"])
        # Drop people without at least 2 categories
        if len(cats) < 2:
            continue

        # Drop people who are stubs
        if any(["_stub" in x for x in info["categories"]]):
            continue

        # Drop people with fewer than 100 tokens
        if len(people[p]["text"]) < 100:
            continue
        info["categories"] = cats
        new_people[p] = info
        vocab.update(people[p]["text"])
    people = new_people
    print("Number of people after filtering", len(people))
    print("Vocab size", len(vocab))

    with open(cache_name,"wb") as file:
        pickle.dump((people, vocab), file, protocol=4)
    return people, vocab

def process_data(tupl, use_propensity, drop_matches=True):
    treatment, matched_sample, matched_pairs = tupl

    if use_propensity:
        prop_scores = [x[2] for x in matched_pairs]
        thresh = np.mean(prop_scores) + np.std(prop_scores)

    print("Orig sizes", len(treatment), len(matched_sample))
    control_counts = Counter()
    # Drop people with too few category matches
    treatment_drop = set()
    control_drop = set()
    new_matched_pairs = []
    prop_scores = []
    for x in matched_pairs:
        control_counts[x[1]] += 1
        if use_propensity:
            if x[2] > thresh:
                treatment_drop.add(x[0])
                control_drop.add(x[1] + "::" + x[0])
            else:
                new_matched_pairs.append(x)
        else:
            cats = [c for c in x[3] if not 'alumn' in c and not 'births' in c]
            if len(cats) < 2:
                treatment_drop.add(x[0])
                control_drop.add(x[1] + "::" + x[0])
            else:
                new_matched_pairs.append(x)

    if drop_matches:
        print("Dropping", len(treatment_drop), len(control_drop))
        treatment = {t:i for t,i in treatment.items() if not t in treatment_drop}
        matched_sample = {t:i for t,i in matched_sample.items() if not t in control_drop}
        matched_pairs = new_matched_pairs

    # Add back dropped categories
    for p,info in treatment.items():
        treatment[p]["categories"] = set(list(info["tfidf"].keys()))

    return treatment, matched_sample, matched_pairs

def get_invalid_cats(invalid_keywords = None, original_treatment_cats = None, treatment_names_list = None, control_names = None, people = None, cache_name = None):
    cache_name = os.path.join(CACHE_PATH, cache_name)
    if os.path.exists(cache_name):
        invalid_cats = [l.strip() for l in open(cache_name).readlines()]
        return set(invalid_cats)

    comparison_cats = [people[n]["categories"] for n in control_names if n in people]
    comparison_cats = set([item for sublist in comparison_cats for item in sublist])

    invalid_treatment_cats = set()
    all_treatment_cats = set()
    for treatment_names in treatment_names_list:
        treatment_cats = [people[n]["categories"] for n in treatment_names if n in people]
        treatment_cats = set([item for sublist in treatment_cats for item in sublist])

        treatment_only_cats = treatment_cats - comparison_cats
        invalid_treatment_only_cats = [c for c in treatment_only_cats if any([x in c for x in invalid_keywords])]
        invalid_treatment_cats.update(invalid_treatment_only_cats)
        all_treatment_cats.update(treatment_cats)

    comparison_only_cats = comparison_cats - all_treatment_cats

    invalid_comparison_only_cats = [c for c in comparison_only_cats if any([x in c for x in invalid_keywords])]

    all_invalid_cats = set(list(original_treatment_cats) + invalid_treatment_only_cats + invalid_comparison_only_cats)
    with open(cache_name, "w") as fp:
        fp.write("\n".join(list(all_invalid_cats)))

    return all_invalid_cats

def get_code_to_language():
    code_to_language  = {}
    with open(os.path.join(CACHE_PATH, "language_codes.txt")) as fp:
        for line in fp.readlines():
            parts = line.strip().split("|")
            code_to_language[parts[0]] = parts[2]
    return code_to_language


if __name__ == "__main__":
    get_filtered_people_with_topics()
    # print_lda_topics()
