# Primary file for making matches
from matching_tools import assess_match, get_category_weights, print_eval, get_num_categories
from utils import load_people, load_filtered_categories, get_people_sample, get_filtered_people_with_topics, kl_divergence, get_cohen_d, load_categories
from collections import Counter
import math
import random
import argparse

TUNE_CATEGORIES = ["Category:USL_League_Two_players",
        "Category:UK_MPs_2019–",
        "Category:20th-century_American_racing_drivers",
        "Category:Austrian_footballers",
        "Category:American_male_child_actors",
        "Category:20th-century_American_guitarists",
        "Category:San_Francisco_Giants_players",
        "Category:21st-century_American_poets",
        "Category:Barnsley_F.C._players",
        "Category:American_feminists"]

def get_pivot_cosine_sim(tgt_info, cand_info, cand_name):
    common_cats = tgt_info["categories"].intersection(cand_info["categories"])
    dot = sum([tgt_info["pivot_tfidf"][c] * cand_info["pivot_tfidf"][c] for c in common_cats])
    return cand_info, cand_name, dot / tgt_info["pivot_norm"] * cand_info["pivot_norm"], common_cats

def get_cosine_sim(tgt_info, cand_info, cand_name):
    common_cats = tgt_info["categories"].intersection(cand_info["categories"])
    dot = sum([tgt_info["tfidf"][c] * cand_info["tfidf"][c] for c in common_cats])
    return cand_info, cand_name, dot / tgt_info["norm"] * cand_info["norm"], common_cats

def get_num_cats(tgt_info, cand_info, cand_name):
    common_cats = tgt_info["categories"].intersection(cand_info["categories"])
    return cand_info, cand_name, len(common_cats), common_cats

def get_propensity_diff(tgt_info, cand_info, cand_name):
    common_cats = tgt_info["categories"].intersection(cand_info["categories"])
    return cand_info, cand_name, abs(tgt_info["propensity_score"] - cand_info["propensity_score"]), common_cats

def get_percent_cats(tgt_info, cand_info, cand_name):
    common_cats = tgt_info["categories"].intersection(cand_info["categories"])
    return cand_info, cand_name, len(common_cats) / len(cand_info["categories"]), common_cats

def make_matches(target_people, candidate_people, sim_metric, max_match_count):
    matched_dict = {}
    treatment_dict = {}
    matched_pairs = []
    candidate_counts = Counter()
    for p,info in target_people.items():
        match_scores = [sim_metric(info, cand, name) for name,cand in candidate_people.items()]
        match_scores = [x for x in match_scores if x is not None]
        if len(match_scores) == 0:
            continue

        match_scores.sort(key=lambda k: k[2], reverse=True)
        match_info, match_name, score, common_cats = match_scores[0]

        matched_dict[match_name + "::" + p] = match_info
        treatment_dict[p] = info
        matched_pairs.append((p, match_name, score, common_cats))
        candidate_counts[match_name] += 1

        if candidate_counts[match_name] >= max_match_count:
            del candidate_people[match_name]

    return treatment_dict, matched_dict, matched_pairs

def compute_propensity_scores(target_people, candidate_people, valid_cats, tf_idf=False):
    cats_to_idx = {c:i for i,c in enumerate(valid_cats)}

    rows = []
    cols = []
    data = []
    row_idx = 0
    for t,tgt_info in target_people.items():
        for c in tgt_info['categories']:
            if c in cats_to_idx:
                rows.append(row_idx)
                cols.append(cats_to_idx[c])
                if tf_idf:
                    data.append(tgt_info["tfidf"][c])
                else:
                    data.append(1)
        row_idx += 1

    for t,cand_info in candidate_people.items():
        for c in cand_info['categories']:
            if c in cats_to_idx:
                rows.append(row_idx)
                cols.append(cats_to_idx[c])
                if tf_idf:
                    data.append(cand_info["tfidf"][c])
                else:
                    data.append(1)
        row_idx += 1
    print("Sizes", len(target_people), len(candidate_people), len(target_people)+len(candidate_people))

    all_feats = csr_matrix((data, (rows, cols)))
    print("Feature shape", all_feats.shape)
    target_labels = np.ones(len(target_people))
    candidate_labels = np.zeros(len(candidate_people))
    all_labels = list(target_labels) + list(candidate_labels)
    model = LogisticRegression(max_iter=5000)
    model.fit(all_feats, all_labels)

    probs = model.predict_proba(all_feats)
    for i,t in enumerate(target_people):
        target_people[t]["propensity_score"] = probs[i, 1]

    start_idx = len(target_people)
    for i,t in enumerate(candidate_people):
        candidate_people[t]["propensity_score"] = probs[start_idx + i, 1]
    print("Candidate propensity score stats", np.mean(probs[start_idx:,1]), np.max(probs[start_idx:,1]), np.min(probs[start_idx:,1]))
    print("Target Propensity score stats", np.mean(probs[:start_idx,1]), np.max(probs[:start_idx,1]), np.min(probs[:start_idx,1]))
    preds = model.predict(all_feats)
    print("F1, Precision, Recall", f1_score(all_labels, preds), precision_score(all_labels, preds), recall_score(all_labels, preds))
    print("Accuracy", accuracy_score(all_labels, preds))


# For category eval, track who is in the category
def prepare_people(slope = 0.4, cat = None, verbose=False, remove_tune_people=False):
    valid_cats = load_filtered_categories()
    people, vocab = get_filtered_people_with_topics()

    if remove_tune_people:
        tune_people = [s.strip() for s in open("./tune_people.txt").readlines()]
        if verbose:
            print(len(people))
        for t in tune_people:
            del people[t]
        if verbose:
            print("Deleted tuning people", len(people))

    if cat is not None:
        del valid_cats[cat]

    valid_cats = set(valid_cats.keys())

    vocab = vocab.most_common(5000)
    vocab = {x[0]:x[1] for x in vocab}

    # Drop the useless categories from all people
    # build new valid cats set
    new_people = {}
    cat_counts = Counter()
    people_in_cat = []
    for p,info in people.items():
        cats = info["categories"].intersection(valid_cats)
        if len(cats) < 2:
            continue
        # We need to check info["categories"] because "cat"
        # was deleted from valid cats. We want to check
        # after we intersect with valid_cats because we don't want
        # to add people who have too few valid categories
        if cat is not None and cat in info["categories"]:
            people_in_cat.append(p)

        new_people[p] = info
        new_people[p]["categories"] = cats
        cat_counts.update(cats)
    people = new_people

    cat_weights = get_category_weights(cat_counts, None, None, "inverse_by_freq")

    # we use this as the pivot
    num_cats_all = get_num_categories(people)
    pivot = sum(num_cats_all) / len(num_cats_all)
    if verbose:
            lns = [len(info["text"]) for p,info in people.items()]
            print("Text Length", sum(lns) /len(lns))
            print("Pivot", pivot)

    # catch the tf-idf vectors
    for p,info in people.items():
        # the term count is 1 for categories that are present and 0
        # for all others

        # Cache plain vectors
        tfidf = {c:cat_weights[c] / (len(info["categories"])) for c in info["categories"]}
        people[p]["tfidf"] = tfidf
        norm = sum([t * t for _,t in tfidf.items()])
        people[p]["norm"] = math.sqrt(norm)

        # Cache pivot-norm vectors
        pivot_tfidf = {c:cat_weights[c] / ((1.0 - slope) * pivot + slope * len(info["categories"])) for c in info["categories"]}
        people[p]["pivot_tfidf"] = pivot_tfidf
        pivot_norm = sum([t * t for _,t in pivot_tfidf.items()])
        people[p]["pivot_norm"] = math.sqrt(pivot_norm)

    return people, people_in_cat, vocab

# Category method for evaluating the matching methodology. Sample a category and sample
# 500 people from that category to be the target set
def simulate_by_category(sim_metric, slope, max_match_count, cat = None, random_seed = None):
    valid_cats = load_filtered_categories()
    candidate_cats = [c for c,p in valid_cats.items() if len(p) >= 500 and not c in TUNE_CATEGORIES]
    if cat is None:
        random.seed(random_seed)
        cat = random.sample(candidate_cats, 1)[0]
        print("Evaluating cat", cat)

    people, people_in_cat, vocab = prepare_people(slope, cat, remove_tune_people=True)

    # Now sample 500 people from the category for analysis (note that we seeded in line 134)
    treatment = random.sample(people_in_cat, min(500, len(people_in_cat)))

    treatment = {p:x for p,x in people.items() if p in treatment}
    candidates = {p:x for p,x in people.items() if not p in people_in_cat}

    if sim_metric_str == "propensity":
        del valid_cats[cat]
        compute_propensity_scores(treatment, candidates, valid_cats, tf_idf=False)
    elif sim_metric_str == "propensity_tfidf":
        del valid_cats[cat]
        compute_propensity_scores(treatment, candidates, valid_cats, tf_idf=True)

    if sim_metric == "random":
        matched_sample = get_people_sample(candidates, len(treatment))
        matched_pairs = None
    else:
        treatment, matched_sample, matched_pairs = make_matches(treatment, candidates, sim_metric, max_match_count)

    eval_matched = assess_match(treatment, matched_sample, vocab, set([cat]))
    print_eval(eval_matched)

    return treatment, matched_pairs, matched_sample

# Main function for evaluating the matching methodology. Sample 1000 people and compare them
# to a matched sample or another random sample
def simulate(sim_metric, slope, max_match_count, random_seed = None):
    people, _, vocab = prepare_people(slope, verbose=False, remove_tune_people=True)

    treatment = get_people_sample(people, 1000, seed=random_seed)
    candidates = {p:x for p,x in people.items() if not p in treatment}

    if sim_metric_str == "propensity":
        del valid_cats[cat]
        compute_propensity_scores(treatment, candidates, valid_cats, tf_idf=False)
    elif sim_metric_str == "propensity_tfidf":
        del valid_cats[cat]
        compute_propensity_scores(treatment, candidates, valid_cats, tf_idf=True)

    if sim_metric == 'random':
        matched_sample = get_people_sample(candidates, len(treatment))
        matched_pairs = []
    else:
        treatment, matched_sample, matched_pairs = make_matches(treatment, candidates, sim_metric, max_match_count)

    eval_matched = assess_match(treatment, matched_sample, vocab)

    print_eval(eval_matched)
    return treatment, matched_pairs, matched_sample


# Used for pivot-slope tf-idf weighting. Try different slope values
def tune_slope(sim_metric=get_pivot_cosine_sim, max_match_count):
    # we use this as the pivot
    slope = 0.5
    for slope in range(5, 10):
        print("#######################################################################################")
        slope = 0.3 + (slope / 100)
        print("Slope", slope)
        people, _, vocab = prepare_people(slope, None)

        treatment = get_people_sample(people, 1000, seed=10)
        candidates = {p:x for p,x in people.items() if not p in treatment}
        random_control = get_people_sample(candidates, len(treatment))

        treatment, matched_sample, matched_pairs = make_matches(treatment, candidates, sim_metric, max_match_count)
        for p in matched_pairs[:10]:
            print(p)
        print("Sample sizes", len(treatment), len(matched_sample))

        eval_matched = assess_match(treatment, matched_sample, vocab)
        eval_random = assess_match(treatment, random_control, vocab)

        print_eval(eval_matched)
        print_eval(eval_random)

    return treatment, matched_pairs, random_control


def get_match_method(match_method_str):
    metric = None
    if match_method_str == "number":
        metric = get_num_cats
    elif match_method_str == "percent":
        metric = get_percent_cats
    elif match_method_str == "tfidf":
        metric = get_cosine_sim
    elif match_method_str == "pivot_tfidf":
        metric = get_pivot_cosine_sim
    elif match_method_str == "random":
        metric = match_method_str
    elif match_method_str == "propensity" or match_method_str == "propensity_tfidf":
        metric = get_propensity_diff
    else:
        assert False
    return metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=['tune', 'simulate', 'simulate_category', 'tune_category'], default='simulate', help="Time of simulation or parameter tuning to run")
    parser.add_argument("--match_method", choices=['number', 'percent', 'tfidf', 'pivot_tfidf', 'random'], default='pivot_tfidf', help="Type of matching to use")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed. Can be used to ensure that the randomly-sampled target groups are the same accross different runs, so that matching algorithms can be compared using the same simulated target groups.")
    parser.add_argument("--slope", type=float, default=0.3, help="Slope parameter. Only used for pivot tf-idf matching")
    parser.add_argument("--max_match_count", type=int, default=10, help="limit the maximum number of times a comparison person can be used as a match")
    args = parser.parse_args()

    metric = get_match_method(args.match_method)

    print("Running " + args.match_method + " " + args.run + " " + str(args.slope))
    if args.run == 'simulate_category':
        _, matched_pairs, _ = simulate_by_category(metric, args.slope, args.max_match_count, random_seed = args.random_seed, sim_metric_str = args.match_method)
    elif args.run == 'simulate':
        _, matched_pairs, _ = simulate(metric, args.slope, args.max_match_count, args.random_seed, sim_metric_str = args.match_method)
    elif args.run == 'tune_category':
        for c in TUNE_CATEGORIES:
            simulate_by_category(metric, args.slope, args.max_match_count, c)
    else:
        tune_slope(metric, args.max_match_count)
