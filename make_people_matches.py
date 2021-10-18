from category_analysis import get_gender, get_races
import argparse, os, random, pickle
from utils import load_categories, CACHE_PATH
from tfidf_matching import prepare_people, make_matches, get_pivot_cosine_sim
from matching_tools import assess_match, print_eval

def do_match(treatment_names, candidate_names, people, vocab, title, invalid_cats):
    treatment = {x:people[x] for x in treatment_names if x in people}
    candidates = {x:people[x] for x in candidate_names if x in people}
    treatment, matched_sample, matched_pairs = make_matches(treatment, candidates, get_pivot_cosine_sim)

    print_metrics(treatment, matched_sample, matched_pairs, vocab, title, invalid_cats)
    return treatment, matched_sample, matched_pairs

def print_metrics(treatment, matched_sample, matched_pairs, vocab, title, invalid_cats, print_with_marker_cats=True):
    print("Matching results for", title)
    print("Treatment and matched sizes", len(treatment), len(matched_sample))

    if print_with_marker_cats:
        print("Evaluation with marker cats")
        eval_matched = assess_match(treatment, matched_sample, vocab)
        print_eval(eval_matched)

    # Pull out the marked categories before assessing the match
    measure_treatment = {}
    for p,info in treatment.items():
        measure_treatment[p] = info
        measure_treatment[p]["categories"] = set([c for c in info["categories"] if not c in invalid_cats])

    measure_matched = {}
    for p,info in matched_sample.items():
        measure_matched[p] = info
        measure_matched[p]["categories"] = set([c for c in info["categories"] if not c in invalid_cats])

    print("Evaluation without marker cats")
    eval_matched = assess_match(measure_treatment, measure_matched, vocab)
    if print_with_marker_cats:
        print_eval(eval_matched)
    else:
        for x in eval_matched:
            if type(x) == int:
                print(x)
            else:
                if x < 1:
                    print("{:.4f}".format(x))
                else:
                    print("{:.2f}".format(x))

    if matched_pairs is not None:
        for x in random.sample(matched_pairs, 10):
            print(x)


def get_gender_matches(cats, people, vocab):
    cache_name = os.path.join(CACHE_PATH, 'matched_gender_cisgender_corrected.pkl')

    name_nb, name_men, name_women, name_transgender_men, name_transgender_women, name_cisgender_men, category_nb, category_LGBT = get_gender(cats, people)

    # Setting the empty set as invalid categores, e.g. no categories are invalid
    women = do_match(name_women, name_men, people, vocab, "Women", set())
    nb = do_match(name_nb, name_cisgender_men, people, vocab, "Non-binary", category_nb)
    transgender_women = do_match(name_transgender_women, name_cisgender_men, people, vocab, "Transgender Women", category_LGBT)
    transgender_men = do_match(name_transgender_men, name_cisgender_men, people, vocab, "Transgender Men", category_LGBT)

    pickle.dump((women, nb, transgender_men, transgender_women), open(cache_name,"wb"))


def get_race_matches(cats, people, vocab):
    cache_name = os.path.join(CACHE_PATH, 'matched_race.pkl')

    name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, _ = get_races(cats, people)

    african_american = do_match(name_african_american, name_white, people, vocab, "African American", category_african_american)
    asian = do_match(name_asians, name_white, people, vocab, "Asian", category_asian)
    latino = do_match(name_latino, name_white, people, vocab, "Hispanic/Latino", category_latino)

    pickle.dump((african_american, asian, latino), open(cache_name,"wb"))

def get_black_women_matches(cats, people, vocab):
    cache_name = os.path.join(CACHE_PATH, 'intersectional.pkl')

    name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, _ = get_races(cats, people)
    name_nb, name_men, name_women, name_transgender_men, name_transgender_women, category_nb = get_gender(cats, people)

    control_women = [n for n in name_white if n in name_women]
    control_men = [n for n in name_white if n in name_men]
    black_men = [n for n in name_african_american if n in name_men]
    black_women = [n for n in name_african_american if n in name_women]

    print("control women", len(control_women))
    print("control men", len(control_men))
    print("Black women", len(black_women))
    print("Black men", len(black_men))

    vs_control_women = do_match(black_women, control_women, people, vocab, "vs. Control women", category_african_american)
    vs_black_men = do_match(black_women, black_men, people, vocab, "vs. Black men", category_african_american)
    vs_control_men = do_match(black_women, control_men, people, vocab, "vs. Control men", category_african_american)

    pickle.dump((vs_control_women, vs_black_men, vs_control_men), open(cache_name,"wb"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--people_type", choices=['gender', 'race', 'black_women'])
    parser.add_argument("--cache_path")
    args = parser.parse_args()

    people, _, vocab = prepare_people(slope = 0.3, verbose=True)
    cats = load_categories()
    print("Total people to analyze", len(people))


    if args.people_type == 'race':
        get_race_matches(cats, people, vocab)
    if args.people_type == 'black_women':
        get_black_women_matches(cats, people, vocab)
    else:
        get_fixed_gender_matches(cats, people, vocab)

if __name__ == "__main__":
    main()
