from matching_tools import get_standardized_bias
import argparse
from utils import get_filtered_people_with_topics, load_categories, process_data, get_invalid_cats, load_filtered_categories
import os
import pickle
from category_analysis import get_races, get_gender
from utils import MATCHED_CACHE_PATH

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
Z_SCORE = 3.29

# MATCH_METHODS = ['number', 'percent', 'tfidf', 'pivot_tfidf', 'random', 'propensity', "propensity_tfidf", "Unmatched"]
MATCH_METHODS_TO_STR = {'Unmatched':'Unmatched',
                        'number': "Number",
                        'percent': "Percent",
                        'tfidf':"TF-IDF",
                        "propensity":"Propensity",
                        "propensity_tfidf": "Prop. TF-IDF",
                        'pivot_tfidf': "Pivot TF-IDF"}

# method_to_color = {
#     "Random": "#117733",
#     "Number": "#44AA99",
#     "Percent": "#88CCEE",
#     "TF-IDF": "#CC6677",
#     "Pivot TF-IDF": "#882255"
# }

# Palette from http://mkweb.bcgsc.ca/colorblind/palettes/8.color.blindness.palette.txt
# color  1-main 000000   0   0   0 black, grey, grey, grey, rich black, grey, cod grey, grey, almost black, grey
# color  2-main 2271B2  34 113 178 honolulu blue, bluish, strong cornflower blue, spanish blue, medium persian blue, sapphire blue, ocean boat blue, french blue, windows blue, tufts blue
# color  2-alt  AA0DB4 170  13 180 barney, strong magenta, heliotrope magenta, strong heliotrope, steel pink, barney purple, purple, violet, violet eggplant, deep magenta
# color  3-main 3DB7E9  61 183 233 summer sky, cyan, picton blue, vivid cerulean, deep sky blue, brilliant cornflower blue, malibu, bright cerulean, cerulean, cerulean
# color  3-alt  FF54ED 255  84 237 light magenta, violet pink, light brilliant magenta, pink flamingo, light brilliant orchid, brilliant magenta, purple pizzazz, candy pink, blush pink, shocking pink
# color  4-main F748A5 247  72 165 barbie pink, rose bonbon, wild strawberry, brilliant rose, brilliant rose, magenta, wild strawberry, light brilliant rose, frostbite, brilliant cerise
# color  4-alt  00B19F   0 177 159 strong opal, tealish, persian green, keppel, topaz, manganese blue, light sea green, sea green light, puerto rico, turquoise
# color  5-main 359B73  53 155 115 ocean green, sea green, viridian, mother earth, moderate spring green, moderate aquamarine, paolo veronese green, observatory, jungle green, ocean green
# color  5-alt  EB057A 235   5 122 vivid rose, red purple, mexican pink, bright pink, rose, strong pink, luminous vivid rose, deep pink, winter sky, hot pink
# color  6-main d55e00 213  94   0 bamboo, smoke tree, red stage, tawny, tenn, tenne, burnt orange, rusty orange, dark orange, mars yellow
# color  6-alt  F8071D 248   7  29 vivid red, luminous vivid amaranth, ruddy, ku crimson, vivid amaranth, light brilliant red, cherry red, red, red, bright red
# color  7-main e69f00 230 159   0 gamboge, squash, buttercup, marigold, dark goldenrod, medium goldenrod, fuel yellow, sun, harvest gold, orange
# color  7-alt  FF8D1A 255 141  26 dark orange, juicy, west side, tangerine, gold drop, pizazz, princeton orange, university of tennessee orange, tangerine, tahiti gold
# color  8-main f0e442 240 228  66 holiday, buzz, paris daisy, starship, golden fizz, dandelion, gorse, lemon yellow, bright lights, sunflower
# color  8-alt  9EFF37 158 255  55 french lime, lime, green yellow, green lizard, luminous vivid spring bud, spring frost, vivid spring bud, bright yellow green, spring bud, acid green

method_to_color = {
    "Unmatched": "#000000",
    "number": "#2271B2",
    "percent": "#AA0DB4",
    "tfidf": "#3DB7E9",
    "pivot_tfidf": "#FF54ED",
    "propensity": "#d55e00",
    "propensity_tfidf": "#00B19F"
}


def get_method_to_scores(invalid_cats, cache_name_str, people_idx, treatment_names, control_names, people):
    valid_cats = load_filtered_categories()
    valid_cats = set([v for v in valid_cats if not v in invalid_cats])

    # Read matches from cache for each method
    method_to_scores = {}
    method_to_count = {}
    for m in MATCH_METHODS_TO_STR:
        cache_name = os.path.join(MATCHED_CACHE_PATH, m + cache_name_str)
        if not os.path.exists(cache_name):
            print("Skipping", cache_name)
            continue

        match_parts = pickle.load(open(cache_name,"rb"))
        tupl = match_parts[people_idx]

        is_propensity = "propensity" in m
        treatment, matched_sample, matched_pairs = process_data(tupl, is_propensity, drop_matches=True)
        bias_measures = get_standardized_bias(treatment, matched_sample, valid_cats, return_all=True)
        method_to_scores[m] = bias_measures
        method_to_count[m] = len(matched_pairs)

    # Get stats if we don't match
    all_treatment = {x:people[x] for x in treatment_names if x in people}
    all_control = {x:people[x] for x in control_names if x in people}
    bias_measures = get_standardized_bias(all_treatment, all_control, valid_cats, return_all=True)
    method_to_scores["Unmatched"] = bias_measures
    method_to_count["Unmatched"] = len(all_treatment)

    return method_to_scores, method_to_count

def make_plot(method_to_scores, method_to_count, fig_name):
    f = plt.figure(figsize=(6,3))
    fig = f.add_subplot()
    # fig = plt.subplot()

    # width of the bars
    barWidth = 0.15
    # The starting x position of bars
    r1 = np.arange(1)

    # Plot them one at a time so we can make them different colors
    for m,m_label in MATCH_METHODS_TO_STR.items():
        if not m in method_to_scores:
            continue
        scores = method_to_scores[m]

        bars = []
        confidence_intervals = []

        stats_mean = np.mean(scores)
        se = np.std(scores) / np.sqrt(len(scores))
        confidence_interval = Z_SCORE * se
        bars.append(stats_mean)
        confidence_intervals.append(confidence_interval)

        legend_label = m_label + " (" + str(method_to_count[m]) + ")"
        # Create bars
        fig.bar(r1, bars, width = barWidth, color = method_to_color[m], edgecolor = 'black', yerr=confidence_intervals, capsize=7, label=legend_label)

        # Set position for next series
        r1 = [x + barWidth for x in r1]

    plt.legend()
    plt.savefig(fig_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--people_type", choices=['african', 'asian', 'latino',
        'women', 'nonbinary', 'transgender_women', 'transgender_men',
        'black_women_vs_control_women', 'black_women_vs_black_men', 'black_women_vs_control_men'],
        default='african')
    args = parser.parse_args()

    people, vocab = get_filtered_people_with_topics()
    cats = load_categories()

    people_to_idx =  {'african' : 0,
        'asian' : 1,
        'latino' : 2,
        'women' : 0,
        'nonbinary': 1,
        'transgender_men': 2,
        'transgender_women': 3,
        'black_women_vs_control_women': 0,
        'black_women_vs_black_men': 1,
        'black_women_vs_control_men': 2}

    if args.people_type in ['african', 'asian', 'latino']:
        invalid_cats = get_invalid_cats(cache_name = "invalid_race_cats.txt")
        name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, _ = get_races(cats, people)

        if args.people_type == 'african':
            method_to_score, method_to_count = get_method_to_scores(invalid_cats, '_matched_race.pkl', people_to_idx[args.people_type], name_african_american, name_white, people)
        if args.people_type == 'asian':
            method_to_score, method_to_count = get_method_to_scores(invalid_cats, '_matched_race.pkl', people_to_idx[args.people_type], name_asians, name_white, people)
        if args.people_type == 'latino':
            method_to_score, method_to_count = get_method_to_scores(invalid_cats, '_matched_race.pkl', people_to_idx[args.people_type], name_latino, name_white, people)

    elif args.people_type in ['black_women_vs_control_women', 'black_women_vs_black_men', 'black_women_vs_control_men']:
        race_invalid_categories = get_invalid_cats(cache_name = "invalid_race_cats.txt")
        gender_invalid_categories = get_invalid_cats(cache_name = "invalid_gender_cats.txt")

        name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, _ = get_races(cats, people)
        name_nb, name_men, name_women, name_transgender_men, name_transgender_women, name_cisgender_men, category_nb, category_LGBT = get_gender(cats, people)

        control_women = [n for n in name_white if n in name_women]
        control_men = [n for n in name_white if n in name_men]
        black_men = [n for n in name_african_american if n in name_men]
        black_women = [n for n in name_african_american if n in name_women]

        if args.people_type == 'black_women_vs_control_women':
            method_to_score, method_to_count = get_method_to_scores(race_invalid_categories, '_intersectional.pkl', people_to_idx[args.people_type], black_women, control_women, people)

        if args.people_type == 'black_women_vs_black_men':
            method_to_score, method_to_count  = get_method_to_scores(gender_invalid_categories, '_intersectional.pkl', people_to_idx[args.people_type], black_women, black_men, people)
        if args.people_type == 'black_women_vs_control_men':
            method_to_score, method_to_count  = get_method_to_scores(race_invalid_categories.union(gender_invalid_categories), '_intersectional.pkl',
                people_to_idx[args.people_type], black_women, control_men, people)
    else:
        invalid_cats = get_invalid_cats(cache_name = "invalid_gender_cats.txt")
        name_nb, name_men, name_women, name_transgender_men, name_transgender_women, name_cisgender_men, category_nb, category_LGBT = get_gender(cats, people)

        if args.people_type == 'women':
            method_to_score, method_to_count  = get_method_to_scores(invalid_cats, '_matched_gender.pkl', people_to_idx[args.people_type], name_women, name_men, people)
        if args.people_type == 'transgender_women':
            method_to_score, method_to_count  = get_method_to_scores(invalid_cats, '_matched_gender.pkl', people_to_idx[args.people_type], name_transgender_women, name_cisgender_men, people)
        if args.people_type == 'transgender_men':
            method_to_score, method_to_count  = get_method_to_scores(invalid_cats, '_matched_gender.pkl', people_to_idx[args.people_type], name_transgender_men, name_cisgender_men, people)
        if args.people_type == 'nonbinary':
            method_to_score, method_to_count  = get_method_to_scores(invalid_cats, '_matched_gender.pkl', people_to_idx[args.people_type], name_nb, name_cisgender_men, people)

    make_plot(method_to_score, method_to_count, args.people_type + "match_eval.pdf")

if __name__ == "__main__":
    main()
