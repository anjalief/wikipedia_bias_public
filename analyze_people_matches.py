# Run the primary analyses in the paper
from make_people_matches import print_metrics
from category_analysis import get_races, get_gender
from utils import get_filtered_people_with_topics, load_categories, CACHE_PATH, MATCHED_CACHE_PATH, process_data, get_invalid_cats, get_code_to_language
import pickle
import argparse, os
import random
from datetime import datetime
from basic_log_odds import write_log_odds
from collections import Counter, defaultdict
from utils import paired_ttest, binary_mcnemar_test, get_mulitple_test, get_cohen_d, ttest
from tfidf_matching import prepare_people

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
english_tokenizer = Tokenizer(nlp.vocab)
import nltk
import pyarabic.araby as araby

def RussianTokenizer(text):
    return nltk.word_tokenize(text, language="russian")

def get_tokenizer(lang):
    if lang == 'de':
        from spacy.lang.de import German
        lang_nlp = German()

    elif lang == 'en':
        return english_tokenizer

    elif lang == 'ar':
        return araby.tokenize

    elif lang == 'ru':
        return RussianTokenizer

    elif lang == 'it':
        from spacy.lang.it import Italian
        lang_nlp = Italian()

    elif lang == 'es':
        from spacy.lang.es import Spanish
        lang_nlp = Spanish()

    # Note that I specified a dummy file in
    # Line 192 /projects/tir1/users/anjalief/anaconda3/envs/py36/lib/python3.6/site-packages/spacy/lang/ja/__init__.py
    # In order to get this to work
    elif lang == 'ja':
        from spacy.lang.ja import Japanese
        lang_nlp = Japanese()

    elif lang == 'pt':
        from spacy.lang.pt import Portuguese
        lang_nlp = Portuguese()

    elif lang == 'zh':
        from spacy.lang.zh import Chinese
        lang_nlp = Chinese()

    elif lang == 'fr':
        from spacy.lang.fr import French
        lang_nlp = French()

    elif lang == 'sw':
        return None

    else:
        print("Unknown Language", lang)
        return None
    return Tokenizer(lang_nlp.vocab)

def print_odds_from_people(treatment, control):
    treatment_counts = Counter()
    control_counts = Counter()
    prior = Counter()
    article_count = Counter()

    for p,info in treatment.items():
        treatment_counts.update(info["text"])
        prior.update(info["text"])
        article_count.update(set(info['text']))
    for p,info in control.items():
        control_counts.update(info["text"])
        prior.update(info["text"])
        article_count.update(set(info['text']))

    delta = write_log_odds(treatment_counts, control_counts, prior)

    print(len(delta))
    # NOTE for transgender men I changed this to 20. There are fewer
    # that 100 articles in that data set
    delta = {d:s for d,s in delta.items() if article_count[d] > 100}

    delta_sorted = sorted(delta, key=delta.get, reverse=True)
    print(len(delta))



    for w in delta_sorted[:25]:
        print(w, "{:0.2f}".format(delta[w]), article_count[w])
    print("################################################################################")
    for w in delta_sorted[-25:]:
        print(w, "{:0.2f}".format(delta[w]), article_count[w])


def compare_article_lengths(treatment, matched_sample):
    treatment_lengths = []
    control_lengths = []

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        treatment_lengths.append(len(treatment[treatment_name]['text']))
        control_lengths.append(len(control_info['text']))

    treatment_average, control_average, pvalue = paired_ttest(treatment_lengths, control_lengths)
    print("Article Lengths {:0.2f} {:0.2f}".format(treatment_average, control_average), pvalue)

def compare_category_counts(treatment, matched_sample):
    treatment_lengths = []
    control_lengths = []

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        treatment_lengths.append(len(treatment[treatment_name]['categories']))
        control_lengths.append(len(control_info['categories']))

    treatment_average, control_average, pvalue = paired_ttest(treatment_lengths, control_lengths)
    print("Category Counts {:0.2f} {:0.2f}".format(treatment_average, control_average), pvalue)

def get_sec_length(section_name, sections, tokenizer = english_tokenizer):
    if section_name in sections:
        tokens = tokenizer(sections[section_name])
        return len(tokens)
    return 0

# 'section-names': ['Early life, military service, and education', 'Early career', 'U.S. House of Representatives']
# 'section-text': {'second': [['summary', 'Charles Bernard Rangel is an American politician who was a U.S. Representative for districts in
# Sections seem to typically have 'second' and 'third'
def compare_sec_lengths(treatment, matched_sample, min_article_count):
    sect_counts = Counter()

    for x,info in treatment.items():
        # Just take the top level sections
        sect_counts.update([y.lower() for y in info['section-names'][0]])
    for x,info in matched_sample.items():
        sect_counts.update([y.lower() for y in info['section-names'][0]])

    sect_counts = {x:c for x,c in sect_counts.items() if c >= min_article_count}
    print("Number of sections", len(sect_counts))

    treatment_sect_to_length = defaultdict(list)
    control_sect_to_length = defaultdict(list)

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        treatment_info = treatment[treatment_name]
        control_sections = {x:y for x,y in control_info['section-text']['second']}
        treatment_sections = {x:y for x,y in treatment_info['section-text']['second']}

        for l in sect_counts:
            control_sect_to_length[l].append(get_sec_length(l, control_sections) / len(control_info['text']))
            treatment_sect_to_length[l].append(get_sec_length(l, treatment_sections) / len(treatment_info['text']))

    print("Section,treat_avg,control_avg,pval")
    print_multiple_tests(sect_counts, treatment_sect_to_length, control_sect_to_length, paired_ttest)

def compare_lang_secs(treatment, matched_sample, lang, min_article_count):
    assert(len(treatment) == len(matched_sample))
    sect_counts = Counter()
    treatment_article_lengths = []
    control_article_lengths = []

    treatment_section_numbers = []
    control_section_numbers = []

    tok = get_tokenizer(lang)
    if tok is None:
        return

    for x in treatment:
        # Just take the top level sections
        sect_counts.update([y[0].lower() for y in x['segmented']['second']])
        treatment_article_lengths.append(len(tok(x['text'])))
        section_count = 0
        for _,sections in x['segmented'].items():
            section_count += len(sections)
        treatment_section_numbers.append(section_count)
    for x in matched_sample:
        sect_counts.update([y[0].lower() for y in x['segmented']['second']])
        control_article_lengths.append(len(tok(x['text'])))

        section_count = 0
        for _,sections in x['segmented'].items():
            section_count += len(sections)
        control_section_numbers.append(section_count)


    treatment_average, control_average, pvalue = paired_ttest(treatment_article_lengths, control_article_lengths)
    print(treatment_average, control_average, pvalue, sep="|", end="|") # Article Lengths


    treatment_average, control_average, pvalue = paired_ttest(treatment_section_numbers, control_section_numbers)
    print(treatment_average, control_average, pvalue, sep="|") # Number of Sections

    sect_counts = {x:c for x,c in sect_counts.items() if c >= min_article_count}

    treatment_sect_to_length = defaultdict(list)
    control_sect_to_length = defaultdict(list)

    for treatment_info,control_info in zip(treatment, matched_sample):
        control_sections = {x:y for x,y in control_info['segmented']['second']}
        treatment_sections = {x:y for x,y in treatment_info['segmented']['second']}

        for l in sect_counts:
            control_sect_to_length[l].append(get_sec_length(l, control_sections, tok) / len(tok(control_info['text'])))
            treatment_sect_to_length[l].append(get_sec_length(l, treatment_sections, tok) / len(tok(treatment_info['text'])))

    if len(treatment_sect_to_length) > 0:
        print_multiple_tests(sect_counts, treatment_sect_to_length, control_sect_to_length, paired_ttest, only_significant=True)


def compare_named_sec(treatment, matched_sample, section_name):
    treatment_lengths = []
    control_lengths = []

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        treatment_info = treatment[treatment_name]
        control_sections = {x:y for x,y in control_info['section-text']['second']}
        treatment_sections = {x:y for x,y in treatment_info['section-text']['second']}

        control_lengths.append(get_sec_length(section_name, control_sections))
        treatment_lengths.append(get_sec_length(section_name, treatment_sections))

    treatment_average, control_average, pvalue = paired_ttest(treatment_lengths, control_lengths)
    print(section_name,treatment_average, control_average, pvalue)


def print_multiple_tests(val_to_counts, treatment_to_list, control_to_list, test_type, skip=[], only_significant=False, code_to_print = None):
    vals = []
    percents = []
    pvals = []
    for l in val_to_counts:
        if l in skip:
            continue
        try:
            treatment_avg, control_avg, pval = test_type(treatment_to_list[l], control_to_list[l])
            vals.append(l)
            percents.append((treatment_avg, control_avg))
            pvals.append(pval)
        except:
            print("Error on hypothesis test for", l)


    is_reject, corrected_pvals = get_mulitple_test(pvals)
    assert(len(is_reject) == len(vals))
    assert(len(corrected_pvals) == len(vals))

    for l,p,reject,percents in zip(vals, corrected_pvals, is_reject, percents):
        if only_significant and not reject:
            continue
        if code_to_print is None:
            print(l, percents[0], percents[1], p, reject, sep=",")
        else:
            print(l, code_to_print.get(l, l), percents[0], percents[1], p, reject, sep="|")


# 'langs': [('en', 'Charles_Rangel'), ('arz', 'تشارليس_بى._رانجيل'), ('de', 'Charles_B._Rangel')]
def compare_langs(treatment, matched_sample, min_article_count):
    lang_counts = Counter()
    for _,info in treatment.items():
        lang_counts.update([y[0] for y in info['langs']])
    for _,info in matched_sample.items():
        lang_counts.update([y[0] for y in info['langs']])

    lang_counts = {x:c for x,c in lang_counts.items() if c >= min_article_count}

    treatment_lang_to_binary = defaultdict(list)
    control_lang_to_binary = defaultdict(list)
    treatment_lang_count = []
    control_lang_count = []

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        treatment_langs = set([y[0] for y in treatment[treatment_name]['langs']])
        control_langs = set([y[0] for y in control_info['langs']])
        # treatment_lang_count.append(len(treatment[treatment_name]['text']))
        # control_lang_count.append(len(control_info['text']))
        treatment_lang_count.append(len(treatment_langs))
        control_lang_count.append(len(control_langs))

        for l in lang_counts:
            if l == 'en':
                continue
            control_pres = 1 if l in control_langs else 0
            control_lang_to_binary[l].append(control_pres)

            treatment_pres = 1 if l in treatment_langs else 0
            treatment_lang_to_binary[l].append(treatment_pres)
    print(len(treatment_lang_to_binary),
            len(control_lang_to_binary),
            len(treatment_lang_count),
            len(control_lang_count))

    treatment_average, control_average, pvalue = paired_ttest(treatment_lang_count, control_lang_count)
    print("Number of languages", treatment_average, control_average, pvalue)
    print("Language,treat_avg,control_avg,pval")
    print_multiple_tests(lang_counts, treatment_lang_to_binary, control_lang_to_binary,
        binary_mcnemar_test, skip=['en'], code_to_print=get_code_to_language())

def compare_edits(treatment, matched_sample):
    edit_cache = os.path.join(CACHE_PATH, 'people_edits.pkl')
    people_to_edits = pickle.load(open(edit_cache, 'rb'))

    treatment_counts = []
    control_counts = []

    treatment_age = []
    control_age = []

    missing_people = []

    def update_edits(counts, ages, info):
        edits = info['history']
        count = sum([x[1] for x in edits])
        published_date  = datetime.strptime(info['published'], '%y-%m')

        delta = collected_date - published_date
        age = int(delta.days / 30)

        counts.append(count)
        ages.append(age)

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        control_name = names.split('::')[0]
        skip = False

        collected_date = datetime.strptime('20-09', '%y-%m')

        if not treatment_name in people_to_edits or people_to_edits[treatment_name]['published'] is None:
            missing_people.append(treatment_name)
            skip = True

        if not control_name in people_to_edits or people_to_edits[control_name]['published'] is None:
            missing_people.append(control_name)
            skip = True

        if skip:
            continue

        update_edits(treatment_counts, treatment_age, people_to_edits[treatment_name])
        update_edits(control_counts, control_age, people_to_edits[control_name])



    print("Missing", len(missing_people))
    assert(len(treatment_counts) == len(control_counts))
    assert(len(treatment_age) == len(control_age))

    treatment_count, control_count, pvalue = paired_ttest(treatment_counts, control_counts)
    print("Number of edits", treatment_count, control_count, pvalue)

    treatment_age, control_age, pvalue = paired_ttest(treatment_age, control_age)
    print("Article ages", treatment_age, control_age, pvalue)



def print_language_counts(treatment, matched_sample):
    language_cache = os.path.join(CACHE_PATH, 'people_articles.pkl')
    code_to_language = get_code_to_language()
    people_multilingual = pickle.load(open(language_cache, 'rb'))
    lang_counts = Counter()
    lang_to_control_list = defaultdict(list)
    lang_to_treatment_list = defaultdict(list)
    lang_to_control_list_english = defaultdict(list)
    lang_to_treatment_list_english = defaultdict(list)

    skipped = 0
    for names,_ in matched_sample.items():
        treatment_name = names.split('::')[1]
        control_name = names.split('::')[0]
        if treatment_name in people_multilingual and control_name in people_multilingual:
            treatment_langs = set(people_multilingual[treatment_name].keys())
            control_langs = set(people_multilingual[control_name].keys())
            both = treatment_langs.intersection(control_langs)

            lang_counts.update(list(both))

            for l in both:
                lang_to_treatment_list[l].append(people_multilingual[treatment_name][l])
                lang_to_control_list[l].append(people_multilingual[control_name][l])
                lang_to_treatment_list_english[l].append(people_multilingual[treatment_name]['en'])
                lang_to_control_list_english[l].append(people_multilingual[control_name]['en'])
        else:
            skipped += 1

    print("Missing language info for %s pairs" % skipped)

    print_sep("Comparing article lengths in different languages")
    print("If there is a significant difference in length of sections for any language, they are printed below the language row (treatment length, control length, p-value")
    print("Number of Pairs,Treatement Article Length,Control Article Length,Length p-value,Treatment Number of Sections,Control Number of sections,Section p-value")
    for x in lang_counts:
        if x != 'en':
            print(x, code_to_language[x], lang_counts[x], sep="|", end="|")
            compare_lang_secs(lang_to_treatment_list[x], lang_to_control_list[x], x, 100)
    print()
    print_sep("Comparing article lengths in English for articles available in multiple languages")
    print("If there is a significant difference in length of sections for any language, they are printed below the language row (treatment length, control length, p-value")
    for x in lang_counts:
        if x != 'en':
            print(x, code_to_language[x], lang_counts[x], sep="|", end="|")
            compare_lang_secs(lang_to_treatment_list_english[x], lang_to_control_list_english[x], 'en', 100)
    

def print_unmatched_quick_counts(all_treatment, all_control):
    def get_counts(people_dict):
        lengths = []
        cat_nums = []
        lang_nums = []

        for _,info in people_dict.items():
            lengths.append(len(info['text']))
            cat_nums.append(len(info['categories']))
            lang_nums.append(len(info['langs']))
        return lengths, cat_nums, lang_nums

    treatment_lengths, treatment_cats, treatment_langs = get_counts(all_treatment)
    control_lengths, control_cats, control_langs = get_counts(all_control)
    print(",Treatment Average, Control Average,p-value")
    print("Article Lengths,", ",".join([str(i) for i in ttest(treatment_lengths, control_lengths)]))
    print("Category Counts,", ",".join([str(i) for i in ttest(treatment_cats, control_cats)]))
    print("Lang Counts,", ",".join([str(i) for i in ttest(treatment_langs, control_langs)]))


def compare_unmatched_sec_lengths(treatment, control, min_article_count):
    sect_counts = Counter()

    for x,info in treatment.items():
        # Just take the top level sections
        sect_counts.update([y.lower() for y in info['section-names'][0]])
    for x,info in control.items():
        sect_counts.update([y.lower() for y in info['section-names'][0]])

    sect_counts = {x:c for x,c in sect_counts.items() if c >= min_article_count}
    print("Number of sections", len(sect_counts))

    def get_sec_counts(people_dict):
        sect_to_length = defaultdict(list)
        for names,info in people_dict.items():
            sections = {x:y for x,y in info['section-text']['second']}

            for l in sect_counts:
                sect_to_length[l].append(get_sec_length(l, sections) / len(info['text']))
        return sect_to_length


    treatment_sect_to_length = get_sec_counts(treatment)
    control_sect_to_length = get_sec_counts(control)
    print("Section,treat_avg,control_avg,pval")
    print_multiple_tests(sect_counts, treatment_sect_to_length, control_sect_to_length, ttest)

def print_sample_pairs(matched_pairs, treatment, matched_sample):
    len_common_cats = sum([len(s[3]) for s in matched_pairs])/len(matched_pairs)
    print("Avg number of common cats", len_common_cats)
    sample = random.sample(matched_pairs, 20)

    print("Name,Text Length, Language Count, Languages, Categories in common,Match Score")

    for s in sample:
        treatment_name, control_name, score, cats = s

        treatment_info = treatment[treatment_name]
        control_info = matched_sample[control_name + "::" + treatment_name]

        def print_stats(name, info, cats, score):
            stats = [name, len(info['text']), len(info['langs']), " ".join([l[0] for l in info['langs']]), " ".join(cats), score]
            stats = [str(i) for i in stats]
            print(','.join(stats))
        print_stats(treatment_name, treatment_info, cats, score)
        print_stats(control_name, control_info, "", "")
        print()


def print_sep(print_str):
    print("############################### %s ##################################################" % print_str)

def run_analysis(tupl, treatment_names, control_names, invalid_categories, people, vocab, min_article_count, is_propensity, print_unmatched = False):
    treatment, matched_sample, matched_pairs = process_data(tupl, is_propensity)

    if print_unmatched:
        print("################################################################################")
        print("Printing unmatched metrics")
        print("################################################################################")

        all_treatment = {x:people[x] for x in treatment_names if x in people}
        all_control = {x:people[x] for x in control_names if x in people}
        print("Data sizes", len(all_treatment), len(all_control))

        print_sep("Printing Match Evaluation metrics")
        print_metrics(all_treatment, all_control, None, vocab, "Empty", invalid_categories, print_with_marker_cats=False)
        
        print_sep("Printing Top and Bottom Log-odds words")
        print_odds_from_people(all_treatment, all_control)
        
        print_sep("Print basic statistics")
        print_unmatched_quick_counts(all_treatment, all_control)

        # compare_unmatched_sec_lengths(all_treatment, all_control, min_article_count)
    else:
        # print_sep("Printing Match Evaluation metrics")
        # print_metrics(treatment, matched_sample, None, vocab, "Empty", invalid_categories, print_with_marker_cats=False)
        
        # print_sep("Printing Sample Pairs")
        # print_sample_pairs(matched_pairs, treatment, matched_sample)

        # print_sep("Printing Top and Bottom Log-odds words")
        # print_odds_from_people(treatment, matched_sample)

        # print_sep("Comparing article lengths")
        # compare_article_lengths(treatment, matched_sample)

        # print_sep("Comparing number of categories each article has")
        # compare_category_counts(treatment, matched_sample)

        # print_sep("Comparing which languages article is available in")
        # compare_langs(treatment, matched_sample, min_article_count)

        # print_sep("Comparing number of edits each article has")
        # compare_edits(treatment, matched_sample)

        # print_sep("Comparing length of common sections")
        # compare_sec_lengths(treatment, matched_sample, min_article_count)

        print_sep("Comparing length of personal life sections")
        compare_named_sec(treatment, matched_sample, "personal life")

        # print_sep("Comparing length of career sections")
        # compare_named_sec(treatment, matched_sample, "career")

        # print_language_counts(treatment, matched_sample)

def get_lang_counts(treatment, matched_sample):
    treatment_lang_to_binary = defaultdict(list)
    control_lang_to_binary = defaultdict(list)
    all_langs = set()

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        treatment_langs = set([y[0] for y in treatment[treatment_name]['langs']])
        control_langs = set([y[0] for y in control_info['langs']])
        all_langs.update(treatment_langs)
        all_langs.update(control_langs)

    for names,control_info in matched_sample.items():
        treatment_name = names.split('::')[1]
        treatment_langs = set([y[0] for y in treatment[treatment_name]['langs']])
        control_langs = set([y[0] for y in control_info['langs']])

        for l in all_langs:
            control_pres = 1 if l in control_langs else 0
            control_lang_to_binary[l].append(control_pres)

            treatment_pres = 1 if l in treatment_langs else 0
            treatment_lang_to_binary[l].append(treatment_pres)

    assert(len(treatment_lang_to_binary) == len(control_lang_to_binary))
    lang_diffs = {l:get_cohen_d(treatment_lang_to_binary[l], control_lang_to_binary[l]) for l in all_langs}
    return lang_diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--people_type", choices=['african', 'asian', 'latino',
        'women', 'nonbinary', 'transgender_women', 'transgender_men',
        'black_women'],
        default='african')
    parser.add_argument("--match_method", choices=['number', 'percent', 'tfidf', 'pivot_tfidf', 'random', 'propensity', "propensity_tfidf"], default='pivot_tfidf')
    parser.add_argument("--print_unmatched", action='store_true')
    args = parser.parse_args()
    print("Running", args.people_type)

    people, vocab = get_filtered_people_with_topics()
    cats = load_categories()
    is_propensity = "propensity" in args.match_method

    if args.people_type in ['african', 'asian', 'latino']:
        cache_name = os.path.join(MATCHED_CACHE_PATH, args.match_method + '_matched_race.pkl')
        invalid_categories = get_invalid_cats(cache_name = "invalid_race_cats.txt")

        african_american, asian, latino = pickle.load(open(cache_name,"rb"))
        name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, _ = get_races(cats, people)

        if args.people_type == 'african':
            run_analysis(african_american, name_african_american, name_white, invalid_categories, people, vocab, 100, is_propensity, args.print_unmatched)
        if args.people_type == 'asian':
            run_analysis(asian, name_asians, name_white, invalid_categories, people, vocab, 100, is_propensity, args.print_unmatched)
        if args.people_type == 'latino':
            run_analysis(latino, name_latino, name_white, invalid_categories, people, vocab, 100, is_propensity, args.print_unmatched)
    elif args.people_type == 'black_women':
        race_invalid_categories = get_invalid_cats(cache_name = "invalid_race_cats.txt")
        gender_invalid_categories = get_invalid_cats(cache_name = "invalid_gender_cats.txt")

        name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, _ = get_races(cats, people)
        name_nb, name_men, name_women, name_transgender_men, name_transgender_women, name_cisgender_men, category_nb, category_LGBT = get_gender(cats, people)

        control_women = [n for n in name_white if n in name_women]
        control_men = [n for n in name_white if n in name_men]
        black_men = [n for n in name_african_american if n in name_men]
        black_women = [n for n in name_african_american if n in name_women]


        cache_name = os.path.join(MATCHED_CACHE_PATH, args.match_method + '_intersectional.pkl')
        vs_control_women, vs_black_men, vs_control_men = pickle.load(open(cache_name,"rb"))

        print("############################ Black women vs. Control women ######################################")
        run_analysis(vs_control_women, black_women, control_women, race_invalid_categories, people, vocab, 50, is_propensity, args.print_unmatched)


        print("############################ Black women vs. Black men ########################################")
        run_analysis(vs_black_men, black_women, black_men, gender_invalid_categories, people, vocab, 50, is_propensity, args.print_unmatched)

        print("############################ Black women vs. Control men #####################################")
        run_analysis(vs_control_men, black_women, control_men, race_invalid_categories.union(gender_invalid_categories), people, vocab, 50, is_propensity, args.print_unmatched)

    else:
        cache_name = os.path.join(MATCHED_CACHE_PATH, args.match_method + '_matched_gender.pkl')
        invalid_categories = get_invalid_cats(cache_name = "invalid_gender_cats.txt")
        women, nb, transgender_men, transgender_women = pickle.load(open(cache_name,"rb"))
        name_nb, name_men, name_women, name_transgender_men, name_transgender_women, name_cisgender_men, category_nb, category_LGBT = get_gender(cats, people)

        if args.people_type == 'women':
            # We didn't define women group with categories, there are no invalid cats
            run_analysis(women, name_women, name_men, invalid_categories, people, vocab, 500, is_propensity, args.print_unmatched)
        if args.people_type == 'transgender_women':
            # We didn't define group with categories, there are no invalid cats
            run_analysis(transgender_women, name_transgender_women, name_men, invalid_categories, people, vocab, 20, is_propensity, args.print_unmatched)
        if args.people_type == 'transgender_men':
            # We didn't define group with categories, there are no invalid cats
            run_analysis(transgender_men, name_transgender_men, name_men, invalid_categories, people, vocab, 20, is_propensity, args.print_unmatched)
        if args.people_type == 'nonbinary':
            run_analysis(nb, name_nb, name_men, category_nb, people, vocab, 50, is_propensity, args.print_unmatched)




if __name__ == "__main__":
    main()
