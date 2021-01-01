This repository contains code for "Controlled Analyses of Social Biases in Wikipedia Bios"



### Matching method simulations
`tfidf_matching.py` is the primary file for constructing matches and testing matching methods with simulations
Example usage: `python tfidf_matching.py --run simulate --match_method pivot_tfidf --slope 0.3 --random_seed 5`

`utils.py` has the primary data loading and cache generating functions. *You must set `CACHE_PATH` in this file to the location of cached files*


### Analysis of race and gender
`category_analysis.py` generates the set of people in each race and gender category. You can run this file (requires downloading `people_nb.pkl` to get a pre-cached list of people with non-binary indicators and `people_to_wikidata.pickle` to get Wikidata info), or you can download the cached outputs: `gender_names_cisgender.pickle`, `race_splits.pkl`

`make_people_matches.py` creates comparison groups for all the target groups using pivot-slope tfidf matching. You can run this file to generate the matched groups, or you can download the cached outputs: `matched_race.pkl`, `intersectional.pkl`, `matched_gender_cisgender_corrected.pkl`

`analyze_people_matches.py` runs all analyses. You must first run `make_people_matches.py` or download the cached_outputs to generate target and comparison groups
Example usage: `python analyze_people_matches.py --people_type asian`

### Cache Files
These files are not generated by this code. If needed, they must be downloaded
* Base data, needed to run anything: `tokenized_people.pickle`, `category_people.pkl`
* Needed for people analyses: `people_edits.pkl`, `people_langs.pkl`
* Needed to re-generate the caches created by `category_analysis.py` (or you can start from pre-cached files `gender_names_cisgender.pickle`, `race_splits.pkl`): `people_nb.pkl`, `people_to_wikidata.pickle`

Optional cache files. You can download these or you can run the provided to code to re-generated them:
* `filtered_people_topics.pickle`
* `matched_race.pkl`, `intersectional.pkl`, `matched_gender_cisgender_corrected.pkl`
* `gender_names_cisgender.pickle`, `race_splits.pkl`
