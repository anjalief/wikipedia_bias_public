import os
import re
import pickle
from utils import load_filtered_categories, load_people, get_people_sample, get_filtered_people, get_filtered_people_with_topics, load_categories, CACHE_PATH
from wikidata_dump import get_living_people_wiki_data, get_item_gender, get_properties, P_ETHNIC_GROUP

def get_people(people_info, set_category, explicit_ignore = None):
    target_people = []
    for name, info in people_info.items():
        cats = set(info["categories"])
        include = len(set_category.intersection(cats)) > 0
        if explicit_ignore is not None:
            for c in cats:
                if any([x in c for x in explicit_ignore]):
                    include = False
        if include:
            target_people.append(name)
    return set(target_people)

def get_category(categories, must_include=[], key_exclude=[], key_include=[]):
    target_cats = []
    for c in categories:
        if must_include and not all([k in c for k in must_include]):
            continue
        if key_include and not any([k in c for k in key_include]):
            continue
        if key_exclude and any([k in c for k in key_exclude]):
            continue
        target_cats.append(c)
    return set(target_cats)

def get_races(cat, people):
    cache_name = os.path.join(CACHE_PATH, 'race_splits.pkl')
    if os.path.exists(cache_name):
        return pickle.load(open(cache_name, 'rb'))
    common_ignore = ["expatriate", "Games"]
    explicit_exclude = ["football_players", "basketball_players"]
    # Asians
    asians = ["Asian","Chinese","Hong_Kong","Cantonese","Fuzhou","Hakka","Hunanese","Hui","Jiangsu","Indian","Hmong","Punjabi","Bengali","Goan","Gujarati","Kannada","Korean","Taiwanese","Atayal","Japanese","Pakistani","Sri_Lankan","Nepalese","Indonesian","Filipino","Ilocano","Kapampangan","Vietnamese","Hoa","Iranian","Thai","Uzbekistani","Malaysian","Nepalese","Kazakhstani","Cambodian","Tajikistani","Singaporean","Mongolian","Bruneian","Hoklo","Laotian","Bangladeshi","Buginese","Bugis","Burmese","Buryat","Kalmyk"]
    asians += ['Kashmiri','Macanese','Malay','Malayali','Manchu','Marathi','Minangkabau','Ningbo','Ningbonese','Okinawan','Parsi','Pashtun','Shanghainese','Sichuanese','Sindhi','Taishan','Tamil','Telugu','Teochew','Tibetan','Uyghurs','Uzbek','Wenzhounese','Wu']
    west_asians = ["Turkish","Iraqi","Saudi_Arabian","Yemeni","Syrian","Jordanian","Azerbaijani","Kuwaiti","Georgian_(country)","Armenian","Qatari","Cypriot","Lebanese","Israeli","Palestinian","Afghan","Baloch","Hazaras","Kurdish"]

    category_asian = get_category(cat, ["American"], common_ignore + ["Muay","War", "African","Latino","Comic","American_descent"], asians)
    category_west_asian = get_category(cat, ["American"], common_ignore + ["War", "African","Latino","Comic","American_descent"], west_asians)
    name_asians = get_people(people, category_asian, explicit_exclude)
    print("Number of Asian people", len(name_asians))

    africans = ["African","Nigerian","Ethiopian","Egyptian","Congo","Tanzanian","South_African","Afrikaner","Kenyan","Ugandan","Algerian","Sudanese","Moroccan","Angolan","Mozambican","Ghanaian","Malagasy","Cameroonian","Ivorian","Burkinabé","Malian","Malawian","Zambian","Senegalese","Chadian","Somali","Somalian","Zimbabwean","Guinean","Rwandan","Beninese","Burundian","Tunisian","South_Sudanese","Togolese","Sierra_Leonean","Libyan","Liberian","Central_African_Republic","Mauritanian","Eritrean","Namibian","Gambian","Botswana_people","Gabonese","Lesotho","Bissau-Guinean","Mauritian","Eswatini","Djiboutian","Comorian","Cape_Verdean","São_Tomé_and_Príncipe","Seychellois","Akan","Arab","Ashanti","Benga","Bubi","Equatoguinean","Fulbe","Hausa","Igbo"]
    africans += ["Kikuyu", "Kota", "Kpelle", "Kru", "Luo", "Mandinka", "Mende", "Nigerien", "Songhai", "Tikar", "Tuareg", "Yoruba"]
    category_african_american = get_category(cat, ["African-American"], common_ignore + ["Activist","African-American_descent"])
    category_african_american.update(get_category(cat, ["American"], common_ignore + ["Asian","Latino","Comic","American_descent"], africans))
    name_african_american = get_people(people, category_african_american, explicit_exclude)
    print("Number of AA people", len(name_african_american))

    # Latino
    latinos = ["Latino","Brazilian","Mexican","Colombian","Argentine","Peruvian","Venezuelan","Chilean","Guatemalan","Ecuadorian","Bolivian","Cuban","Haitian","Dominican_Republic","Honduran","Paraguayan","Nicaraguan","Salvadoran","Costa_Rican","Panamanian","Uruguayan","Jamaican","Puerto_Rican","Trinidad_and_Tobago","Guyanese","Surinamese","Guadeloupean","Belizean","Bahamian","Martiniquais","French_Guianan","Barbadian","Saint_Lucian","Curaçaoan","Curaçao","Grenadian","Saint_Vincent_and_the_Grenadines","Aruban","Virgin_Islands","Antigua_and_Barbuda","Dominican_Republic","Dominica","Caymanian","Saint_Kitts_and_Nevis","Sint_Maarten","Turks_and_Caicos_Islands","British_Virgin_Islands","Caribbean","Anguillan","Montserratian","Falkland_Islands","Antillean","Bermudian","Carriacouan"]
    latinos += ["Purépecha", "Quechua", "Taíno"]

    category_latino = get_category(cat, ["Latino_American"], common_ignore + ["Activist"])
    category_latino.update(get_category(cat, ["American"], common_ignore + ["African","Asian","Comic","American_descent"], latinos))
    name_latino = get_people(people, category_latino, explicit_exclude)
    print("Number of Latino", len(name_latino))

    middle_eastern = ["Middle_Eastern","Bahraini","Jordanian","Saudi_Arabian","Cypriot","Kuwaiti","Syrian","Egyptian","Lebanese","Turkish","Iranian","Omani","Emirati","Iraqi","Qatari","Yemeni","Israeli","Assyrian","Coptic","Druze","Maronite", "Tajiks"]
    category_middle_east = get_category(cat, ["American"], common_ignore + ["African","Asian","Latino","Comic","American_descent"], middle_eastern)
    name_middle_eastern = get_people(people, category_middle_east)

    # Native American (to exclude)
    native_american = ["Native_American","Blackfoot","Chamorro","Cherokee","Cheyenne","Chickasaw","Choctaw","Comanche","Kiowa", "Lakota", "Lumbee", "Mohawk", "Muscogee", "Métis", "First_Nations", "Navajo", "Ojibwe", "Osage", "Paiute", "Potawatomi", "Seminole", "Sioux", "Wyandot", "Yaqui", "Yupik"]
    category_native_american = get_category(cat, ["American"], common_ignore + ["African","Asian","Latino","Comic","American_descent"], native_american)
    name_native_american = get_people(people, category_native_american)

    # Ignore (to exclude)
    ignore = ["Creole","Fijian","Hawaiian","Indo-Fijian","Maltese", "Marshallese", "Mestizo", "Micronesia", "Montenegrin", "Māori", "Niuean", "Polynesian", "Rapanui", "Rotuman", "Samoan", "Kiribati", "Azorean"]
    category_ignore = get_category(cat, ["American"], common_ignore + ["African","Asian","Latino","Comic","American_descent"], ignore)
    name_ignore = get_people(people, category_ignore)

    # White
    category_white = get_category(cat, ["Category:American"], common_ignore + ["Central_American","American_Mormon", "Latino_American","American_Games","Native_American","South_American","African-American","American_descent","American_English","American_Samoa","alumni","emigrants"]+asians)
    category_white = category_white - category_latino - category_asian - category_african_american - category_west_asian - category_middle_east - category_native_american - category_ignore

    name_white = get_people(people, category_white, explicit_exclude)
    name_white = name_white - name_latino - name_asians - name_african_american - name_middle_eastern - name_native_american - name_ignore

    people_to_wikidata = get_living_people_wiki_data()
    wikidata_groups_to_exclude = set([s.split()[0] for s in open("./exclude_from_control.txt").readlines()])
    foreign = set(asians+africans+latinos+middle_eastern+ignore+native_american)
    for n in list(name_white):
        if n in people_to_wikidata:
            ethnic_groups = get_properties(people_to_wikidata[n], P_ETHNIC_GROUP)
            if any([p in wikidata_groups_to_exclude for p in ethnic_groups]):
                name_white.remove(n)
                continue
        category_words = set([w for c in people[n]["categories"] for w in c.replace("Category:","").split("_")])
        for w in category_words:
            if w in foreign:
                name_white.remove(n)
                break

    for c in list(category_white):
        if all([p not in name_white for p in cat[c]]):
            category_white.remove(c)

    print("Number of white people", len(name_white))
    pickle.dump((name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, category_white), open(cache_name, 'wb'))
    return name_african_american, name_asians, name_latino, name_white, category_african_american, category_asian, category_latino, category_white

def get_gender(cat, people):
    cache_name = os.path.join(CACHE_PATH, "gender_names_cisgender.pickle")
    if os.path.exists(cache_name):
        return pickle.load(open(cache_name, 'rb'))
    # Non-binary
    with open(os.path.join(CACHE_PATH, "people_nb.pkl"),"rb") as file:
        name_nb = pickle.load(file)
    # category_nb = get_category(cat, [],[],["-binary", "Genderqueer","Asexual"])
    category_nb = get_category(cat, [],[],["-binary", "Genderqueer"]) # ,"Asexual"])
    name_nb.update(get_people(people, category_nb))

    # If Wikidata is available, use that. If not, use 
    # pronouns
    people_to_wikidata = get_living_people_wiki_data()

    name_women = []
    name_men = []
    transgender_women = []
    transgender_men = []
    for n,info in people.items():
        wikidata = people_to_wikidata.get(n, None)
        if wikidata is not None:
            gender = get_item_gender(wikidata) # this might return None
            # There are a bunch of gender categories that only have 1 person
            # We are skipping those
            if gender is None:
                continue
        else:
            gender = info['gender']

        if gender == 'F':
            name_women.append(n)
        if gender == 'M':
            name_men.append(n)
        if gender == 'nb':
            name_nb.add(n)
        if gender == 'TF':
            transgender_women.append(n)
        if gender == 'TM':
            transgender_men.append(n)


    print("Number of non-binary people", len(name_nb))

    # Women
    name_women = set(name_women) - name_nb
    print("Number of women", len(name_women))

    # Men
    name_men = set(name_men) - name_nb
    print("Number of men", len(name_men))

    # Trangender Men
    name_transgender_men = set(transgender_men) - name_nb
    print("Number of transgender men", len(name_transgender_men))

    # Trangender Women
    name_transgender_women = set(transgender_women) - name_nb
    print("Number of transgender women", len(name_transgender_women))

    category_LGBT = get_category(cat, [],[],["LGBT", "Transsexual", "Transgender"])
    print(category_LGBT)
    name_LGBT = get_people(people, category_LGBT)
    name_cisgender_men = set(name_men) - name_LGBT
    print("Number of cisgender men", len(name_cisgender_men))


    pickle.dump((name_nb, name_men, name_women, name_transgender_men, name_transgender_women, name_cisgender_men, category_nb, category_LGBT),
        open(cache_name, 'wb'))

    return name_nb, name_men, name_women, name_transgender_men, name_transgender_women, name_cisgender_men, category_nb, category_LGBT


if __name__ == "__main__":
    main()
