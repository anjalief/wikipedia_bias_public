from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json
from utils import get_filtered_people_with_topics, load_categories, CACHE_PATH


import pickle
import os

P_ETHNIC_GROUP = "P172"
Q_AFRICAN_AMERICAN = "Q49085"

P_GENDER = "P21"
Q_FEMALE = "Q6581072"
Q_MALE = "Q6581097"
Q_INTERSEX = "Q1097630"
Q_TRANSGENDER_FEMALE = "Q1052281"
Q_TRANSGENDER_MALE = "Q2449503"
Q_NONBINARY = "Q48270"
Q_GENDER_FLUID = "Q18116794"
Q_CISGENDER_MALE = "Q15145778"
Q_CISGENDER_FEMALE = "Q15145779"

def get_living_people_wiki_data():
    return pickle.load(open(os.path.join(CACHE_PATH, "people_to_wikidata.pickle"), "rb"))

# Pull Wiki data for the people in our corpus. To run this, you first need to download
# a dump of Wikidata wikidata-20200802-all.json.bz2. We don't provide this file
# since it's very large
def cache_our_people():
    people, _ = get_filtered_people_with_topics()
    people_to_wikidata = {}

    wjd_dump_path = os.path.join(CACHE_PATH, "wikidata-20200802-all.json.bz2")
    wjd = WikidataJsonDump(wjd_dump_path)

    print(wjd)
    for ii, entity_dict in enumerate(wjd):

        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
            name = entity.get_enwiki_title().replace(" ", "_")
            if name in people:
                people_to_wikidata[name] = entity

    print("Found", len(people_to_wikidata), "out of", len(people))
    pickle.dump(people_to_wikidata, open(os.path.join(CACHE_PATH, "people_to_wikidata.pickle"), "wb"))


def has_african_american_ethnicity(item: WikidataItem, truthy: bool = True) -> bool:
    """Return True if the Wikidata Item has ethnicity African American."""
    if truthy:
        claim_group = item.get_truthy_claim_group(P_ETHNIC_GROUP)
    else:
        claim_group = item.get_claim_group(P_ETHNIC_GROUP)

    ethnicity_qids = [
        claim.mainsnak.datavalue.value["id"]
        for claim in claim_group
        if claim.mainsnak.snaktype == "value"
    ]
    print(ethnicity_qids)
    return Q_AFRICAN_AMERICAN in ethnicity_qids

def get_item_gender(item):
    ids = get_properties(item, P_GENDER)
    if Q_FEMALE in ids or Q_CISGENDER_FEMALE in ids:
        return 'F'
    if Q_TRANSGENDER_FEMALE in ids:
        return 'TF'
    if Q_MALE in ids or Q_CISGENDER_MALE in ids:
        return 'M'
    if Q_TRANSGENDER_MALE in ids:
        return 'TM'
    if Q_NONBINARY in ids or Q_GENDER_FLUID in ids:
        return 'nb'
    if Q_INTERSEX in ids:
        return 'intersex'
    return None

def get_properties(item: WikidataItem, property_code):
    claim_group = item.get_truthy_claim_group(property_code)

    qids = [
        claim.mainsnak.datavalue.value["id"]
        for claim in claim_group
        if claim.mainsnak.snaktype == "value"
    ]

    return qids


if __name__ == "__main__":
    cache_our_people()
