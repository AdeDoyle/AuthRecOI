
from LoadDocs import get_data, conllu_parse, get_metadata
from conllu import parse, TokenList


def renumber_sent(sentence, new_num):
    """Add a new sentence ID number to a CoNLL_U file sentence"""
    meta = get_metadata(sentence)
    meta["sent_id"] = f"{new_num}"


def clean_punct(sentence):
    """Remove various forms of punctuation as necessary to split glosses into sentences"""
    return sentence


def clean_unkknown(sentence):
    """Divide glosses into separate sentences at points where unknown parts-of-speech occur"""
    return sentence


def remove_foreign(sentence):
    """Divide glosses into separate sentences at points where non-vernacular words occur, return only Irish text"""

    # Create the metadata to be added to each sub-gloss
    meta = get_metadata(sentence)
    this_id = f'# sent_id = {meta.get("sent_id")}'
    ref = f'# reference = {meta.get("reference")}'
    full_gloss = f'# text = {meta.get("text")}'
    translation = f'# translation = {meta.get("translation")}'
    if meta.get("scribe"):
        hand = f'# scribe = {meta.get("scribe")}'
        meta = f'{this_id}\n{ref}\n{hand}\n{full_gloss}\n{translation}\n'
    else:
        meta = f'{this_id}\n{ref}\n{full_gloss}\n{translation}\n'

    # List lemmata and features of tokens which cannot be a sentence on their own, and which cannot end a sentence
    split_sents = None
    exclusions = [[".i.", {'Abbr': 'Yes'}],
                  ["et", {'Foreign': 'Yes'}]]
    cur_sent = list()
    tok_id = 0
    # Check each token in a sentence to see if it's non-vernacular
    for tok_data in sentence:
        tok_id += 1
        tok_data["id"] = tok_id
        feats = tok_data.get("feats")
        if feats:
            # If a non-vernacular word is found (other than latin "et")
            if feats.get("Foreign") == "Yes" and tok_data.get("form") != "et":
                # If a sentence has been compiled up to this point
                if cur_sent:
                    # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                    # add the metadata tothe sentence, and concatenate it with any preceeding sentence splits
                    if [i for i in [[tok.get("lemma"), tok.get("feats")] for tok in cur_sent] if i not in exclusions]:
                        # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                        if [cur_sent[-1].get("lemma"), cur_sent[-1].get("feats")] in exclusions:
                            while [cur_sent[-1].get("lemma"), cur_sent[-1].get("feats")] in exclusions:
                                cur_sent = cur_sent[:-1]
                        if not split_sents:
                            split_sents = meta + TokenList(cur_sent).serialize()
                        else:
                            split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                    cur_sent = list()
                tok_id = 0
            else:
                cur_sent.append(tok_data)
        else:
            cur_sent.append(tok_data)
    split_sents = split_sents.strip("\n") + "\n"
    split_sents = parse(split_sents)

    # Renumber each substring's sentence ID to separately identify any splits made
    for i, sub_sent in enumerate(split_sents):
        meta = get_metadata(sub_sent)
        meta["sent_id"] = f'{meta.get("sent_id")}.{i}'

    return split_sents


if __name__ == "__main__":

    # Open the Wb. Glosses JSON file as wb_data
    wb_data = conllu_parse(get_data("Wb. Manual Tokenisation.json"))

    # # Open the Sg. Glosses CoNLL_U file as sg_data
    # sg_data = get_data("sga_dipsgg-ud-test_combined_POS.conllu")
    sg_data = get_data("sga_dipsgg-ud-test_split_POS.conllu")

    # TEST FUNCTIONS

    # Test renumber_sent function

    # print(get_metadata(wb_data[0]))
    # renumber_sent(wb_data[0], 123)
    # print(get_metadata(wb_data[0]))

    # Test clean_punct, clean_unknown and remove_foreign functions

    # print(clean_punct(wb_data[0]))
    # print(clean_unknown(wb_data[0]))
    # print(remove_foreign(wb_data[0]))
