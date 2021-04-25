
from LoadDocs import get_data, conllu_parse, get_metadata, get_pos
from conllu import parse, TokenList
import itertools


# List lemmata and features of tokens which cannot be a sentence on their own, and which cannot end a sentence
exclusions = [[".i.", "ADV", {'Abbr': 'Yes'}],
              ["nó", "CCONJ", None],
              ["ocus", "CCONJ", None],
              ["et", "X", {'Foreign': 'Yes'}],
              ['⁊rl.', "ADV", {'Abbr': 'Yes'}]]


def renumber_sent(sentence, new_num):
    """Add a new sentence ID number to a CoNLL_U file sentence"""
    meta = get_metadata(sentence)
    meta["sent_id"] = f"{new_num}"


def clean_punct(sentence):
    """Remove various forms of punctuation as necessary to split glosses into sentences"""

    # Check if any POS in the sentence identifies punctuation. If not, return the sentence unchanged.
    unknown_check = get_pos(sentence)
    if "PUNCT" not in unknown_check:
        return [sentence]
    else:

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

        # List punctuation types which can be removed without splitting a sentence
        removables = [",", ":"]
        keepables = ["·"]

        split_sents = None
        cur_sent = list()
        tok_id = 0
        # Check each POS in a sentence to see if it's unknown
        for tok_num, tok_data in enumerate(sentence):
            tok_id += 1
            tok_data["id"] = tok_id
            pos = tok_data.get("upos")
            lemma = tok_data.get("lemma")
            # If a POS identifies punctuation that splits a sentence
            if pos == "PUNCT" and lemma not in removables and lemma not in keepables:
                # If a sentence has been compiled up to this point
                if cur_sent:
                    # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                    # add the metadata to the sentence, and concatenate it with any preceding sentence splits
                    if [i for i in [[
                        tok.get("lemma"),
                        tok.get("upos"),
                        tok.get("feats")
                    ] for tok in cur_sent] if i not in exclusions]:
                        # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                        if [
                            cur_sent[-1].get("lemma"),
                            cur_sent[-1].get("upos"),
                            cur_sent[-1].get("feats")
                        ] in exclusions:
                            while [
                                cur_sent[-1].get("lemma"),
                                   cur_sent[-1].get("upos"),
                                   cur_sent[-1].get("feats")
                            ] in exclusions:
                                cur_sent = cur_sent[:-1]
                        if not split_sents:
                            split_sents = meta + TokenList(cur_sent).serialize()
                        else:
                            split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                    cur_sent = list()
                tok_id = 0
            # If a POS identifies punctuation that can be removed without splitting a sentence
            elif pos == "PUNCT" and lemma in removables:
                if tok_num == len(sentence) - 1:
                    # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                    # add the metadata to the sentence, and concatenate it with any preceeding sentence splits
                    if [i for i in [[
                        tok.get("lemma"),
                        tok.get("upos"),
                        tok.get("feats")
                    ] for tok in cur_sent] if i not in exclusions]:
                        # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                        if [
                            cur_sent[-1].get("lemma"),
                            cur_sent[-1].get("upos"),
                            cur_sent[-1].get("feats")
                        ] in exclusions:
                            while [
                                cur_sent[-1].get("lemma"),
                                cur_sent[-1].get("upos"),
                                cur_sent[-1].get("feats")
                            ] in exclusions:
                                cur_sent = cur_sent[:-1]
                        if not split_sents:
                            split_sents = meta + TokenList(cur_sent).serialize()
                        else:
                            split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                    cur_sent = list()
                    tok_id = 0
                else:
                    tok_id -= 1
            # If this is the last token in the sentence add it to the current split
            elif tok_num == len(sentence) - 1:
                cur_sent.append(tok_data)
                # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                # add the metadata to the sentence, and concatenate it with any preceeding sentence splits
                if [i for i in [[
                    tok.get("lemma"),
                    tok.get("upos"),
                    tok.get("feats")
                ] for tok in cur_sent] if i not in exclusions]:
                    # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                    if [
                        cur_sent[-1].get("lemma"),
                        cur_sent[-1].get("upos"),
                        cur_sent[-1].get("feats")
                    ] in exclusions:
                        while [
                            cur_sent[-1].get("lemma"),
                            cur_sent[-1].get("upos"),
                            cur_sent[-1].get("feats")
                        ] in exclusions:
                            cur_sent = cur_sent[:-1]
                    if not split_sents:
                        split_sents = meta + TokenList(cur_sent).serialize()
                    else:
                        split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                cur_sent = list()
                tok_id = 0
            else:
                cur_sent.append(tok_data)

        if not split_sents:
            split_sents = []
        else:
            split_sents = split_sents.strip("\n") + "\n"
            split_sents = parse(split_sents)

        # Renumber each substring's sentence ID to separately identify any splits made
        for i, sub_sent in enumerate(split_sents):
            meta = get_metadata(sub_sent)
            meta["sent_id"] = f'{meta.get("sent_id")}.{i}'

        return split_sents


def clean_unknown(sentence):
    """Divide glosses into separate sentences at points where unknown parts-of-speech occur"""

    # Check if any POS in the sentence is unknown. If not, return the sentence unchanged.
    unknown_check = get_pos(sentence)
    if "<unknown>" not in unknown_check:
        return [sentence]
    else:

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

        split_sents = None
        cur_sent = list()
        tok_id = 0
        # Check each POS in a sentence to see if it's unknown
        for tok_num, tok_data in enumerate(sentence):
            tok_id += 1
            tok_data["id"] = tok_id
            pos = tok_data.get("upos")
            # If a POS is unknown
            if pos == "<unknown>":
                # If a sentence has been compiled up to this point
                if cur_sent:
                    # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                    # add the metadata to the sentence, and concatenate it with any preceding sentence splits
                    if [i for i in [[
                        tok.get("lemma"),
                        tok.get("upos"),
                        tok.get("feats")
                    ] for tok in cur_sent] if i not in exclusions]:
                        # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                        if [
                            cur_sent[-1].get("lemma"),
                            cur_sent[-1].get("upos"),
                            cur_sent[-1].get("feats")
                        ] in exclusions:
                            while [
                                cur_sent[-1].get("lemma"),
                                cur_sent[-1].get("upos"),
                                cur_sent[-1].get("feats")
                            ] in exclusions:
                                cur_sent = cur_sent[:-1]
                        if not split_sents:
                            split_sents = meta + TokenList(cur_sent).serialize()
                        else:
                            split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                    cur_sent = list()
                tok_id = 0
            # If this is the last token in the sentence add it to the current split
            elif tok_num == len(sentence) - 1:
                cur_sent.append(tok_data)
                # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                # add the metadata tothe sentence, and concatenate it with any preceeding sentence splits
                if [i for i in [[
                    tok.get("lemma"),
                    tok.get("upos"),
                    tok.get("feats")
                ] for tok in cur_sent] if i not in exclusions]:
                    # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                    if [
                        cur_sent[-1].get("lemma"),
                        cur_sent[-1].get("upos"),
                        cur_sent[-1].get("feats")
                    ] in exclusions:
                        while [
                            cur_sent[-1].get("lemma"),
                            cur_sent[-1].get("upos"),
                            cur_sent[-1].get("feats")
                        ] in exclusions:
                            cur_sent = cur_sent[:-1]
                    if not split_sents:
                        split_sents = meta + TokenList(cur_sent).serialize()
                    else:
                        split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                cur_sent = list()
                tok_id = 0
            else:
                cur_sent.append(tok_data)

        if not split_sents:
            split_sents = []
        else:
            split_sents = split_sents.strip("\n") + "\n"
            split_sents = parse(split_sents)

        # Renumber each substring's sentence ID to separately identify any splits made
        for i, sub_sent in enumerate(split_sents):
            meta = get_metadata(sub_sent)
            meta["sent_id"] = f'{meta.get("sent_id")}.{i}'

        return split_sents


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

    split_sents = None
    cur_sent = list()
    tok_id = 0
    # Check each token in a sentence to see if it's non-vernacular
    for tok_num, tok_data in enumerate(sentence):
        tok_id += 1
        tok_data["id"] = tok_id
        feats = tok_data.get("feats")
        if feats:
            # If a non-vernacular word is found (other than latin "et")
            if feats.get("Foreign") == "Yes" and tok_data.get("form") != "et":
                # If a sentence has been compiled up to this point
                if cur_sent:
                    # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                    # add the metadata to the sentence, and concatenate it with any preceding sentence splits
                    if [i for i in [[
                        tok.get("lemma"),
                        tok.get("upos"),
                        tok.get("feats")
                    ] for tok in cur_sent] if i not in exclusions]:
                        # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                        if [
                            cur_sent[-1].get("lemma"),
                            cur_sent[-1].get("upos"),
                            cur_sent[-1].get("feats")
                        ] in exclusions:
                            while [
                                cur_sent[-1].get("lemma"),
                                cur_sent[-1].get("upos"),
                                cur_sent[-1].get("feats")
                            ] in exclusions:
                                cur_sent = cur_sent[:-1]
                        if not split_sents:
                            split_sents = meta + TokenList(cur_sent).serialize()
                        else:
                            split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                    cur_sent = list()
                tok_id = 0
            # If this is the last token in the sentence and it has features add it to the current split
            elif tok_num == len(sentence) - 1:
                cur_sent.append(tok_data)
                # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
                # add the metadata to the sentence, and concatenate it with any preceding sentence splits
                if [i for i in [[
                    tok.get("lemma"),
                    tok.get("upos"),
                    tok.get("feats")
                ] for tok in cur_sent] if i not in exclusions]:
                    # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                    if [
                        cur_sent[-1].get("lemma"),
                        cur_sent[-1].get("upos"),
                        cur_sent[-1].get("feats")
                    ] in exclusions:
                        while [
                            cur_sent[-1].get("lemma"),
                            cur_sent[-1].get("upos"),
                            cur_sent[-1].get("feats")
                        ] in exclusions:
                            cur_sent = cur_sent[:-1]
                    if not split_sents:
                        split_sents = meta + TokenList(cur_sent).serialize()
                    else:
                        split_sents = split_sents + meta + TokenList(cur_sent).serialize()
                cur_sent = list()
                tok_id = 0
            else:
                cur_sent.append(tok_data)
        # If this is the last token in the sentence but it has no features add it to the current split
        elif tok_num == len(sentence) - 1:
            cur_sent.append(tok_data)
            # If all of the words in the compiled sentence are valid, Irish words (not exclusions)
            # add the metadata to the sentence, and concatenate it with any preceding sentence splits
            if [i for i in [[
                tok.get("lemma"),
                tok.get("upos"),
                tok.get("feats")
            ] for tok in cur_sent] if i not in exclusions]:
                # If any split sentence ends with an invalid word (an exclusion) remove it from the end
                if [
                    cur_sent[-1].get("lemma"),
                    cur_sent[-1].get("upos"),
                    cur_sent[-1].get("feats")
                ] in exclusions:
                    while [
                        cur_sent[-1].get("lemma"),
                        cur_sent[-1].get("upos"),
                        cur_sent[-1].get("feats")
                    ] in exclusions:
                        cur_sent = cur_sent[:-1]
                if not split_sents:
                    split_sents = meta + TokenList(cur_sent).serialize()
                else:
                    split_sents = split_sents + meta + TokenList(cur_sent).serialize()
            cur_sent = list()
            tok_id = 0
        else:
            cur_sent.append(tok_data)

    if not split_sents:
        split_sents = []
    else:
        split_sents = split_sents.strip("\n") + "\n"
        split_sents = parse(split_sents)

    # Renumber each substring's sentence ID to separately identify any splits made
    for i, sub_sent in enumerate(split_sents):
        meta = get_metadata(sub_sent)
        meta["sent_id"] = f'{meta.get("sent_id")}.{i}'

    return split_sents


def clean_all(sentence):
    """Apply all cleaning to a single sentence, i.e. split it on punctuation, unknown POS and foreign words"""
    split_sents = clean_punct(sentence)
    split_sents = list(itertools.chain(*[clean_unknown(splitsent) for splitsent in split_sents]))
    split_sents = list(itertools.chain(*[remove_foreign(splitsent) for splitsent in split_sents]))
    return split_sents


def file_clean(file):
    """Apply all cleaning to all sentences in a file, and renumber all sentences appropriately"""
    outfile = list(itertools.chain(*[clean_all(sent) for sent in file]))
    for sent_num, sent in enumerate(outfile):
        renumber_sent(sent, sent_num + 1)
    return outfile


if __name__ == "__main__":

    # # Open the Wb. Glosses JSON file as wb_data
    wb_data = conllu_parse(get_data("Wb. Manual Tokenisation.json"))

    # # Open the Sg. Glosses CoNLL_U file as sg_data
    # sg_data = get_data("sga_dipsgg-ud-test_combined_POS.conllu")
    sg_data = get_data("sga_dipsgg-ud-test_split_POS.conllu")

    # # TEST FUNCTIONS

    # # Test renumber_sent function

    # print(get_metadata(wb_data[0]))
    # renumber_sent(wb_data[0], 123)
    # print(get_metadata(wb_data[0]))

    # # Test clean_punct, clean_unknown and remove_foreign functions

    # print(clean_punct(wb_data[26]))
    # print(clean_unknown(wb_data[26]))
    # print(remove_foreign(wb_data[0]))

    # # Test clean_all and file_clean functions

    # print(clean_all(wb_data[18]))
    # print(file_clean(wb_data))
