
from LoadDocs import get_data, conllu_parse, get_metadata, get_tokens, get_funcwrds, get_pos
from conllu import parse, TokenList
import itertools
import re


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
    """Apply all cleaning to a single sentence, i.e. split it on punctuation, unknown POS, and foreign words"""
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


def separate_hands(file):
    """Separates a file into separate documents based on scribal hand (requires this metadata to be available)"""
    all_hands = sorted(list(set(sent.metadata.get("scribe") for sent in file)))
    separated_files = list()
    for hand in all_hands:
        separated_files.append([sent for sent in file if sent.metadata.get("scribe") == hand])
    return separated_files


def separate_columns(file):
    """Separates a file into individual documents based on folio-column (requires this metadata to be available)"""
    all_columns = [sent.metadata.get("reference") for sent in file]
    for i, folcol in enumerate(all_columns):
        if folcol == "Not in Thesaurus Palaeohibernicus":
            all_columns[i] = "No folio information available"
        else:
            new_folcol = None
            folcolpat = re.compile(r'^(\d{1,3}([a-d]|,\d)|f\.\d{1,3}[a-d]?)')
            folcolpatiter = folcolpat.finditer(folcol)
            for j in folcolpatiter:
                new_folcol = j.group()
            if new_folcol:
                if "," in new_folcol:
                    new_folcol = new_folcol[:-2]
                elif "f." in new_folcol:
                    new_folcol = new_folcol[2:]
                if new_folcol[-1] not in ["a", "b", "c", "d"]:
                    new_folcol = new_folcol + "x"
                all_columns[i] = re.sub(r'^(\d{1,3}([a-d]|,\d)|f\.\d{1,3}[a-d]?).*', new_folcol, all_columns[i])
            else:
                raise RuntimeError(f"No folio pattern found for folio/column:\n    {folcol}")
    try:
        all_columns = [[int(folcol[:-1]), folcol[-1:]] for folcol in set(all_columns)]
        all_columns.sort(key=lambda x: x[1])
        all_columns.sort(key=lambda x: x[0])
        all_columns = [f'{folcol[0]}{folcol[1]}' for folcol in all_columns]
    except ValueError:
        new_columns = list()
        for folcol in set(all_columns):
            if folcol == "No folio information available":
                folcol = [0, "No folio information available"]
            else:
                folcol = [int(folcol[:-1]), folcol[-1:]]
            new_columns.append(folcol)
        all_columns = new_columns
        all_columns.sort(key=lambda x: x[1])
        all_columns.sort(key=lambda x: x[0])
        all_columns = [f'{folcol[0]}{folcol[1]}' if (folcol[1] != "x" and folcol[0] != 0)
                       else folcol for folcol in all_columns]
        all_columns = [folcol[1] if folcol[0] == 0 else folcol for folcol in all_columns]
        all_columns = [f'{folcol[0]}' if isinstance(folcol, list) and folcol[-1] == "x"
                       else folcol for folcol in all_columns]
    separated_columns = list()
    for folcol in all_columns:
        returned_sents = [sent for sent in file if sent.metadata.get("reference")[:len(folcol)] == folcol]
        if not returned_sents:
            if folcol == "No folio information available":
                returned_sents = [sent for sent in file
                                  if sent.metadata.get("reference") == "Not in Thesaurus Palaeohibernicus"]
            else:
                returned_sents = [sent for sent in file if sent.metadata.get("reference")[2:len(folcol) + 2] == folcol]
                if not returned_sents:
                    raise RuntimeError(f"Unknown folio-column: {folcol}")
        separated_columns.append(returned_sents)
    return separated_columns


def compile_hand_data(file, function_words=False):
    """Compiles a list of glosses for each hand, then creates a list of lables for each"""
    cleaned_data = file_clean(file)
    hand_data = separate_hands(cleaned_data)
    hand_labels = list()
    compiled_data = list()
    compiled_fw_data = list()
    for handlist in hand_data:
        hand_labels.append(handlist[0].metadata.get("scribe"))
        compiled_data.append(handlist)
    if function_words:
        compiled_fw_data = compiled_data[:]
        for i, hand_fol in enumerate(compiled_fw_data):
            compiled_fw_data[i] = [" ".join(get_funcwrds(sent)) for sent in hand_fol]
        compiled_fw_data = ["\n".join(gloss_text) for gloss_text in compiled_fw_data]
        for i, fw_datum in enumerate(compiled_fw_data):
            if "\n" in fw_datum:
                fw_datum = fw_datum.strip()
                while "\n\n" in fw_datum:
                    fw_datum = "\n".join(fw_datum.split("\n\n"))
                compiled_fw_data[i] = fw_datum
    for i, hand_fol in enumerate(compiled_data):
        compiled_data[i] = [" ".join(get_tokens(sent)) for sent in hand_fol]
    compiled_data = ["\n".join(gloss_text) for gloss_text in compiled_data]
    if function_words:
        return [compiled_data, compiled_fw_data, hand_labels]
    else:
        return [compiled_data, hand_labels]


def compile_doc_data(file, function_words=False, standard_word_forms=False, add_feats=False):
    """Compiles a list of glosses for each hand and folio-column, then creates a list of lables for each"""
    cleaned_data = file_clean(file)
    hand_data = separate_hands(cleaned_data)
    hand_labels = list()
    compiled_data = list()
    compiled_fw_data = list()
    for handlist in hand_data:
        hand = handlist[0].metadata.get("scribe")
        # collect all glosses by rare hands in one group
        if hand in ["Hand One (Prima Manus) and Hand Two",
                    "Main Hand", "Main Hand (Ogam)", "Glossator A (Ogam)",
                    "Glossator C", "Glossator C?", "Glossator C+E",
                    "Glossator D", "Glossator D?", "Glossator G", "Glossator G/C?",
                    "Glossator _", "Glossator +"
                    ]:
            compiled_data.append(handlist)
            hand_labels.append(hand)
        # collect all Wb. Prima Manus glosses, then split them into any specified number of groups
        elif hand == "Hand One (Prima Manus)":
            # list points to split Prima Manus glosses, eg. ["7c", "14b", "18c"]
            cutoffs = []
            if cutoffs:
                previous_cutoff = None
                pos_cols = ["a", "b", "c", "d"]
                for co_num, cutoff in enumerate(cutoffs):
                    fol_cutoff = int(cutoff[:-1])
                    col_cutoff = pos_cols.index(cutoff[-1:])
                    # if this is the first division of text for this hand
                    # add all glosses to the cutoff point to a list
                    # add that list to list of gloss divisions for this hand
                    if not previous_cutoff:
                        break_index = 0
                        for i, sent in enumerate(handlist):
                            this_folcol = sent.metadata.get("reference")
                            folcolpat = re.compile(r'^\d{1,3}[a-d]')
                            folcolpatiter = folcolpat.finditer(this_folcol)
                            for j in folcolpatiter:
                                this_folcol = j.group()
                            this_fol = int(this_folcol[:-1])
                            this_col = pos_cols.index(this_folcol[-1:])
                            if this_fol > fol_cutoff or (this_fol == fol_cutoff and this_col >= col_cutoff):
                                previous_cutoff = break_index
                                break
                            else:
                                break_index = i + 1
                        compiled_data.append(handlist[:break_index])
                        hand_labels.append(hand)
                    # if this is neither the first nor the last division of text for this hand
                    # add all glosses between the last cutoff point and this cutoff point to a list
                    # add that list to list of gloss divisions for this hand
                    else:
                        break_index = 0
                        for i, sent in enumerate(handlist):
                            this_folcol = sent.metadata.get("reference")
                            folcolpat = re.compile(r'^\d{1,3}[a-d]')
                            folcolpatiter = folcolpat.finditer(this_folcol)
                            for j in folcolpatiter:
                                this_folcol = j.group()
                            this_fol = int(this_folcol[:-1])
                            this_col = pos_cols.index(this_folcol[-1:])
                            if this_fol > fol_cutoff or (this_fol == fol_cutoff and this_col >= col_cutoff):
                                compiled_data.append(handlist[previous_cutoff:break_index])
                                hand_labels.append(hand)
                                previous_cutoff = break_index
                                break
                            else:
                                break_index = i + 1
                    # if this is the last division of text for this hand
                    # add all remaining glosses to a list
                    # add that to list of gloss divisions for this hand
                    if co_num + 1 == len(cutoffs):
                        compiled_data.append(handlist[previous_cutoff:])
                        hand_labels.append(hand)
            else:
                compiled_data.append(handlist)
                hand_labels.append(hand)
        else:
            hand_folcol_list = list()
            cur_folcol = None
            # iterate through sentences in each hand-list and get its folio and column information
            for i, sent in enumerate(handlist):
                this_folcol = sent.metadata.get("reference")
                folcolpat = re.compile(r'^\d{1,3}[a-d]')
                folcolpatiter = folcolpat.finditer(this_folcol)
                for j in folcolpatiter:
                    this_folcol = j.group()
                # if this is the last sentence in the hand-list
                # add the complete list of glosses for this hand on this folio and column to the compiled data list
                if i+1 == len(handlist):
                    hand_labels.append(hand)
                    hand_folcol_list.append(sent)
                    compiled_data.append(hand_folcol_list)
                # if this is the first folio and column update the current folio and column information,
                # start a new list of glosses for this hand on the new folio and column
                elif not cur_folcol:
                    cur_folcol = this_folcol
                    hand_folcol_list = [sent]
                # if this folio/column is different to the last one update the current folio and column information,
                # add the complete list of glosses for the hand on the last folio and column to the compiled data list
                # start a new list of glosses for this hand on the new folio and column
                elif this_folcol != cur_folcol:
                    hand_labels.append(hand)
                    compiled_data.append(hand_folcol_list)
                    cur_folcol = this_folcol
                    hand_folcol_list = [sent]
                # if this this folio/column is the same as the last one
                # add the current sentence to the list of glosses for this hand on the current folio and column
                elif this_folcol == cur_folcol:
                    hand_folcol_list.append(sent)
    if function_words:
        compiled_fw_data = compiled_data[:]
        for i, hand_fol in enumerate(compiled_fw_data):
            compiled_fw_data[i] = [" ".join(get_funcwrds(sent, standard_word_forms, add_feats)) for sent in hand_fol]
        compiled_fw_data = [" ".join(gloss_text) for gloss_text in compiled_fw_data]
        compiled_fw_data = [i for i in compiled_fw_data if i != '']
        for i, hand_datum in enumerate(compiled_fw_data):
            if " " in hand_datum:
                while "  " in hand_datum:
                    hand_datum = " ".join(hand_datum.split("  "))
                hand_datum = hand_datum.strip()
                compiled_fw_data[i] = hand_datum
    for i, hand_fol in enumerate(compiled_data):
        compiled_data[i] = [" ".join(get_tokens(sent, standard_word_forms, add_feats)) for sent in hand_fol]
    compiled_data = [" ".join(gloss_text) for gloss_text in compiled_data]
    if function_words:
        return [compiled_data, compiled_fw_data, hand_labels]
    else:
        return [compiled_data, hand_labels]


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

    # # Test separate_hands and separate_columns functions

    # print(separate_hands(wb_data))
    # print(separate_columns(wb_data))
    # print(separate_hands(sg_data))
    # print(separate_columns(sg_data))

    # print(len(separate_hands(wb_data)))
    # print(len(separate_columns(wb_data)))
    # print(len(separate_hands(sg_data)))
    # print(len(separate_columns(sg_data)))

    # # Test compile_doc_data and compile_hand_data functions

    # print(compile_hand_data(wb_data))
    # print(compile_hand_data(wb_data, True))
    # print(compile_doc_data(wb_data))
    # print(compile_doc_data(wb_data, True))
    # print(compile_doc_data(wb_data, True, True))
    # print(compile_doc_data(wb_data, True, True, True))

    # print(compile_hand_data(sg_data))
    # print(compile_hand_data(sg_data, True))
    # print(compile_doc_data(sg_data))
    # print(compile_doc_data(sg_data, True))
    # print(compile_doc_data(sg_data, True, True))
    # print(compile_doc_data(sg_data, True, True, True))
