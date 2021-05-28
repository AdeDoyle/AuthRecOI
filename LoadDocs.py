
import os
import json
from conllu import parse, TokenList
from collections import OrderedDict


# List parts-of-speech which constitute function words
function_words = ["ADP", "CCONJ", "DET", "PRON", "SCONJ"]


def get_data(file_name):
    """return data from the file requested"""

    # Navigate to a directory containing a JSON file of the Wb. Glosses and CoNLL_U files of the Sg. Glosses
    maindir = os.getcwd()
    files_dir = os.path.join(maindir, "background files")
    try:
        os.chdir(files_dir)
    except FileNotFoundError:
        raise RuntimeError("Could not locate folder containing base files")

    # Test the file type inputted into the function
    if file_name[-7:] == ".conllu":

        # Open .conllu file-types
        with open("sga_dipsgg-ud-test_combined_POS.conllu", "r", encoding="utf-8") as conllu_file_import:
            raw_data = conllu_file_import.read()
        file_data = parse(raw_data)

    elif file_name[-5:] == ".json":

        # Open .json file-types
        with open(file_name, 'r', encoding="utf-8") as wb_json:
            file_data = json.load(wb_json)

    else:
        raise RuntimeError(f"Could not find file type in file, {file_name} - found {file_name[:]}")

    # Return to the main directory
    os.chdir(maindir)

    return file_data


def conllu_parse(json_file, tok_style=2, tagged_only=True):
    """takes the Wb. Glosses from the .json file
       changes their format to match the Sg. Glosses from the .conllu file"""

    # Extract data from the JSON file format
    parse_list = list()
    for level_0 in json_file:
        fol = level_0.get("folios")
        for level_1 in fol:
            fol_col = level_1.get("folio")
            glosses = level_1.get("glosses")
            for level_2 in glosses:
                glossnum = level_2.get("glossNo")
                gloss_text = level_2.get("glossText")
                gloss_trans = level_2.get("glossTrans")
                gloss_hand = level_2.get("glossHand")
                tokens = level_2.get(f"glossTokens{tok_style}")
                # Check that glosses have been tagged before inclusion by ensuring they contain at least one POS which
                # does not appear in Latin/Greek-only glosses.
                if tagged_only:
                    vernacular_pos = ['ADJ', 'ADP', 'AUX', 'DET', 'NOUN', 'NUM',
                                      'PART', 'PRON', 'PROPN', 'SCONJ', 'VERB']
                    if [i for i in [tok[1] for tok in tokens] if i in vernacular_pos]:
                        parse_list.append([fol_col[3:] + glossnum, gloss_text, gloss_trans, gloss_hand,
                                           [[tok[0], tok[1], tok[2], tok[3]] for tok in tokens]])
                else:
                    parse_list.append([fol_col[3:] + glossnum, gloss_text, gloss_trans, gloss_hand,
                                       [[tok[0], tok[1], tok[2], tok[3]] for tok in tokens]])

    # Compile the data into CoNLL_U file format
    conllu_format = None
    for sentnum, sent in enumerate(parse_list):
        sent_id = sentnum + 1
        this_id = f'# sent_id = {sent_id}'
        gloss_id = sent[0]
        ref = f'# reference = {gloss_id}'
        sent_toks = sent[4]
        full_gloss = f'# text = {sent[1]}'
        full_gloss = "".join(full_gloss.split("<em>"))
        full_gloss = "".join(full_gloss.split("</em>"))
        full_gloss = "".join(full_gloss.split("<sup>"))
        full_gloss = "".join(full_gloss.split("</sup>"))
        translation = f'# translation = {sent[2]}'
        translation = "".join(translation.split("<em>"))
        translation = "".join(translation.split("</em>"))
        translation = "".join(translation.split("<sup>"))
        translation = "".join(translation.split("</sup>"))
        hand = f'# scribe = {sent[3]}'
        meta = f'{this_id}\n{ref}\n{hand}\n{full_gloss}\n{translation}\n'
        sent_list = list()
        for i, tok_data in enumerate(sent_toks):
            tok_id = i + 1
            tok = tok_data[0]
            head = tok_data[2]
            if not head:
                head = "_"
            pos = tok_data[1]
            feats = tok_data[3]
            if pos in ["<Latin>", "<Latin CCONJ>", "<Greek>"] and (not feats or feats == "Foreign=Yes"):
                pos = "X"
                feats = "Foreign=Yes"
            elif pos in ["<Latin>", "<Latin CCONJ>"]:
                raise RuntimeError(f"Latin word found with features: {feats}")
            if feats:
                feats = feats.split("|")
                feats = OrderedDict({i.split("=")[0]: i.split("=")[1] for i in feats})
            compiled_tok = OrderedDict({'id': tok_id, 'form': tok, 'lemma': head, 'upostag': pos, 'xpostag': None,
                                        'feats': feats, 'head': None, 'deprel': None, 'deps': None, 'misc': None})
            sent_list.append(compiled_tok)
        sent_list = TokenList(sent_list).serialize()
        if not conllu_format:
            conllu_format = meta + sent_list
        else:
            conllu_format = conllu_format + meta + sent_list
    conllu_format = conllu_format.strip("\n") + "\n"
    conllu_format = parse(conllu_format)
    return conllu_format


def get_tokens(sentence, standardise_tokens=False, add_feats=False):
    """return just the tokens from a parsed .conllu sentence"""
    if standardise_tokens:
        if add_feats:
            tokens = [f'{tok.get("lemma")}{tok.get("upos")}{tok.get("feats")}' for tok in sentence
                      if tok.get("upos") != "PUNCT"]
            tokens = ["".join(i.split(":")) for i in tokens]
            tokens = ["".join(i.split("{")) for i in tokens]
            tokens = ["".join(i.split("}")) for i in tokens]
            tokens = ["".join(i.split(" ")) for i in tokens]
            tokens = ["".join(i.split("'")) for i in tokens]
        else:
            tokens = [f'{tok.get("lemma")}{tok.get("upos")}' for tok in sentence if tok.get("upos") != "PUNCT"]
        tokens = ["".join(i.split(".")) for i in tokens]
    else:
        if add_feats:
            tokens = [f'{tok.get("form")}{tok.get("feats")}' for tok in sentence]
        else:
            tokens = [tok.get("form") for tok in sentence]
    return tokens


def get_funcwrds(sentence, standardise_tokens=False, add_feats=False):
    """return just the tokens from a parsed .conllu sentence whicha re function words"""
    func_sent = [i for i in sentence if i.get("upos") in function_words + ["X"]]
    func_sent = [i for i in func_sent if i.get("upos") != "X" or i.get("upos") == "X" and i.get("lemma") == "et"]
    func_sent = [i for i in func_sent if i.get("upos") != "ADP"
                 or i.get("upos") == "ADP" and "PronType" in i.get("feats")]

    if standardise_tokens:
        if add_feats:
            func_wrds = [f'{tok.get("lemma")}{tok.get("upos")}{tok.get("feats")}' for tok in func_sent
                         if tok.get("upos") != "PUNCT"]
            func_wrds = ["".join(i.split(":")) for i in func_wrds]
            func_wrds = ["".join(i.split("{")) for i in func_wrds]
            func_wrds = ["".join(i.split("}")) for i in func_wrds]
            func_wrds = ["".join(i.split(" ")) for i in func_wrds]
            func_wrds = ["".join(i.split("'")) for i in func_wrds]
        else:
            func_wrds = [f'{tok.get("lemma")}{tok.get("upos")}' for tok in func_sent if tok.get("upos") != "PUNCT"]
        func_wrds = ["".join(i.split(".")) for i in func_wrds]
    else:
        if add_feats:
            func_wrds = [f'{tok.get("form")}{tok.get("feats")}' for tok in func_sent]
        else:
            func_wrds = [tok.get("form") for tok in func_sent]

    return func_wrds


def get_lemmas(sentence):
    """return just the lemmata from a parsed .conllu sentence"""
    lemmas = [tok.get("lemma") for tok in sentence]
    return lemmas


def get_pos(sentence):
    """return just the parts-of-speech from a parsed .conllu sentence"""
    pos = [tok.get("upos") for tok in sentence]
    return pos


def get_feats(sentence):
    """return just the features from a parsed .conllu sentence"""
    feats = [tok.get("feats") for tok in sentence]
    return feats


def get_toks_data(sentence):
    """return all token data for a sentence"""
    return zip(get_tokens(sentence), get_lemmas(sentence), get_pos(sentence), get_feats(sentence))


def get_metadata(sentence):
    """return all metadata for a sentence in the form of a dictionary"""
    return sentence.metadata


if __name__ == "__main__":

    # Open the Wb. Glosses JSON file as wb_data
    wb_data = conllu_parse(get_data("Wb. Manual Tokenisation.json"))

    # # Open the Sg. Glosses CoNLL_U file as sg_data
    # sg_data = get_data("sga_dipsgg-ud-test_combined_POS.conllu")
    sg_data = get_data("sga_dipsgg-ud-test_split_POS.conllu")

    # # Open the OI lexicon as oi_lex
    # oi_lex = get_data("Working_lexicon_file_1.json")
    oi_lex = get_data("Working_lexicon_file_2.json")

    # TEST FUNCTIONS

    # Test get_data and conllu_parse functions

    # print(wb_data[0])
    # print(sg_data[0])
    # print(oi_lex[0])

    # Test get_tokens, get_funcwrds, get_lemmas, get_pos and get_feats functions

    # print(get_tokens(sg_data[0]))
    # print(get_tokens(wb_data[0]))
    # print(get_funcwrds(sg_data[0]))
    # print(get_funcwrds(wb_data[0]))
    # print(get_lemmas(sg_data[0]))
    # print(get_lemmas(wb_data[0]))
    # print(get_pos(sg_data[0]))
    # print(get_pos(wb_data[0]))
    # print(get_feats(sg_data[0]))
    # print(get_feats(wb_data[0]))

    # Test get_sent_data and get_toks_data functions
    # print(get_toks_data(sg_data[0]))
    # print(get_toks_data(wb_data[0]))

    # print(get_metadata(sg_data[0]))
    # print(get_metadata(wb_data[0]))
