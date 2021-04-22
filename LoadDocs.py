
import os
import json
from conllu import parse, TokenList


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


def get_tokens(sentence):
    """return just the tokens from a parsed .conllu sentence"""
    tokens = [tok.get("form") for tok in sentence]
    return tokens


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


if __name__ == "__main__":

    # Open the Wb. Glosses JSON file as wb_data
    wb_data = get_data("Wb. Manual Tokenisation.json")

    # # Open the Sg. Glosses CoNLL_U file as sg_data
    # sg_data = get_data("sga_dipsgg-ud-test_combined_POS.conllu")
    sg_data = get_data("sga_dipsgg-ud-test_split_POS.conllu")

    # # Open the OI lexicon as oi_lex
    # oi_lex = get_data("Working_lexicon_file_1.json")
    oi_lex = get_data("Working_lexicon_file_2.json")

    # print(wb_data[0])
    # print(sg_data[0])
    # print(oi_lex[0])

    # print(get_tokens(sg_data[0]))
    # print(get_lemmas(sg_data[0]))
    # print(get_pos(sg_data[0]))
    # print(get_feats(sg_data[0]))
