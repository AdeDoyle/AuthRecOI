
# from CombineInfoLists import combine_infolists
# from MakeJSON import make_json, make_lex_json
# from SaveJSON import save_json
import os
import json
from conllu import parse, TokenList
# from decimal import Decimal
# from ClearTags import clear_tags
# from tkinter import *
# from tkinter import ttk
# from tkinter import font
# from nltk import edit_distance
# import unidecode
# import platform


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


if __name__ == "__main__":

    # Open the Wb. Glosses JSON file as wb_data
    wb_data = get_data("Wb. Manual Tokenisation.json")

    # # Open the Sg. Glosses CoNLL_U file as sg_data
    # sg_data = get_data("sga_dipsgg-ud-test_combined_POS.conllu")
    sg_data = get_data("sga_dipsgg-ud-test_split_POS.conllu")

    # # Open the OI lexicon as oi_lex
    # oi_lex = get_data("Working_lexicon_file_1.json")
    oi_lex = get_data("Working_lexicon_file_2.json")

    # print(wb_data)
    # print(sg_data)
    # print(oi_lex)

    for token in sg_data[0]:
        print(token)