import numpy as np
import pandas as pd
import re
from spectrum_fundamentals.mod_string import parse_modstrings
from dlomix.constants import PTMS_ALPHABET
from typing import Tuple

PTMS_ALPHABET['W[UNIMOD:425]'] = 57
PTMS_ALPHABET['K[UNIMOD:1342]'] = 58
PTMS_ALPHABET['[UNIMOD:27]-'] = 59

def count_peptide_length(peptide):

    peptide = peptide.split('-')[-2] if len(peptide.split('-')) >= 2 else peptide
    cleaned_peptide = re.sub(r'\[.*?\]', '', peptide)
    return len(cleaned_peptide)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'peptide_length' not in df.columns:
        seqcol_name = "modified_sequence" if "modified_sequence" in df.columns else 'peptide_sequences'
        df['peptide_length'] = df[seqcol_name].apply(count_peptide_length)

    df = df.loc[(df['peptide_length'] <= 30)]

    return df

def to_dlomix(df) -> Tuple:
    df = preprocess_df(df)
    seqcol_name = "modified_sequence" if "modified_sequence" in df.columns else 'peptide_sequences'
    seq_coded_list = list(parse_modstrings(list(df[seqcol_name]), PTMS_ALPHABET, True))

    X = np.zeros((len(seq_coded_list), 32), dtype=int)

    for i in range(len(seq_coded_list)):
        seq_list = seq_coded_list[i]
        for count in range(len(seq_list)):
            X[i][count] = seq_list[count]
    
    y = df.filter(regex=("charge_[0-9]+"))
    return X, y
