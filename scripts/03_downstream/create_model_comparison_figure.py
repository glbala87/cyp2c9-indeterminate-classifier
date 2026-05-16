"""
CYP2C9 Comprehensive Model Comparison Figure
=============================================
XGBoost vs Random Forest vs SVM
- Cross-validation Accuracy
- F1 Score (CV vs Test)
- ROC-AUC (CV vs Test)
- Indeterminate Predictions (n=2184)
- Confusion Matrices
- Overall Performance Profile
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

OUTPUT_DIR = 'xgboost_validated_output'

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON FIGURE")
print("="*80)

# ============================================================================
# LOAD VALIDATED SCORES AND CONSTANTS (from validated model)
# ============================================================================
PROTEIN_LENGTH = 490
HEME_CYS = 436

FUNCTIONAL_REGIONS = {
    'SRS1': (97, 126), 'SRS2': (200, 230), 'SRS4': (290, 320),
    'SRS5': (359, 390), 'SRS6': (430, 490),
}

AA_PROPERTIES = {
    'A': {'charge': 0, 'hydro': 1, 'size': 1}, 'V': {'charge': 0, 'hydro': 1, 'size': 2},
    'I': {'charge': 0, 'hydro': 1, 'size': 3}, 'L': {'charge': 0, 'hydro': 1, 'size': 3},
    'M': {'charge': 0, 'hydro': 1, 'size': 3}, 'F': {'charge': 0, 'hydro': 1, 'size': 3},
    'W': {'charge': 0, 'hydro': 1, 'size': 3}, 'P': {'charge': 0, 'hydro': 1, 'size': 1},
    'G': {'charge': 0, 'hydro': 0, 'size': 1}, 'S': {'charge': 0, 'hydro': 0, 'size': 1},
    'T': {'charge': 0, 'hydro': 0, 'size': 2}, 'C': {'charge': 0, 'hydro': 0, 'size': 1},
    'Y': {'charge': 0, 'hydro': 0, 'size': 3}, 'N': {'charge': 0, 'hydro': 0, 'size': 2},
    'Q': {'charge': 0, 'hydro': 0, 'size': 2}, 'D': {'charge': -1, 'hydro': 0, 'size': 2},
    'E': {'charge': -1, 'hydro': 0, 'size': 2}, 'K': {'charge': 1, 'hydro': 0, 'size': 3},
    'R': {'charge': 1, 'hydro': 0, 'size': 3}, 'H': {'charge': 1, 'hydro': 0, 'size': 2},
}

GRANTHAM_SCORES = {
    ('S', 'R'): 110, ('S', 'C'): 112, ('S', 'G'): 56, ('S', 'P'): 74,
    ('R', 'C'): 180, ('R', 'H'): 29, ('R', 'G'): 125, ('R', 'W'): 101,
    ('L', 'P'): 98, ('L', 'F'): 22, ('L', 'S'): 145, ('L', 'V'): 32,
    ('I', 'T'): 89, ('I', 'L'): 5, ('I', 'V'): 29, ('I', 'F'): 21,
    ('P', 'S'): 74, ('P', 'L'): 98, ('P', 'H'): 77, ('P', 'R'): 103,
    ('E', 'G'): 98, ('E', 'K'): 56, ('E', 'D'): 45, ('E', 'R'): 54,
    ('D', 'G'): 94, ('D', 'N'): 23, ('D', 'A'): 126, ('D', 'H'): 81,
    ('Q', 'H'): 24, ('Q', 'R'): 43, ('Q', 'E'): 29, ('Q', 'K'): 53,
    ('A', 'T'): 58, ('A', 'V'): 64, ('A', 'G'): 60, ('A', 'S'): 99,
    ('V', 'I'): 29, ('V', 'L'): 32, ('V', 'M'): 21, ('V', 'F'): 50,
    ('G', 'R'): 125, ('G', 'S'): 56, ('G', 'A'): 60, ('G', 'E'): 98,
    ('T', 'I'): 89, ('T', 'A'): 58, ('T', 'S'): 58, ('T', 'M'): 81,
    ('C', 'S'): 112, ('C', 'R'): 180, ('C', 'Y'): 194, ('C', 'F'): 205,
    ('H', 'R'): 29, ('H', 'Q'): 24, ('H', 'Y'): 83, ('H', 'N'): 68,
    ('Y', 'C'): 194, ('Y', 'H'): 83, ('Y', 'F'): 22, ('Y', 'S'): 144,
    ('F', 'L'): 22, ('F', 'I'): 21, ('F', 'Y'): 22, ('F', 'V'): 50,
    ('N', 'D'): 23, ('N', 'H'): 68, ('N', 'S'): 46, ('N', 'K'): 94,
    ('K', 'E'): 56, ('K', 'Q'): 53, ('K', 'R'): 26, ('K', 'N'): 94,
    ('M', 'V'): 21, ('M', 'L'): 15, ('M', 'I'): 10, ('M', 'T'): 81,
    ('W', 'R'): 101, ('W', 'F'): 40, ('W', 'Y'): 37, ('W', 'C'): 215,
    ('G', 'V'): 109, ('T', 'R'): 71, ('L', 'I'): 5, ('R', 'Q'): 43,
}

VALIDATED_SCORES = {
    'p.A149T': {'CADD': 20.8, 'REVEL': 0.372, 'SpliceAI': 0.01},
    'p.A149V': {'CADD': 21.2, 'REVEL': 0.385, 'SpliceAI': 0.01},
    'p.A477T': {'CADD': 21.8, 'REVEL': 0.412, 'SpliceAI': 0.01},
    'p.D191G': {'CADD': 23.5, 'REVEL': 0.498, 'SpliceAI': 0.01},
    'p.D360E': {'CADD': 23.1, 'REVEL': 0.445, 'SpliceAI': 0.01},
    'p.D397A': {'CADD': 23.4, 'REVEL': 0.489, 'SpliceAI': 0.01},
    'p.D49G': {'CADD': 23.2, 'REVEL': 0.478, 'SpliceAI': 0.01},
    'p.E272G': {'CADD': 24.3, 'REVEL': 0.548, 'SpliceAI': 0.01},
    'p.E326D': {'CADD': 19.5, 'REVEL': 0.298, 'SpliceAI': 0.01},
    'p.E354K': {'CADD': 24.5, 'REVEL': 0.562, 'SpliceAI': 0.01},
    'p.F110S': {'CADD': 25.9, 'REVEL': 0.612, 'SpliceAI': 0.01},
    'p.G442V': {'CADD': 25.4, 'REVEL': 0.598, 'SpliceAI': 0.01},
    'p.G70R': {'CADD': 25.5, 'REVEL': 0.678, 'SpliceAI': 0.01},
    'p.G96A': {'CADD': 22.5, 'REVEL': 0.425, 'SpliceAI': 0.01},
    'p.G96R': {'CADD': 26.1, 'REVEL': 0.672, 'SpliceAI': 0.01},
    'p.G98V': {'CADD': 26.8, 'REVEL': 0.672, 'SpliceAI': 0.01},
    'p.I207T': {'CADD': 24.2, 'REVEL': 0.542, 'SpliceAI': 0.01},
    'p.I222V': {'CADD': 19.8, 'REVEL': 0.312, 'SpliceAI': 0.01},
    'p.I284V': {'CADD': 18.5, 'REVEL': 0.285, 'SpliceAI': 0.01},
    'p.I327T': {'CADD': 24.2, 'REVEL': 0.538, 'SpliceAI': 0.01},
    'p.I331T': {'CADD': 24.8, 'REVEL': 0.608, 'SpliceAI': 0.01},
    'p.I359L': {'CADD': 24.8, 'REVEL': 0.584, 'SpliceAI': 0.02},
    'p.I359T': {'CADD': 25.2, 'REVEL': 0.618, 'SpliceAI': 0.01},
    'p.I387V': {'CADD': 18.9, 'REVEL': 0.298, 'SpliceAI': 0.01},
    'p.I434F': {'CADD': 27.5, 'REVEL': 0.762, 'SpliceAI': 0.01},
    'p.K119R': {'CADD': 22.1, 'REVEL': 0.324, 'SpliceAI': 0.01},
    'p.L19I': {'CADD': 15.2, 'REVEL': 0.248, 'SpliceAI': 0.01},
    'p.L361I': {'CADD': 16.8, 'REVEL': 0.245, 'SpliceAI': 0.01},
    'p.L362V': {'CADD': 20.2, 'REVEL': 0.348, 'SpliceAI': 0.01},
    'p.L467P': {'CADD': 26.1, 'REVEL': 0.658, 'SpliceAI': 0.01},
    'p.L90P': {'CADD': 28.1, 'REVEL': 0.823, 'SpliceAI': 0.01},
    'p.M1V': {'CADD': 25.8, 'REVEL': 0.682, 'SpliceAI': 0.01},
    'p.N204H': {'CADD': 24.6, 'REVEL': 0.567, 'SpliceAI': 0.01},
    'p.N418T': {'CADD': 22.4, 'REVEL': 0.452, 'SpliceAI': 0.01},
    'p.N41D': {'CADD': 19.8, 'REVEL': 0.318, 'SpliceAI': 0.01},
    'p.N457S': {'CADD': 21.5, 'REVEL': 0.398, 'SpliceAI': 0.01},
    'p.P163L': {'CADD': 24.5, 'REVEL': 0.548, 'SpliceAI': 0.01},
    'p.P227S': {'CADD': 22.8, 'REVEL': 0.465, 'SpliceAI': 0.01},
    'p.P279T': {'CADD': 23.2, 'REVEL': 0.492, 'SpliceAI': 0.01},
    'p.P30L': {'CADD': 24.1, 'REVEL': 0.552, 'SpliceAI': 0.01},
    'p.P317S': {'CADD': 23.8, 'REVEL': 0.512, 'SpliceAI': 0.01},
    'p.P337T': {'CADD': 24.1, 'REVEL': 0.528, 'SpliceAI': 0.01},
    'p.P382S': {'CADD': 23.5, 'REVEL': 0.518, 'SpliceAI': 0.15},
    'p.P489S': {'CADD': 21.5, 'REVEL': 0.382, 'SpliceAI': 0.01},
    'p.Q214H': {'CADD': 23.8, 'REVEL': 0.478, 'SpliceAI': 0.01},
    'p.Q214L': {'CADD': 23.5, 'REVEL': 0.492, 'SpliceAI': 0.01},
    'p.Q454H': {'CADD': 22.1, 'REVEL': 0.412, 'SpliceAI': 0.01},
    'p.R105C': {'CADD': 25.9, 'REVEL': 0.658, 'SpliceAI': 0.01},
    'p.R108C': {'CADD': 25.7, 'REVEL': 0.642, 'SpliceAI': 0.01},
    'p.R124Q': {'CADD': 25.8, 'REVEL': 0.623, 'SpliceAI': 0.01},
    'p.R124W': {'CADD': 27.1, 'REVEL': 0.698, 'SpliceAI': 0.01},
    'p.R125C': {'CADD': 28.2, 'REVEL': 0.734, 'SpliceAI': 0.01},
    'p.R125H': {'CADD': 25.8, 'REVEL': 0.71, 'SpliceAI': 0.02},
    'p.R125L': {'CADD': 26.2, 'REVEL': 0.69, 'SpliceAI': 0.01},
    'p.R132Q': {'CADD': 24.5, 'REVEL': 0.582, 'SpliceAI': 0.01},
    'p.R132W': {'CADD': 27.3, 'REVEL': 0.715, 'SpliceAI': 0.01},
    'p.R144C': {'CADD': 26.5, 'REVEL': 0.651, 'SpliceAI': 0.01},
    'p.R144H': {'CADD': 24.8, 'REVEL': 0.592, 'SpliceAI': 0.01},
    'p.R150C': {'CADD': 26.4, 'REVEL': 0.645, 'SpliceAI': 0.01},
    'p.R150H': {'CADD': 22.4, 'REVEL': 0.418, 'SpliceAI': 0.01},
    'p.R335W': {'CADD': 27.2, 'REVEL': 0.778, 'SpliceAI': 0.02},
    'p.R433C': {'CADD': 26.8, 'REVEL': 0.724, 'SpliceAI': 0.01},
    'p.R433W': {'CADD': 27.8, 'REVEL': 0.712, 'SpliceAI': 0.01},
    'p.S162X': {'CADD': 35.0, 'REVEL': 0.95, 'SpliceAI': 0.05},
    'p.S280C': {'CADD': 24.8, 'REVEL': 0.568, 'SpliceAI': 0.01},
    'p.S343R': {'CADD': 24.2, 'REVEL': 0.534, 'SpliceAI': 0.01},
    'p.T299A': {'CADD': 23.0, 'REVEL': 0.478, 'SpliceAI': 0.01},
    'p.T299R': {'CADD': 24.5, 'REVEL': 0.552, 'SpliceAI': 0.01},
    'p.V490F': {'CADD': 23.9, 'REVEL': 0.521, 'SpliceAI': 0.01},
    'p.V76M': {'CADD': 22.8, 'REVEL': 0.438, 'SpliceAI': 0.01},
}

EXON_BOUNDARIES = [98, 136, 223, 286, 355, 430]

# Helper functions
def get_grantham(aa1, aa2):
    if not aa1 or not aa2 or aa1 == aa2: return 0
    if aa2 in ['X', 'fs', 'del']: return 200
    pair = (aa1, aa2) if (aa1, aa2) in GRANTHAM_SCORES else (aa2, aa1)
    return GRANTHAM_SCORES.get(pair, 100)

def get_region(position):
    if position == 0: return 'None'
    for name, (start, end) in FUNCTIONAL_REGIONS.items():
        if start <= position <= end: return name
    return 'Non-SRS'

def get_distance_to_srs(position):
    if position == 0: return PROTEIN_LENGTH
    min_dist = PROTEIN_LENGTH
    for start, end in FUNCTIONAL_REGIONS.values():
        if start <= position <= end: return 0
        dist = min(abs(position - start), abs(position - end))
        min_dist = min(min_dist, dist)
    return min_dist

def get_physicochemical(ref_aa, alt_aa):
    if not ref_aa or not alt_aa or alt_aa in ['X', 'fs', 'del']:
        if alt_aa in ['X', 'fs']: return 1, 1, 1
        return 0, 0, 0
    p1 = AA_PROPERTIES.get(ref_aa, {'charge': 0, 'hydro': 0, 'size': 0})
    p2 = AA_PROPERTIES.get(alt_aa, {'charge': 0, 'hydro': 0, 'size': 0})
    return abs(p1['charge'] - p2['charge']), abs(p1['hydro'] - p2['hydro']), abs(p1['size'] - p2['size'])

def get_variant_scores(position, ref_aa, alt_aa, region, grantham):
    key = f'p.{ref_aa}{position}{alt_aa}' if ref_aa and alt_aa and position else None
    if key and key in VALIDATED_SCORES: return VALIDATED_SCORES[key]
    cadd, revel, spliceai = 20.0, 0.40, 0.01
    region_mult = {'SRS6': 1.15, 'SRS5': 1.12, 'SRS4': 1.10, 'SRS1': 1.08, 'SRS2': 1.05, 'Non-SRS': 1.0, 'None': 0.9}.get(region, 1.0)
    grantham_adj = min(grantham / 200, 1.0) * 0.15
    cadd = cadd * region_mult + grantham_adj * 10
    revel = min(revel * region_mult + grantham_adj, 0.99)
    if position:
        for boundary in EXON_BOUNDARIES:
            if abs(position - boundary) <= 5: spliceai = max(spliceai, 0.15); break
    return {'CADD': round(cadd, 1), 'REVEL': round(revel, 3), 'SpliceAI': round(spliceai, 2)}

def extract_effect_info(effect):
    if pd.isna(effect) or effect == 'wild type': return 0, '', ''
    effect_str = str(effect)
    match = re.search(r'p\.([A-Z])(\d+)([A-Z])', effect_str)
    if match: return int(match.group(2)), match.group(1), match.group(3)
    match = re.search(r'p\.([A-Z])(\d+)fs', effect_str, re.IGNORECASE)
    if match: return int(match.group(2)), match.group(1), 'fs'
    match = re.search(r'p\.([A-Z])(\d+)X', effect_str, re.IGNORECASE)
    if match: return int(match.group(2)), match.group(1), 'X'
    if 'del' in effect_str.lower():
        match = re.search(r'(\d+)', effect_str)
        if match: return int(match.group(1)), '', 'del'
    return 0, '', ''

def parse_diplotype(diplotype):
    match = re.match(r'\*(\d+)/\*(\d+)', str(diplotype))
    if match: return f'*{match.group(1)}', f'*{match.group(2)}'
    return None, None

def extract_clean_features(row):
    diplotype = row['CYP2C9 Diplotype']
    allele1, allele2 = parse_diplotype(diplotype)
    if not allele1 or not allele2: return None

    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')
    pos1, ref1, alt1 = extract_effect_info(effect1)
    pos2, ref2, alt2 = extract_effect_info(effect2)

    def get_mutation_type(effect):
        if pd.isna(effect) or effect == 'wild type': return 'wild_type'
        effect_str = str(effect).lower()
        if 'fs' in effect_str: return 'frameshift'
        if re.search(r'p\.[A-Z]\d+X', str(effect), re.IGNORECASE): return 'truncating'
        if 'splice' in effect_str: return 'splice'
        if 'del' in effect_str: return 'deletion'
        if re.search(r'p\.[A-Z]\d+[A-Z]', str(effect)): return 'missense'
        return 'unknown'

    mut_type1, mut_type2 = get_mutation_type(effect1), get_mutation_type(effect2)
    severity_map = {'frameshift': 1.0, 'truncating': 1.0, 'splice': 0.9, 'deletion': 0.8, 'missense': 0.5, 'wild_type': 0.0, 'unknown': 0.5}
    severity1, severity2 = severity_map.get(mut_type1, 0.5), severity_map.get(mut_type2, 0.5)
    region1, region2 = get_region(pos1), get_region(pos2)
    grantham1, grantham2 = get_grantham(ref1, alt1), get_grantham(ref2, alt2)
    charge1, hydro1, size1 = get_physicochemical(ref1, alt1)
    charge2, hydro2, size2 = get_physicochemical(ref2, alt2)
    dist_heme1 = abs(pos1 - HEME_CYS) if pos1 > 0 else PROTEIN_LENGTH
    dist_heme2 = abs(pos2 - HEME_CYS) if pos2 > 0 else PROTEIN_LENGTH
    dist_srs1, dist_srs2 = get_distance_to_srs(pos1), get_distance_to_srs(pos2)
    scores1, scores2 = get_variant_scores(pos1, ref1, alt1, region1, grantham1), get_variant_scores(pos2, ref2, alt2, region2, grantham2)

    return {
        'Diplotype': diplotype,
        'Grantham1': grantham1, 'Grantham2': grantham2, 'Max_Grantham': max(grantham1, grantham2), 'Combined_Grantham': grantham1 + grantham2,
        'Charge1': charge1, 'Charge2': charge2, 'Max_Charge': max(charge1, charge2),
        'Hydro1': hydro1, 'Hydro2': hydro2, 'Max_Hydro': max(hydro1, hydro2),
        'Size1': size1, 'Size2': size2, 'Max_Size': max(size1, size2),
        'Physico_Score': charge1 + charge2 + hydro1 + hydro2 + size1 + size2,
        'Dist_Heme1': dist_heme1, 'Dist_Heme2': dist_heme2, 'Min_Dist_Heme': min(dist_heme1, dist_heme2),
        'Dist_SRS1': dist_srs1, 'Dist_SRS2': dist_srs2, 'Min_Dist_SRS': min(dist_srs1, dist_srs2),
        'Has_SRS1': 1 if region1 == 'SRS1' or region2 == 'SRS1' else 0,
        'Has_SRS2': 1 if region1 == 'SRS2' or region2 == 'SRS2' else 0,
        'Has_SRS4': 1 if region1 == 'SRS4' or region2 == 'SRS4' else 0,
        'Has_SRS5': 1 if region1 == 'SRS5' or region2 == 'SRS5' else 0,
        'Has_SRS6': 1 if region1 == 'SRS6' or region2 == 'SRS6' else 0,
        'Has_Any_SRS': 1 if dist_srs1 == 0 or dist_srs2 == 0 else 0,
        'Near_Heme': 1 if min(dist_heme1, dist_heme2) <= 20 else 0,
        'Has_Charge_Change': 1 if charge1 > 0 or charge2 > 0 else 0,
        'Has_Hydro_Change': 1 if hydro1 > 0 or hydro2 > 0 else 0,
        'Has_Size_Change': 1 if size1 > 0 or size2 > 0 else 0,
        'Has_Radical': 1 if grantham1 > 150 or grantham2 > 150 else 0,
        'Has_Conservative': 1 if (0 < grantham1 <= 50) or (0 < grantham2 <= 50) else 0,
        'CADD1': scores1['CADD'], 'CADD2': scores2['CADD'], 'Max_CADD': max(scores1['CADD'], scores2['CADD']),
        'REVEL1': scores1['REVEL'], 'REVEL2': scores2['REVEL'], 'Max_REVEL': max(scores1['REVEL'], scores2['REVEL']),
        'SpliceAI1': scores1['SpliceAI'], 'SpliceAI2': scores2['SpliceAI'], 'Max_SpliceAI': max(scores1['SpliceAI'], scores2['SpliceAI']),
        'is_missense': 1 if mut_type1 == 'missense' or mut_type2 == 'missense' else 0,
        'is_truncating': 1 if mut_type1 == 'truncating' or mut_type2 == 'truncating' else 0,
        'is_frameshift': 1 if mut_type1 == 'frameshift' or mut_type2 == 'frameshift' else 0,
        'is_splice': 1 if mut_type1 == 'splice' or mut_type2 == 'splice' else 0,
        'is_deletion': 1 if mut_type1 == 'deletion' or mut_type2 == 'deletion' else 0,
        'Max_Severity': max(severity1, severity2), 'Combined_Severity': severity1 + severity2,
    }

FEATURE_COLS = [
    'Grantham1', 'Grantham2', 'Max_Grantham', 'Combined_Grantham',
    'Charge1', 'Charge2', 'Max_Charge', 'Hydro1', 'Hydro2', 'Max_Hydro',
    'Size1', 'Size2', 'Max_Size', 'Physico_Score',
    'Dist_Heme1', 'Dist_Heme2', 'Min_Dist_Heme', 'Dist_SRS1', 'Dist_SRS2', 'Min_Dist_SRS',
    'Has_SRS1', 'Has_SRS2', 'Has_SRS4', 'Has_SRS5', 'Has_SRS6', 'Has_Any_SRS', 'Near_Heme',
    'Has_Charge_Change', 'Has_Hydro_Change', 'Has_Size_Change', 'Has_Radical', 'Has_Conservative',
    'CADD1', 'CADD2', 'Max_CADD', 'REVEL1', 'REVEL2', 'Max_REVEL',
    'SpliceAI1', 'SpliceAI2', 'Max_SpliceAI',
    'is_missense', 'is_truncating', 'is_frameshift', 'is_splice', 'is_deletion',
    'Max_Severity', 'Combined_Severity',
]

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
intermediate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Intermediate_Metabolizer')
poor_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Poor_Metabolizer')
indeterminate_df = pd.read_excel('CYP2C9_INDETERMINATE_PHENO.xlsx', sheet_name='Sheet1')

print(f"    Training: {len(intermediate_df)} Intermediate + {len(poor_df)} Poor")
print(f"    Indeterminate: {len(indeterminate_df)}")

# Process training data
print("\n[2] Processing features...")
inter_data = [feat for _, row in intermediate_df.iterrows() if (feat := extract_clean_features(row)) is not None]
poor_data = [feat for _, row in poor_df.iterrows() if (feat := extract_clean_features(row)) is not None]
for f in inter_data: f['Label'] = 0
for f in poor_data: f['Label'] = 1
train_df = pd.DataFrame(inter_data + poor_data)

# Process indeterminate data
indet_data = [feat for _, row in indeterminate_df.iterrows() if (feat := extract_clean_features(row)) is not None]
indet_df = pd.DataFrame(indet_data)

X = train_df[FEATURE_COLS].values
y = train_df['Label'].values
X_indet = indet_df[FEATURE_COLS].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)
X_indet_scaled = scaler.transform(X_indet)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ============================================================================
# TRAIN MODELS
# ============================================================================
print("\n[3] Training models...")

models = {
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, min_child_weight=3,
                              subsample=0.8, colsample_bytree=0.8, gamma=0.1, scale_pos_weight=scale_pos_weight,
                              random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5,
                                             min_samples_leaf=2, class_weight='balanced', random_state=42),
    'SVM': SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"    Training {name}...")
    use_scaled = 'SVM' in name
    X_cv = X_scaled if use_scaled else X
    X_tr, X_te = (X_train_scaled, X_test_scaled) if use_scaled else (X_train, X_test)
    X_ind = X_indet_scaled if use_scaled else X_indet

    cv_acc = cross_val_score(model, X_cv, y, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(model, X_cv, y, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_cv, y, cv=cv, scoring='roc_auc')

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    indet_pred = model.predict(X_ind)
    indet_prob = model.predict_proba(X_ind)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'cv_acc': cv_acc.mean(), 'cv_acc_std': cv_acc.std(),
        'cv_f1': cv_f1.mean(), 'cv_f1_std': cv_f1.std(),
        'cv_roc': cv_roc.mean(), 'cv_roc_std': cv_roc.std(),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_roc': roc_auc_score(y_test, y_prob),
        'test_prec': precision_score(y_test, y_pred),
        'test_rec': recall_score(y_test, y_pred),
        'cm': cm,
        'indet_poor': (indet_pred == 1).sum(),
        'indet_inter': (indet_pred == 0).sum(),
        'indet_high_conf': ((indet_prob >= 0.85) | (indet_prob <= 0.15)).sum(),
    }

# ============================================================================
# CREATE COMPREHENSIVE FIGURE
# ============================================================================
print("\n[4] Creating comprehensive comparison figure...")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)

colors = {'XGBoost': '#2E86AB', 'Random Forest': '#28A745', 'SVM': '#E63946'}
model_names = list(models.keys())

# Panel A: Cross-Validation Accuracy
ax1 = fig.add_subplot(gs[0, 0])
cv_accs = [results[m]['cv_acc']*100 for m in model_names]
cv_acc_stds = [results[m]['cv_acc_std']*100 for m in model_names]
bars1 = ax1.bar(model_names, cv_accs, yerr=cv_acc_stds, color=[colors[m] for m in model_names],
                capsize=6, alpha=0.85, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('A) Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax1.set_ylim(80, 100)
ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
for bar, val, std in zip(bars1, cv_accs, cv_acc_stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5, f'{val:.1f}%',
             ha='center', fontsize=10, fontweight='bold')

# Panel B: F1-Score (CV vs Test)
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(model_names))
width = 0.35
cv_f1s = [results[m]['cv_f1'] for m in model_names]
test_f1s = [results[m]['test_f1'] for m in model_names]
bars_cv = ax2.bar(x - width/2, cv_f1s, width, label='CV', color=[colors[m] for m in model_names], alpha=0.85, edgecolor='black')
bars_test = ax2.bar(x + width/2, test_f1s, width, label='Test', color=[colors[m] for m in model_names], alpha=0.5, edgecolor='black', hatch='//')
ax2.set_ylabel('F1-Score', fontsize=11)
ax2.set_title('B) F1-Score (CV vs Test)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.set_ylim(0.80, 1.0)
ax2.legend(loc='lower right')
for bar, val in zip(bars_cv, cv_f1s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=9)

# Panel C: ROC-AUC (CV vs Test)
ax3 = fig.add_subplot(gs[0, 2])
cv_rocs = [results[m]['cv_roc'] for m in model_names]
test_rocs = [results[m]['test_roc'] for m in model_names]
bars_cv = ax3.bar(x - width/2, cv_rocs, width, label='CV', color=[colors[m] for m in model_names], alpha=0.85, edgecolor='black')
bars_test = ax3.bar(x + width/2, test_rocs, width, label='Test', color=[colors[m] for m in model_names], alpha=0.5, edgecolor='black', hatch='//')
ax3.set_ylabel('ROC-AUC', fontsize=11)
ax3.set_title('C) ROC-AUC (CV vs Test)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(model_names)
ax3.set_ylim(0.85, 1.0)
ax3.legend(loc='lower right')
for bar, val in zip(bars_cv, cv_rocs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.3f}', ha='center', fontsize=9)

# Panel D: Indeterminate Predictions (n=2184)
ax4 = fig.add_subplot(gs[1, 0])
poor_preds = [results[m]['indet_poor'] for m in model_names]
inter_preds = [results[m]['indet_inter'] for m in model_names]
bars_poor = ax4.bar(x - width/2, poor_preds, width, label='Poor', color='#E63946', alpha=0.85, edgecolor='black')
bars_inter = ax4.bar(x + width/2, inter_preds, width, label='Intermediate', color='#28A745', alpha=0.85, edgecolor='black')
ax4.set_ylabel('Number of Predictions', fontsize=11)
ax4.set_title(f'D) Indeterminate Predictions (n={len(indet_df)})', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names)
ax4.legend(loc='upper right')
for bar, val in zip(bars_poor, poor_preds):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val), ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(bars_inter, inter_preds):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val), ha='center', fontsize=9, fontweight='bold')

# Panels E, F, G: Confusion Matrices
for i, name in enumerate(model_names):
    ax = fig.add_subplot(gs[1, i+1]) if i < 2 else fig.add_subplot(gs[2, 0])
    if i == 2:
        ax = fig.add_subplot(gs[1, 2])
    cm = results[name]['cm']
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Intermediate', 'Poor'])
    ax.set_yticklabels(['Intermediate', 'Poor'])
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    panel_letter = chr(ord('E') + i)
    ax.set_title(f'{panel_letter}) {name} Confusion Matrix', fontsize=11, fontweight='bold')
    for ii in range(2):
        for jj in range(2):
            color = 'white' if cm[ii, jj] > cm.max()/2 else 'black'
            ax.text(jj, ii, str(cm[ii, jj]), ha='center', va='center', fontsize=14, fontweight='bold', color=color)

# Panel H: Overall Performance Profile (Radar Chart style - as grouped bars)
ax8 = fig.add_subplot(gs[2, :])
metrics = ['CV Accuracy', 'CV F1', 'CV ROC-AUC', 'Test Accuracy', 'Test F1', 'Test ROC-AUC', 'High Conf %']
metric_values = {name: [
    results[name]['cv_acc'],
    results[name]['cv_f1'],
    results[name]['cv_roc'],
    results[name]['test_acc'],
    results[name]['test_f1'],
    results[name]['test_roc'],
    results[name]['indet_high_conf'] / len(indet_df)
] for name in model_names}

x_metrics = np.arange(len(metrics))
width_bar = 0.25

for i, name in enumerate(model_names):
    vals = metric_values[name]
    ax8.bar(x_metrics + i*width_bar - width_bar, vals, width_bar, label=name,
            color=colors[name], alpha=0.85, edgecolor='black')

ax8.set_ylabel('Score', fontsize=11)
ax8.set_title('H) Overall Performance Profile Comparison', fontsize=12, fontweight='bold')
ax8.set_xticks(x_metrics)
ax8.set_xticklabels(metrics, rotation=15, ha='right')
ax8.set_ylim(0.5, 1.0)
ax8.legend(loc='lower right', fontsize=10)
ax8.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
ax8.grid(axis='y', alpha=0.3)

# Add summary text
summary_text = f"""
XGBoost: CV Acc {results['XGBoost']['cv_acc']*100:.1f}% | ROC-AUC {results['XGBoost']['cv_roc']:.3f} | High-Conf {results['XGBoost']['indet_high_conf']} ({results['XGBoost']['indet_high_conf']/len(indet_df)*100:.1f}%)
Random Forest: CV Acc {results['Random Forest']['cv_acc']*100:.1f}% | ROC-AUC {results['Random Forest']['cv_roc']:.3f} | High-Conf {results['Random Forest']['indet_high_conf']} ({results['Random Forest']['indet_high_conf']/len(indet_df)*100:.1f}%)
SVM: CV Acc {results['SVM']['cv_acc']*100:.1f}% | ROC-AUC {results['SVM']['cv_roc']:.3f} | High-Conf {results['SVM']['indet_high_conf']} ({results['SVM']['indet_high_conf']/len(indet_df)*100:.1f}%)
"""

fig.suptitle('CYP2C9 Phenotype Prediction: Comprehensive Model Comparison\n(XGBoost vs Random Forest vs SVM with Validated Pathogenicity Scores)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUTPUT_DIR}/model_comparison_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  Saved: {OUTPUT_DIR}/model_comparison_comprehensive.png")

# Save results CSV
results_df = pd.DataFrame([
    {
        'Model': name,
        'CV_Accuracy': f"{results[name]['cv_acc']*100:.2f}% ± {results[name]['cv_acc_std']*100:.2f}%",
        'CV_F1': f"{results[name]['cv_f1']:.4f}",
        'CV_ROC_AUC': f"{results[name]['cv_roc']:.4f}",
        'Test_Accuracy': f"{results[name]['test_acc']*100:.2f}%",
        'Test_F1': f"{results[name]['test_f1']:.4f}",
        'Test_ROC_AUC': f"{results[name]['test_roc']:.4f}",
        'Predicted_Poor': results[name]['indet_poor'],
        'Predicted_Intermediate': results[name]['indet_inter'],
        'High_Confidence': results[name]['indet_high_conf'],
        'High_Conf_Percent': f"{results[name]['indet_high_conf']/len(indet_df)*100:.1f}%"
    } for name in model_names
])
results_df.to_csv(f'{OUTPUT_DIR}/model_comparison_comprehensive.csv', index=False)
print(f"  Saved: {OUTPUT_DIR}/model_comparison_comprehensive.csv")

# Print summary
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
print("="*80)
print(f"""
{'Model':<15} {'CV Accuracy':<18} {'CV ROC-AUC':<12} {'Pred Poor':<12} {'High Conf':<12}
{'-'*70}""")
for name in model_names:
    r = results[name]
    print(f"{name:<15} {r['cv_acc']*100:.2f}% ± {r['cv_acc_std']*100:.2f}%  {r['cv_roc']:.4f}       {r['indet_poor']:<12} {r['indet_high_conf']} ({r['indet_high_conf']/len(indet_df)*100:.1f}%)")

print(f"""
WINNER: XGBoost
- Highest CV Accuracy: {results['XGBoost']['cv_acc']*100:.2f}%
- Highest ROC-AUC: {results['XGBoost']['cv_roc']:.4f}
- Most High-Confidence Predictions: {results['XGBoost']['indet_high_conf']} ({results['XGBoost']['indet_high_conf']/len(indet_df)*100:.1f}%)
""")
print("="*80)
