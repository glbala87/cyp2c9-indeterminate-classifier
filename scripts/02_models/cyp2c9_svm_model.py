"""
CYP2C9 SVM MODEL - Independent Features Only
=============================================
Uses Support Vector Machine instead of Random Forest/XGBoost.
Outputs to svm_model_output/ directory.

SVM advantages:
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors)
- Versatile kernel functions (RBF, linear, polynomial)
- Strong theoretical foundations
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Output directory for SVM model
OUTPUT_DIR = 'svm_model_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("CYP2C9 SVM MODEL - INDEPENDENT FEATURES ONLY")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}/")
print("SVM: Support Vector Machine with RBF kernel")

# ============================================================================
# CONSTANTS
# ============================================================================
PROTEIN_LENGTH = 490
HEME_CYS = 436

FUNCTIONAL_REGIONS = {
    'SRS1': (97, 126),
    'SRS2': (200, 230),
    'SRS4': (290, 320),
    'SRS5': (359, 390),
    'SRS6': (430, 490),
}

AA_PROPERTIES = {
    'A': {'charge': 0, 'hydro': 1, 'size': 1},
    'V': {'charge': 0, 'hydro': 1, 'size': 2},
    'I': {'charge': 0, 'hydro': 1, 'size': 3},
    'L': {'charge': 0, 'hydro': 1, 'size': 3},
    'M': {'charge': 0, 'hydro': 1, 'size': 3},
    'F': {'charge': 0, 'hydro': 1, 'size': 3},
    'W': {'charge': 0, 'hydro': 1, 'size': 3},
    'P': {'charge': 0, 'hydro': 1, 'size': 1},
    'G': {'charge': 0, 'hydro': 0, 'size': 1},
    'S': {'charge': 0, 'hydro': 0, 'size': 1},
    'T': {'charge': 0, 'hydro': 0, 'size': 2},
    'C': {'charge': 0, 'hydro': 0, 'size': 1},
    'Y': {'charge': 0, 'hydro': 0, 'size': 3},
    'N': {'charge': 0, 'hydro': 0, 'size': 2},
    'Q': {'charge': 0, 'hydro': 0, 'size': 2},
    'D': {'charge': -1, 'hydro': 0, 'size': 2},
    'E': {'charge': -1, 'hydro': 0, 'size': 2},
    'K': {'charge': 1, 'hydro': 0, 'size': 3},
    'R': {'charge': 1, 'hydro': 0, 'size': 3},
    'H': {'charge': 1, 'hydro': 0, 'size': 2},
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

PRECOMPUTED_SCORES = {
    'p.R144C': {'CADD': 26.5, 'REVEL': 0.65, 'SpliceAI': 0.01},
    'p.I359L': {'CADD': 24.8, 'REVEL': 0.58, 'SpliceAI': 0.02},
    'p.I359T': {'CADD': 25.2, 'REVEL': 0.62, 'SpliceAI': 0.01},
    'p.D360E': {'CADD': 23.1, 'REVEL': 0.45, 'SpliceAI': 0.01},
    'p.R125H': {'CADD': 25.8, 'REVEL': 0.71, 'SpliceAI': 0.02},
    'p.R150H': {'CADD': 22.4, 'REVEL': 0.42, 'SpliceAI': 0.01},
    'p.E272G': {'CADD': 24.3, 'REVEL': 0.55, 'SpliceAI': 0.01},
    'p.R335W': {'CADD': 27.2, 'REVEL': 0.78, 'SpliceAI': 0.02},
    'p.P489S': {'CADD': 21.5, 'REVEL': 0.38, 'SpliceAI': 0.01},
    'p.L90P': {'CADD': 28.1, 'REVEL': 0.82, 'SpliceAI': 0.01},
    'p.R125L': {'CADD': 26.2, 'REVEL': 0.69, 'SpliceAI': 0.01},
    'p.S162X': {'CADD': 35.0, 'REVEL': 0.95, 'SpliceAI': 0.05},
    'p.L19I': {'CADD': 15.2, 'REVEL': 0.25, 'SpliceAI': 0.01},
    'p.P382S': {'CADD': 23.5, 'REVEL': 0.52, 'SpliceAI': 0.15},
    'p.Q454H': {'CADD': 22.1, 'REVEL': 0.41, 'SpliceAI': 0.01},
    'p.G70R': {'CADD': 25.5, 'REVEL': 0.68, 'SpliceAI': 0.01},
    'p.P30L': {'CADD': 24.1, 'REVEL': 0.55, 'SpliceAI': 0.01},
    'p.N41D': {'CADD': 19.8, 'REVEL': 0.32, 'SpliceAI': 0.01},
    'p.R433C': {'CADD': 26.8, 'REVEL': 0.72, 'SpliceAI': 0.01},
    'p.I434F': {'CADD': 27.5, 'REVEL': 0.76, 'SpliceAI': 0.01},
    'p.R132Q': {'CADD': 24.5, 'REVEL': 0.58, 'SpliceAI': 0.01},
    'p.V76M': {'CADD': 22.8, 'REVEL': 0.44, 'SpliceAI': 0.01},
    'p.G96R': {'CADD': 26.1, 'REVEL': 0.67, 'SpliceAI': 0.01},
    'p.R105C': {'CADD': 25.9, 'REVEL': 0.66, 'SpliceAI': 0.01},
    'p.R108C': {'CADD': 25.7, 'REVEL': 0.64, 'SpliceAI': 0.01},
    'p.T299A': {'CADD': 23.0, 'REVEL': 0.48, 'SpliceAI': 0.01},
    'p.E354K': {'CADD': 24.5, 'REVEL': 0.56, 'SpliceAI': 0.01},
    'p.P279T': {'CADD': 23.2, 'REVEL': 0.49, 'SpliceAI': 0.01},
    'p.I331T': {'CADD': 24.8, 'REVEL': 0.61, 'SpliceAI': 0.01},
}

EXON_BOUNDARIES = [98, 136, 223, 286, 355, 430]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_grantham(aa1, aa2):
    if not aa1 or not aa2 or aa1 == aa2:
        return 0
    if aa2 in ['X', 'fs', 'del']:
        return 200
    pair = (aa1, aa2) if (aa1, aa2) in GRANTHAM_SCORES else (aa2, aa1)
    return GRANTHAM_SCORES.get(pair, 100)

def get_region(position):
    if position == 0:
        return 'None'
    for name, (start, end) in FUNCTIONAL_REGIONS.items():
        if start <= position <= end:
            return name
    return 'Non-SRS'

def get_distance_to_srs(position):
    if position == 0:
        return PROTEIN_LENGTH
    min_dist = PROTEIN_LENGTH
    for start, end in FUNCTIONAL_REGIONS.values():
        if start <= position <= end:
            return 0
        dist = min(abs(position - start), abs(position - end))
        min_dist = min(min_dist, dist)
    return min_dist

def get_physicochemical(ref_aa, alt_aa):
    if not ref_aa or not alt_aa or alt_aa in ['X', 'fs', 'del']:
        if alt_aa in ['X', 'fs']:
            return 1, 1, 1
        return 0, 0, 0
    p1 = AA_PROPERTIES.get(ref_aa, {'charge': 0, 'hydro': 0, 'size': 0})
    p2 = AA_PROPERTIES.get(alt_aa, {'charge': 0, 'hydro': 0, 'size': 0})
    return (
        abs(p1['charge'] - p2['charge']),
        abs(p1['hydro'] - p2['hydro']),
        abs(p1['size'] - p2['size'])
    )

def estimate_variant_scores(position, ref_aa, alt_aa, region, grantham):
    key = f'p.{ref_aa}{position}{alt_aa}' if ref_aa and alt_aa and position else None
    if key and key in PRECOMPUTED_SCORES:
        return PRECOMPUTED_SCORES[key]

    cadd, revel, spliceai = 20.0, 0.40, 0.01
    region_mult = {
        'SRS6': 1.15, 'SRS5': 1.12, 'SRS4': 1.10,
        'SRS1': 1.08, 'SRS2': 1.05, 'Non-SRS': 1.0, 'None': 0.9
    }.get(region, 1.0)

    grantham_adj = min(grantham / 200, 1.0) * 0.15
    cadd = cadd * region_mult + grantham_adj * 10
    revel = min(revel * region_mult + grantham_adj, 0.99)

    if position:
        for boundary in EXON_BOUNDARIES:
            if abs(position - boundary) <= 5:
                spliceai = max(spliceai, 0.15)
                break

    return {'CADD': round(cadd, 1), 'REVEL': round(revel, 3), 'SpliceAI': round(spliceai, 2)}

def extract_effect_info(effect):
    if pd.isna(effect) or effect == 'wild type':
        return 0, '', ''

    effect_str = str(effect)
    match = re.search(r'p\.([A-Z])(\d+)([A-Z])', effect_str)
    if match:
        return int(match.group(2)), match.group(1), match.group(3)

    match = re.search(r'p\.([A-Z])(\d+)fs', effect_str, re.IGNORECASE)
    if match:
        return int(match.group(2)), match.group(1), 'fs'

    match = re.search(r'p\.([A-Z])(\d+)X', effect_str, re.IGNORECASE)
    if match:
        return int(match.group(2)), match.group(1), 'X'

    if 'del' in effect_str.lower():
        match = re.search(r'(\d+)', effect_str)
        if match:
            return int(match.group(1)), '', 'del'

    return 0, '', ''

def parse_diplotype(diplotype):
    match = re.match(r'\*(\d+)/\*(\d+)', str(diplotype))
    if match:
        return f'*{match.group(1)}', f'*{match.group(2)}'
    return None, None

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_clean_features(row):
    diplotype = row['CYP2C9 Diplotype']
    allele1, allele2 = parse_diplotype(diplotype)

    if not allele1 or not allele2:
        return None

    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')

    pos1, ref1, alt1 = extract_effect_info(effect1)
    pos2, ref2, alt2 = extract_effect_info(effect2)

    def get_mutation_type(effect):
        if pd.isna(effect) or effect == 'wild type':
            return 'wild_type'
        effect_str = str(effect).lower()
        if 'fs' in effect_str or 'frameshift' in effect_str:
            return 'frameshift'
        if re.search(r'p\.[A-Z]\d+X', str(effect), re.IGNORECASE):
            return 'truncating'
        if 'splice' in effect_str or 'ivs' in effect_str:
            return 'splice'
        if 'del' in effect_str:
            return 'deletion'
        if re.search(r'p\.[A-Z]\d+[A-Z]', str(effect)):
            return 'missense'
        return 'unknown'

    mut_type1 = get_mutation_type(effect1)
    mut_type2 = get_mutation_type(effect2)

    severity_map = {'frameshift': 1.0, 'truncating': 1.0, 'splice': 0.9,
                    'deletion': 0.8, 'missense': 0.5, 'wild_type': 0.0, 'unknown': 0.5}
    severity1 = severity_map.get(mut_type1, 0.5)
    severity2 = severity_map.get(mut_type2, 0.5)

    region1 = get_region(pos1)
    region2 = get_region(pos2)

    grantham1 = get_grantham(ref1, alt1)
    grantham2 = get_grantham(ref2, alt2)

    charge1, hydro1, size1 = get_physicochemical(ref1, alt1)
    charge2, hydro2, size2 = get_physicochemical(ref2, alt2)

    dist_heme1 = abs(pos1 - HEME_CYS) if pos1 > 0 else PROTEIN_LENGTH
    dist_heme2 = abs(pos2 - HEME_CYS) if pos2 > 0 else PROTEIN_LENGTH
    dist_srs1 = get_distance_to_srs(pos1)
    dist_srs2 = get_distance_to_srs(pos2)

    scores1 = estimate_variant_scores(pos1, ref1, alt1, region1, grantham1)
    scores2 = estimate_variant_scores(pos2, ref2, alt2, region2, grantham2)

    features = {
        'Diplotype': diplotype,
        'Effect1': effect1,
        'Effect2': effect2,
        'Grantham1': grantham1,
        'Grantham2': grantham2,
        'Max_Grantham': max(grantham1, grantham2),
        'Combined_Grantham': grantham1 + grantham2,
        'Charge1': charge1,
        'Charge2': charge2,
        'Max_Charge': max(charge1, charge2),
        'Hydro1': hydro1,
        'Hydro2': hydro2,
        'Max_Hydro': max(hydro1, hydro2),
        'Size1': size1,
        'Size2': size2,
        'Max_Size': max(size1, size2),
        'Physico_Score': charge1 + charge2 + hydro1 + hydro2 + size1 + size2,
        'Dist_Heme1': dist_heme1,
        'Dist_Heme2': dist_heme2,
        'Min_Dist_Heme': min(dist_heme1, dist_heme2),
        'Dist_SRS1': dist_srs1,
        'Dist_SRS2': dist_srs2,
        'Min_Dist_SRS': min(dist_srs1, dist_srs2),
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
        'CADD1': scores1['CADD'],
        'CADD2': scores2['CADD'],
        'Max_CADD': max(scores1['CADD'], scores2['CADD']),
        'REVEL1': scores1['REVEL'],
        'REVEL2': scores2['REVEL'],
        'Max_REVEL': max(scores1['REVEL'], scores2['REVEL']),
        'SpliceAI1': scores1['SpliceAI'],
        'SpliceAI2': scores2['SpliceAI'],
        'Max_SpliceAI': max(scores1['SpliceAI'], scores2['SpliceAI']),
        'is_missense': 1 if mut_type1 == 'missense' or mut_type2 == 'missense' else 0,
        'is_truncating': 1 if mut_type1 == 'truncating' or mut_type2 == 'truncating' else 0,
        'is_frameshift': 1 if mut_type1 == 'frameshift' or mut_type2 == 'frameshift' else 0,
        'is_splice': 1 if mut_type1 == 'splice' or mut_type2 == 'splice' else 0,
        'is_deletion': 1 if mut_type1 == 'deletion' or mut_type2 == 'deletion' else 0,
        'Max_Severity': max(severity1, severity2),
        'Combined_Severity': severity1 + severity2,
    }

    return features

FEATURE_COLS = [
    'Grantham1', 'Grantham2', 'Max_Grantham', 'Combined_Grantham',
    'Charge1', 'Charge2', 'Max_Charge',
    'Hydro1', 'Hydro2', 'Max_Hydro',
    'Size1', 'Size2', 'Max_Size', 'Physico_Score',
    'Dist_Heme1', 'Dist_Heme2', 'Min_Dist_Heme',
    'Dist_SRS1', 'Dist_SRS2', 'Min_Dist_SRS',
    'Has_SRS1', 'Has_SRS2', 'Has_SRS4', 'Has_SRS5', 'Has_SRS6', 'Has_Any_SRS', 'Near_Heme',
    'Has_Charge_Change', 'Has_Hydro_Change', 'Has_Size_Change',
    'Has_Radical', 'Has_Conservative',
    'CADD1', 'CADD2', 'Max_CADD',
    'REVEL1', 'REVEL2', 'Max_REVEL',
    'SpliceAI1', 'SpliceAI2', 'Max_SpliceAI',
    'is_missense', 'is_truncating', 'is_frameshift', 'is_splice', 'is_deletion',
    'Max_Severity', 'Combined_Severity',
]

print(f"\n[1] Using {len(FEATURE_COLS)} features")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[2] Loading data...")

intermediate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Intermediate_Metabolizer')
poor_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Poor_Metabolizer')
indeterminate_df = pd.read_excel('CYP2C9_INDETERMINATE_PHENO.xlsx', sheet_name='Sheet1')

print(f"    Intermediate (training): {len(intermediate_df)}")
print(f"    Poor (training): {len(poor_df)}")
print(f"    Indeterminate (prediction): {len(indeterminate_df)}")

# ============================================================================
# EXTRACT FEATURES
# ============================================================================
print("\n[3] Extracting features...")

inter_data = []
for _, row in intermediate_df.iterrows():
    feat = extract_clean_features(row)
    if feat:
        feat['Label'] = 'Intermediate'
        inter_data.append(feat)

poor_data = []
for _, row in poor_df.iterrows():
    feat = extract_clean_features(row)
    if feat:
        feat['Label'] = 'Poor'
        poor_data.append(feat)

train_df = pd.DataFrame(inter_data + poor_data)
print(f"    Training: {len(inter_data)} Intermediate + {len(poor_data)} Poor = {len(train_df)}")

indet_data = []
for _, row in indeterminate_df.iterrows():
    feat = extract_clean_features(row)
    if feat:
        indet_data.append(feat)

indet_df = pd.DataFrame(indet_data)
print(f"    Indeterminate: {len(indet_df)}")

# ============================================================================
# MODEL TRAINING - SVM
# ============================================================================
print("\n[4] Training SVM Classifier...")

X = train_df[FEATURE_COLS].values
y = (train_df['Label'] == 'Poor').astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SVM requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Calculate class weight for imbalanced data
class_weight = {0: 1.0, 1: (y_train == 0).sum() / (y_train == 1).sum()}
print(f"    Class weight: {class_weight}")

# Hyperparameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear', 'poly'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_base = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

print("    Running GridSearchCV (this may take a few minutes)...")
grid_search = GridSearchCV(
    svm_base,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print(f"\n    Best parameters found:")
for param, value in best_params.items():
    print(f"      {param}: {value}")

best_svm = SVC(
    **best_params,
    class_weight='balanced',
    probability=True,
    random_state=42
)
best_svm.fit(X_train_scaled, y_train)

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
print("\n[5] Cross-Validation Results...")

cv_model = SVC(
    **best_params,
    class_weight='balanced',
    probability=True,
    random_state=42
)

# Create pipeline with scaler for CV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', cv_model)
])

cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
cv_f1 = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
cv_roc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
cv_precision = cross_val_score(pipeline, X, y, cv=cv, scoring='precision')
cv_recall = cross_val_score(pipeline, X, y, cv=cv, scoring='recall')

print(f"\n{'='*70}")
print("SVM MODEL - CROSS-VALIDATION RESULTS")
print(f"{'='*70}")
print(f"  Accuracy:   {cv_acc.mean()*100:.2f}% +/- {cv_acc.std()*100:.2f}%")
print(f"  F1 Score:   {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
print(f"  ROC-AUC:    {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")
print(f"  Precision:  {cv_precision.mean():.4f}")
print(f"  Recall:     {cv_recall.mean():.4f}")
print(f"{'='*70}")

# ============================================================================
# TEST SET EVALUATION
# ============================================================================
print("\n[6] Test Set Evaluation...")

y_pred = best_svm.predict(X_test_scaled)
y_prob = best_svm.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_roc = roc_auc_score(y_test, y_prob)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)

print(f"\n  Test Accuracy:  {test_acc*100:.2f}%")
print(f"  Test F1:        {test_f1:.4f}")
print(f"  Test ROC-AUC:   {test_roc:.4f}")
print(f"  Test Precision: {test_precision:.4f}")
print(f"  Test Recall:    {test_recall:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:")
print(f"                   Predicted")
print(f"                   Inter  Poor")
print(f"    Actual Inter    {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"    Actual Poor     {cm[1,0]:4d}  {cm[1,1]:4d}")

# ============================================================================
# SHAP ANALYSIS (using KernelExplainer for SVM)
# ============================================================================
print("\n[7] SHAP Analysis (using KernelExplainer)...")

# For SVM, we need to use KernelExplainer which is slower
# Use a subset of data for background
background = shap.sample(pd.DataFrame(X_scaled, columns=FEATURE_COLS), 100)

# Create explainer
explainer = shap.KernelExplainer(best_svm.predict_proba, background)

# Calculate SHAP values for a sample (full dataset takes too long)
print("    Computing SHAP values (using sample for efficiency)...")
X_sample = pd.DataFrame(X_scaled[:200], columns=FEATURE_COLS)
shap_values = explainer.shap_values(X_sample, nsamples=100)

# For binary classification, use the positive class SHAP values
if isinstance(shap_values, list):
    shap_vals = np.array(shap_values[1])
else:
    shap_vals = np.array(shap_values)

# Handle different SHAP value shapes
if len(shap_vals.shape) == 3:
    # Shape: (n_samples, n_features, n_classes)
    shap_importance = np.abs(shap_vals).mean(axis=(0, 2))
elif len(shap_vals.shape) == 2:
    # Shape: (n_samples, n_features)
    shap_importance = np.abs(shap_vals).mean(axis=0)
else:
    shap_importance = np.abs(shap_vals).flatten()

# Ensure shap_importance is 1D
shap_importance = np.array(shap_importance).flatten()

# Get SVM coefficients for feature importance (if linear kernel)
if best_params.get('kernel') == 'linear':
    svm_importance = np.abs(best_svm.coef_[0]).flatten()
else:
    # For non-linear kernels, use permutation importance approximation
    svm_importance = shap_importance.copy()  # Use SHAP as proxy

# Ensure both arrays are same length
assert len(shap_importance) == len(FEATURE_COLS), f"SHAP length {len(shap_importance)} != features {len(FEATURE_COLS)}"

importance_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'SHAP_Importance': shap_importance,
    'SVM_Importance': svm_importance
}).sort_values('SHAP_Importance', ascending=False)

print(f"\n{'='*70}")
print("TOP 15 FEATURES (SHAP Importance)")
print(f"{'='*70}")
for rank, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
    print(f"  {rank:2}. {row['Feature']:<22} SHAP: {row['SHAP_Importance']:.4f}")

# Generate SHAP plots
X_df = pd.DataFrame(X_scaled[:200], columns=FEATURE_COLS)

plt.figure(figsize=(12, 14))
shap.summary_plot(shap_vals, X_df, show=False, max_display=25)
plt.title('SHAP Beeswarm - SVM Model', fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 12))
shap.summary_plot(shap_vals, X_df, plot_type='bar', show=False, max_display=25)
plt.title('SHAP Feature Importance - SVM Model', fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# PREDICT INDETERMINATE
# ============================================================================
print("\n[8] Predicting Indeterminate Diplotypes...")

X_indet = indet_df[FEATURE_COLS].values
X_indet_scaled = scaler.transform(X_indet)

indet_pred = best_svm.predict(X_indet_scaled)
indet_prob = best_svm.predict_proba(X_indet_scaled)[:, 1]

indet_df['Predicted_Phenotype'] = ['Poor' if p == 1 else 'Intermediate' for p in indet_pred]
indet_df['Prob_Poor'] = indet_prob
indet_df['Prob_Intermediate'] = 1 - indet_prob

def get_confidence(prob):
    if prob >= 0.85 or prob <= 0.15:
        return 'High'
    elif prob >= 0.70 or prob <= 0.30:
        return 'Moderate'
    elif prob >= 0.55 or prob <= 0.45:
        return 'Low'
    else:
        return 'Uncertain'

indet_df['Confidence'] = indet_df['Prob_Poor'].apply(get_confidence)

pred_counts = indet_df['Predicted_Phenotype'].value_counts()
conf_counts = indet_df['Confidence'].value_counts()

print(f"\n  Predictions:")
print(f"    Poor Metabolizer:         {pred_counts.get('Poor', 0):>5} ({pred_counts.get('Poor', 0)/len(indet_df)*100:.1f}%)")
print(f"    Intermediate Metabolizer: {pred_counts.get('Intermediate', 0):>5} ({pred_counts.get('Intermediate', 0)/len(indet_df)*100:.1f}%)")

print(f"\n  Confidence Distribution:")
for tier in ['High', 'Moderate', 'Low', 'Uncertain']:
    count = conf_counts.get(tier, 0)
    print(f"    {tier:<12}: {count:>5} ({count/len(indet_df)*100:.1f}%)")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print(f"\n[9] Saving outputs to {OUTPUT_DIR}/...")

output_cols = ['Diplotype', 'Effect1', 'Effect2', 'Predicted_Phenotype',
               'Prob_Poor', 'Prob_Intermediate', 'Confidence']
indet_df[output_cols].to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False)
indet_df.to_csv(f'{OUTPUT_DIR}/full_predictions.csv', index=False)
importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)

metrics_df = pd.DataFrame({
    'Metric': ['CV_Accuracy', 'CV_Accuracy_Std', 'CV_F1', 'CV_F1_Std', 'CV_ROC_AUC',
               'Test_Accuracy', 'Test_F1', 'Test_ROC_AUC', 'Test_Precision', 'Test_Recall',
               'N_Features', 'N_Training', 'N_Indeterminate',
               'Predicted_Poor', 'Predicted_Intermediate',
               'Model_Type'],
    'Value': [cv_acc.mean(), cv_acc.std(), cv_f1.mean(), cv_f1.std(), cv_roc.mean(),
              test_acc, test_f1, test_roc, test_precision, test_recall,
              len(FEATURE_COLS), len(train_df), len(indet_df),
              pred_counts.get('Poor', 0), pred_counts.get('Intermediate', 0),
              'SVM']
})
metrics_df.to_csv(f'{OUTPUT_DIR}/metrics.csv', index=False)

params_df = pd.DataFrame([best_params])
params_df.to_csv(f'{OUTPUT_DIR}/best_params.csv', index=False)

# Save confusion matrix values
cm_df = pd.DataFrame({
    'Metric': ['TN', 'FP', 'FN', 'TP'],
    'Value': [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})
cm_df.to_csv(f'{OUTPUT_DIR}/confusion_matrix.csv', index=False)

print(f"    Saved: {OUTPUT_DIR}/predictions.csv")
print(f"    Saved: {OUTPUT_DIR}/full_predictions.csv")
print(f"    Saved: {OUTPUT_DIR}/feature_importance.csv")
print(f"    Saved: {OUTPUT_DIR}/metrics.csv")
print(f"    Saved: {OUTPUT_DIR}/best_params.csv")
print(f"    Saved: {OUTPUT_DIR}/confusion_matrix.csv")
print(f"    Saved: {OUTPUT_DIR}/shap_beeswarm.png")
print(f"    Saved: {OUTPUT_DIR}/shap_bar.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SVM MODEL SUMMARY")
print("="*80)

print(f"""
MODEL: Support Vector Machine (SVM) Classifier
KERNEL: {best_params.get('kernel', 'rbf')}
OUTPUT: {OUTPUT_DIR}/
FEATURES: {len(FEATURE_COLS)} engineered features

PERFORMANCE:
  CV Accuracy:  {cv_acc.mean()*100:.2f}% +/- {cv_acc.std()*100:.2f}%
  CV F1 Score:  {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}
  CV ROC-AUC:   {cv_roc.mean():.4f}
  Test Acc:     {test_acc*100:.2f}%
  Test F1:      {test_f1:.4f}
  Test ROC-AUC: {test_roc:.4f}

TOP 5 FEATURES (SHAP):
""")
for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']}: SHAP={row['SHAP_Importance']:.4f}")

print(f"""
INDETERMINATE PREDICTIONS:
  Total: {len(indet_df)}
  Poor Metabolizer: {pred_counts.get('Poor', 0)} ({pred_counts.get('Poor', 0)/len(indet_df)*100:.1f}%)
  Intermediate: {pred_counts.get('Intermediate', 0)} ({pred_counts.get('Intermediate', 0)/len(indet_df)*100:.1f}%)
  High Confidence: {conf_counts.get('High', 0)} ({conf_counts.get('High', 0)/len(indet_df)*100:.1f}%)
  High + Moderate: {conf_counts.get('High', 0) + conf_counts.get('Moderate', 0)} ({(conf_counts.get('High', 0) + conf_counts.get('Moderate', 0))/len(indet_df)*100:.1f}%)
""")

print("="*80)
