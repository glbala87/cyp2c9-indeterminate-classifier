"""
Annotate Indeterminate Metabolizer DIPLOTYPES
==============================================
Uses CYP2C9_INDETERMINATE_PHENO.xlsx to annotate all 2184 diplotypes
with combined features from BOTH alleles for downstream ML.
"""

import pandas as pd
import numpy as np
import re

print("="*80)
print("ANNOTATING INDETERMINATE METABOLIZER DIPLOTYPES")
print("Using: CYP2C9_INDETERMINATE_PHENO.xlsx")
print("="*80)

# ============================================================================
# LOAD INDETERMINATE DATA
# ============================================================================
print("\n[1] Loading indeterminate data...")

indet_df = pd.read_excel('CYP2C9_INDETERMINATE_PHENO.xlsx', sheet_name='Sheet1')
print(f"    Total diplotypes: {len(indet_df)}")

# ============================================================================
# PROTEIN FUNCTIONAL REGIONS
# ============================================================================
FUNCTIONAL_REGIONS = {
    'SRS1': {'range': (97, 126), 'score': 0.7},
    'SRS2': {'range': (200, 230), 'score': 0.7},
    'SRS3': {'range': (230, 250), 'score': 0.6},
    'SRS4': {'range': (290, 320), 'score': 0.8},
    'SRS5': {'range': (359, 390), 'score': 0.9},
    'SRS6': {'range': (430, 490), 'score': 1.0},
}

HEME_CYS = 436
PROTEIN_LENGTH = 490

# ============================================================================
# AMINO ACID PROPERTIES & GRANTHAM SCORES
# ============================================================================
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1, 'charge': 0, 'size': 1},
    'V': {'hydrophobicity': 1, 'charge': 0, 'size': 2},
    'I': {'hydrophobicity': 1, 'charge': 0, 'size': 3},
    'L': {'hydrophobicity': 1, 'charge': 0, 'size': 3},
    'M': {'hydrophobicity': 1, 'charge': 0, 'size': 3},
    'F': {'hydrophobicity': 1, 'charge': 0, 'size': 3},
    'W': {'hydrophobicity': 1, 'charge': 0, 'size': 3},
    'P': {'hydrophobicity': 1, 'charge': 0, 'size': 1},
    'G': {'hydrophobicity': 0, 'charge': 0, 'size': 1},
    'S': {'hydrophobicity': 0, 'charge': 0, 'size': 1},
    'T': {'hydrophobicity': 0, 'charge': 0, 'size': 2},
    'C': {'hydrophobicity': 0, 'charge': 0, 'size': 1},
    'Y': {'hydrophobicity': 0, 'charge': 0, 'size': 3},
    'N': {'hydrophobicity': 0, 'charge': 0, 'size': 2},
    'Q': {'hydrophobicity': 0, 'charge': 0, 'size': 2},
    'D': {'hydrophobicity': 0, 'charge': -1, 'size': 2},
    'E': {'hydrophobicity': 0, 'charge': -1, 'size': 2},
    'K': {'hydrophobicity': 0, 'charge': 1, 'size': 3},
    'R': {'hydrophobicity': 0, 'charge': 1, 'size': 3},
    'H': {'hydrophobicity': 0, 'charge': 1, 'size': 2},
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
    ('V', 'G'): 109, ('Q', 'L'): 113, ('N', 'I'): 149, ('F', 'S'): 155,
}

def get_grantham(aa1, aa2):
    if not aa1 or not aa2 or aa1 == aa2:
        return 0
    if aa2 in ['X', 'fs', 'f']:
        return 200
    pair = (aa1, aa2) if (aa1, aa2) in GRANTHAM_SCORES else (aa2, aa1)
    return GRANTHAM_SCORES.get(pair, 100)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_diplotype(diplotype):
    match = re.match(r'\*(\d+)/\*(\d+)', str(diplotype))
    if match:
        return f'*{match.group(1)}', f'*{match.group(2)}'
    return None, None

def extract_effect_info(effect):
    """Extract position, ref_aa, alt_aa, mutation_type from effect string"""
    if pd.isna(effect) or effect == 'wild type':
        return 0, '', '', 'wild_type'

    effect_str = str(effect)

    # Missense: p.R144C
    match = re.search(r'p\.([A-Z])(\d+)([A-Z])', effect_str)
    if match:
        return int(match.group(2)), match.group(1), match.group(3), 'missense'

    # Frameshift
    match = re.search(r'(\d+)fs', effect_str, re.IGNORECASE)
    if match:
        return int(match.group(1)), '', 'fs', 'frameshift'

    # Truncation: p.S162X
    match = re.search(r'p\.([A-Z])(\d+)X', effect_str, re.IGNORECASE)
    if match:
        return int(match.group(2)), match.group(1), 'X', 'truncating'

    # Splice
    if 'splice' in effect_str.lower() or 'ivs' in effect_str.lower():
        return 0, '', '', 'splice'

    # Deletion
    if 'del' in effect_str.lower():
        match = re.search(r'(\d+)', effect_str)
        if match:
            return int(match.group(1)), '', 'del', 'deletion'

    return 0, '', '', 'unknown'

def get_region(position):
    if position == 0:
        return 'None', 0.3
    for region, info in FUNCTIONAL_REGIONS.items():
        if info['range'][0] <= position <= info['range'][1]:
            return region, info['score']
    return 'Non-SRS', 0.3

def get_distance_to_srs(position):
    if position == 0:
        return PROTEIN_LENGTH
    min_dist = PROTEIN_LENGTH
    for region, info in FUNCTIONAL_REGIONS.items():
        start, end = info['range']
        if start <= position <= end:
            return 0
        dist = min(abs(position - start), abs(position - end))
        min_dist = min(min_dist, dist)
    return min_dist

def get_physicochemical_changes(ref_aa, alt_aa):
    """Get charge, hydrophobicity, size changes"""
    if not ref_aa or not alt_aa or alt_aa in ['X', 'fs', 'f', 'del']:
        if alt_aa in ['X', 'fs', 'f']:
            return 1, 1, 1, 1  # Severe
        return 0, 0, 0, 0

    ref_props = AA_PROPERTIES.get(ref_aa, {'charge': 0, 'hydrophobicity': 0, 'size': 0})
    alt_props = AA_PROPERTIES.get(alt_aa, {'charge': 0, 'hydrophobicity': 0, 'size': 0})

    charge = abs(ref_props['charge'] - alt_props['charge'])
    hydro = abs(ref_props['hydrophobicity'] - alt_props['hydrophobicity'])
    size = abs(ref_props['size'] - alt_props['size'])
    total = charge + hydro + size

    return charge, hydro, size, total

# ============================================================================
# ANNOTATE EACH DIPLOTYPE
# ============================================================================
print("\n[2] Annotating all diplotypes...")

annotations = []

for idx, row in indet_df.iterrows():
    diplotype = row['CYP2C9 Diplotype']
    allele1, allele2 = parse_diplotype(diplotype)
    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')
    reason = row.get('Reason', '')

    if not allele1 or not allele2:
        continue

    # Extract info for ALLELE 1
    pos1, ref1, alt1, mut_type1 = extract_effect_info(effect1)
    region1, region_score1 = get_region(pos1)
    grantham1 = get_grantham(ref1, alt1)
    charge1, hydro1, size1, physico1 = get_physicochemical_changes(ref1, alt1)
    dist_heme1 = abs(pos1 - HEME_CYS) if pos1 > 0 else PROTEIN_LENGTH
    dist_srs1 = get_distance_to_srs(pos1)

    # Extract info for ALLELE 2
    pos2, ref2, alt2, mut_type2 = extract_effect_info(effect2)
    region2, region_score2 = get_region(pos2)
    grantham2 = get_grantham(ref2, alt2)
    charge2, hydro2, size2, physico2 = get_physicochemical_changes(ref2, alt2)
    dist_heme2 = abs(pos2 - HEME_CYS) if pos2 > 0 else PROTEIN_LENGTH
    dist_srs2 = get_distance_to_srs(pos2)

    # Mutation severity
    severity_map = {'frameshift': 1.0, 'truncating': 1.0, 'splice': 0.9,
                    'deletion': 0.8, 'missense': 0.5, 'wild_type': 0.0, 'unknown': 0.5}
    severity1 = severity_map.get(mut_type1, 0.5)
    severity2 = severity_map.get(mut_type2, 0.5)

    # COMBINED FEATURES (from both alleles)
    annotation = {
        # Identifiers
        'Diplotype': diplotype,
        'Allele1': allele1,
        'Allele2': allele2,
        'Effect1': effect1,
        'Effect2': effect2,
        'Reason': reason,

        # ALLELE 1 Features
        'Position1': pos1,
        'Ref_AA1': ref1,
        'Alt_AA1': alt1,
        'Mutation_Type1': mut_type1,
        'Mutation_Severity1': severity1,
        'Region1': region1,
        'Region_Score1': region_score1,
        'Grantham1': grantham1,
        'Charge_Change1': charge1,
        'Hydro_Change1': hydro1,
        'Size_Change1': size1,
        'Physico_Score1': physico1,
        'Dist_Heme1': dist_heme1,
        'Dist_SRS1': dist_srs1,

        # ALLELE 2 Features
        'Position2': pos2,
        'Ref_AA2': ref2,
        'Alt_AA2': alt2,
        'Mutation_Type2': mut_type2,
        'Mutation_Severity2': severity2,
        'Region2': region2,
        'Region_Score2': region_score2,
        'Grantham2': grantham2,
        'Charge_Change2': charge2,
        'Hydro_Change2': hydro2,
        'Size_Change2': size2,
        'Physico_Score2': physico2,
        'Dist_Heme2': dist_heme2,
        'Dist_SRS2': dist_srs2,

        # COMBINED Features (aggregated from both alleles)
        'Max_Position': max(pos1, pos2),
        'Max_Region_Score': max(region_score1, region_score2),
        'Max_Grantham': max(grantham1, grantham2),
        'Combined_Grantham': grantham1 + grantham2,
        'Max_Severity': max(severity1, severity2),
        'Combined_Severity': severity1 + severity2,
        'Min_Dist_Heme': min(dist_heme1, dist_heme2),
        'Min_Dist_SRS': min(dist_srs1, dist_srs2),
        'Max_Charge_Change': max(charge1, charge2),
        'Max_Hydro_Change': max(hydro1, hydro2),
        'Max_Size_Change': max(size1, size2),
        'Combined_Physico': physico1 + physico2,

        # BINARY FLAGS
        'Has_Wildtype': 1 if allele1 == '*1' or allele2 == '*1' else 0,
        'Both_Wildtype': 1 if allele1 == '*1' and allele2 == '*1' else 0,
        'Has_Missense': 1 if mut_type1 == 'missense' or mut_type2 == 'missense' else 0,
        'Has_Truncating': 1 if mut_type1 == 'truncating' or mut_type2 == 'truncating' else 0,
        'Has_Frameshift': 1 if mut_type1 == 'frameshift' or mut_type2 == 'frameshift' else 0,
        'Has_Splice': 1 if mut_type1 == 'splice' or mut_type2 == 'splice' else 0,
        'Has_Deletion': 1 if mut_type1 == 'deletion' or mut_type2 == 'deletion' else 0,

        # SRS Region flags
        'Has_SRS1': 1 if region1 == 'SRS1' or region2 == 'SRS1' else 0,
        'Has_SRS2': 1 if region1 == 'SRS2' or region2 == 'SRS2' else 0,
        'Has_SRS4': 1 if region1 == 'SRS4' or region2 == 'SRS4' else 0,
        'Has_SRS5': 1 if region1 == 'SRS5' or region2 == 'SRS5' else 0,
        'Has_SRS6': 1 if region1 == 'SRS6' or region2 == 'SRS6' else 0,
        'Has_Any_SRS': 1 if dist_srs1 == 0 or dist_srs2 == 0 else 0,
        'Near_Heme': 1 if min(dist_heme1, dist_heme2) <= 20 else 0,

        # Charge change flags
        'Has_Charge_Change': 1 if charge1 > 0 or charge2 > 0 else 0,
        'Has_Hydro_Change': 1 if hydro1 > 0 or hydro2 > 0 else 0,
        'Has_Size_Change': 1 if size1 > 0 or size2 > 0 else 0,

        # Radical substitution flags
        'Has_Radical': 1 if grantham1 > 150 or grantham2 > 150 else 0,
        'Has_Conservative': 1 if (0 < grantham1 <= 50) or (0 < grantham2 <= 50) else 0,

        # Known allele flags
        'Has_Star2': 1 if '*2' in [allele1, allele2] else 0,
        'Has_Star3': 1 if '*3' in [allele1, allele2] else 0,
        'Has_Star5': 1 if '*5' in [allele1, allele2] else 0,
        'Has_Star6': 1 if '*6' in [allele1, allele2] else 0,
        'Has_Star8': 1 if '*8' in [allele1, allele2] else 0,
        'Has_Star11': 1 if '*11' in [allele1, allele2] else 0,

        # Normalized positions
        'Norm_Position1': pos1 / PROTEIN_LENGTH if pos1 > 0 else 0,
        'Norm_Position2': pos2 / PROTEIN_LENGTH if pos2 > 0 else 0,
        'Max_Norm_Position': max(pos1, pos2) / PROTEIN_LENGTH,

        # Predicted Impact Score (combined)
        'Predicted_Impact': (
            max(region_score1, region_score2) * 0.25 +
            max(severity1, severity2) * 0.25 +
            min(max(grantham1, grantham2) / 200, 1.0) * 0.2 +
            (physico1 + physico2) / 10 * 0.15 +
            (1 - min(dist_heme1, dist_heme2) / PROTEIN_LENGTH) * 0.15
        ),
    }

    annotations.append(annotation)

annot_df = pd.DataFrame(annotations)
print(f"    Annotated {len(annot_df)} diplotypes")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n[3] Summary statistics...")

print(f"\n{'='*70}")
print("DIPLOTYPE ANNOTATION SUMMARY")
print(f"{'='*70}")

print(f"\nTotal diplotypes annotated: {len(annot_df)}")

print(f"\n--- By Has_Wildtype (*1 allele) ---")
print(f"  With *1: {annot_df['Has_Wildtype'].sum()}")
print(f"  Without *1: {len(annot_df) - annot_df['Has_Wildtype'].sum()}")

print(f"\n--- By Mutation Type (either allele) ---")
print(f"  Has Missense: {annot_df['Has_Missense'].sum()}")
print(f"  Has Truncating: {annot_df['Has_Truncating'].sum()}")
print(f"  Has Frameshift: {annot_df['Has_Frameshift'].sum()}")
print(f"  Has Splice: {annot_df['Has_Splice'].sum()}")

print(f"\n--- By SRS Region (either allele) ---")
print(f"  Has SRS1: {annot_df['Has_SRS1'].sum()}")
print(f"  Has SRS2: {annot_df['Has_SRS2'].sum()}")
print(f"  Has SRS4: {annot_df['Has_SRS4'].sum()}")
print(f"  Has SRS5: {annot_df['Has_SRS5'].sum()}")
print(f"  Has SRS6: {annot_df['Has_SRS6'].sum()}")
print(f"  Has Any SRS: {annot_df['Has_Any_SRS'].sum()}")

print(f"\n--- By Grantham Score ---")
print(f"  Has Radical (>150): {annot_df['Has_Radical'].sum()}")
print(f"  Has Conservative (<=50): {annot_df['Has_Conservative'].sum()}")

print(f"\n--- Predicted Impact Score Distribution ---")
print(f"  Min: {annot_df['Predicted_Impact'].min():.3f}")
print(f"  Max: {annot_df['Predicted_Impact'].max():.3f}")
print(f"  Mean: {annot_df['Predicted_Impact'].mean():.3f}")
print(f"  Median: {annot_df['Predicted_Impact'].median():.3f}")

print(f"\n--- Sample Diplotypes ---")
print(f"{'Diplotype':<12} {'Effect1':<20} {'Effect2':<20} {'Impact':<8}")
for _, row in annot_df.head(10).iterrows():
    e1 = str(row['Effect1'])[:20]
    e2 = str(row['Effect2'])[:20]
    print(f"{row['Diplotype']:<12} {e1:<20} {e2:<20} {row['Predicted_Impact']:.3f}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print(f"\n[4] Saving outputs...")

annot_df.to_csv('indeterminate_diplotype_annotations.csv', index=False)
print(f"    Saved: indeterminate_diplotype_annotations.csv ({len(annot_df)} diplotypes)")

# Save feature columns list
feature_cols = [c for c in annot_df.columns if c not in ['Diplotype', 'Allele1', 'Allele2', 'Effect1', 'Effect2', 'Reason',
                                                          'Ref_AA1', 'Alt_AA1', 'Ref_AA2', 'Alt_AA2',
                                                          'Mutation_Type1', 'Mutation_Type2', 'Region1', 'Region2']]
with open('indeterminate_feature_columns.txt', 'w') as f:
    f.write("FEATURE COLUMNS FOR ML MODEL\n")
    f.write("="*50 + "\n\n")
    for col in feature_cols:
        f.write(f"{col}\n")
print(f"    Saved: indeterminate_feature_columns.txt ({len(feature_cols)} features)")

print(f"\n{'='*70}")
print("DIPLOTYPE ANNOTATION COMPLETE")
print(f"{'='*70}")
print(f"\nTotal: {len(annot_df)} diplotypes with {len(feature_cols)} features each")
print("Ready for ML model feature extraction!")
