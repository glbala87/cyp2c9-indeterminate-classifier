"""
Comprehensive Allele Annotation with Phenotype Information
===========================================================
Annotates all CYP2C9 alleles with:
- Known phenotype associations (Normal, Intermediate, Poor)
- Phenotype counts from training data
- Structural features (position, region, mutation type)
- Grantham scores and physicochemical properties

Uses training data to determine allele-phenotype associations.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter

print("="*80)
print("COMPREHENSIVE ALLELE ANNOTATION WITH PHENOTYPE INFORMATION")
print("="*80)

# ============================================================================
# LOAD ALL DATA SOURCES
# ============================================================================
print("\n[1] Loading data sources...")

# Training data with known phenotypes
intermediate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Intermediate_Metabolizer')
poor_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Poor_Metabolizer')

# Indeterminate data to predict
indeterminate_df = pd.read_excel('CYP2C9_INDETERMINATE_PHENO.xlsx', sheet_name='Sheet1')

print(f"    Intermediate Metabolizer samples: {len(intermediate_df)}")
print(f"    Poor Metabolizer samples: {len(poor_df)}")
print(f"    Indeterminate samples: {len(indeterminate_df)}")

# ============================================================================
# PROTEIN FUNCTIONAL REGIONS
# ============================================================================
FUNCTIONAL_REGIONS = {
    'SRS1': {'range': (97, 126), 'score': 0.7, 'desc': 'Substrate Recognition Site 1'},
    'SRS2': {'range': (200, 230), 'score': 0.7, 'desc': 'Substrate Recognition Site 2'},
    'SRS3': {'range': (230, 250), 'score': 0.6, 'desc': 'Substrate Recognition Site 3'},
    'SRS4': {'range': (290, 320), 'score': 0.8, 'desc': 'I-helix (catalytic)'},
    'SRS5': {'range': (359, 390), 'score': 0.9, 'desc': 'Substrate Recognition Site 5'},
    'SRS6': {'range': (430, 490), 'score': 1.0, 'desc': 'Heme binding (critical)'},
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
}

def get_grantham(aa1, aa2):
    if not aa1 or not aa2 or aa1 == aa2:
        return 0
    if aa2 in ['X', 'fs']:
        return 200
    pair = (aa1, aa2) if (aa1, aa2) in GRANTHAM_SCORES else (aa2, aa1)
    return GRANTHAM_SCORES.get(pair, 100)

# ============================================================================
# CPIC/PharmGKB KNOWN ALLELE FUNCTIONS
# ============================================================================
CPIC_ALLELE_FUNCTION = {
    # Normal Function (Activity = 1.0)
    '*1': {'function': 'Normal', 'activity': 1.0, 'evidence': 'CPIC Level A'},
    '*9': {'function': 'Normal', 'activity': 1.0, 'evidence': 'CPIC'},

    # Decreased Function (Activity = 0.5)
    '*2': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'CPIC Level A'},
    '*8': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'CPIC Level A'},
    '*11': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'CPIC Level A'},

    # No Function (Activity = 0.0)
    '*3': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC Level A'},
    '*4': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC'},
    '*5': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC Level A'},
    '*6': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC Level A'},
    '*13': {'function': 'No Function', 'activity': 0.0, 'evidence': 'Literature'},
    '*15': {'function': 'No Function', 'activity': 0.0, 'evidence': 'Literature'},
}

# ============================================================================
# EXTRACT ALLELE-PHENOTYPE ASSOCIATIONS FROM TRAINING DATA
# ============================================================================
print("\n[2] Extracting allele-phenotype associations from training data...")

def parse_diplotype(diplotype):
    match = re.match(r'\*(\d+)/\*(\d+)', str(diplotype))
    if match:
        return f'*{match.group(1)}', f'*{match.group(2)}'
    return None, None

def extract_effect_info(effect):
    """Extract position, ref_aa, alt_aa from effect string"""
    if pd.isna(effect) or effect == 'wild type':
        return 0, '', '', 'wild_type'

    effect_str = str(effect)

    # Missense: p.R144C
    match = re.search(r'p\.([A-Z])(\d+)([A-Z])', effect_str)
    if match:
        return int(match.group(2)), match.group(1), match.group(3), 'missense'

    # Frameshift: p.K273fs
    match = re.search(r'p\.[A-Z]?(\d+)fs', effect_str, re.IGNORECASE)
    if match:
        return int(match.group(1)), '', 'fs', 'frameshift'

    # Truncation: p.S162X
    match = re.search(r'p\.([A-Z])(\d+)X', effect_str, re.IGNORECASE)
    if match:
        return int(match.group(2)), match.group(1), 'X', 'truncating'

    return 0, '', '', 'unknown'

# Count allele occurrences in each phenotype class
allele_phenotype_counts = defaultdict(lambda: {'Normal': 0, 'Intermediate': 0, 'Poor': 0, 'effects': set()})

# Process Intermediate metabolizer data
for _, row in intermediate_df.iterrows():
    allele1, allele2 = parse_diplotype(row['CYP2C9 Diplotype'])
    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')

    if allele1:
        allele_phenotype_counts[allele1]['Intermediate'] += 1
        allele_phenotype_counts[allele1]['effects'].add(str(effect1))
        # *1 with wild type is Normal
        if allele1 == '*1' and (pd.isna(effect1) or effect1 == 'wild type'):
            allele_phenotype_counts[allele1]['Normal'] += 1

    if allele2:
        allele_phenotype_counts[allele2]['Intermediate'] += 1
        allele_phenotype_counts[allele2]['effects'].add(str(effect2))

# Process Poor metabolizer data
for _, row in poor_df.iterrows():
    allele1, allele2 = parse_diplotype(row['CYP2C9 Diplotype'])
    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')

    if allele1:
        allele_phenotype_counts[allele1]['Poor'] += 1
        allele_phenotype_counts[allele1]['effects'].add(str(effect1))

    if allele2:
        allele_phenotype_counts[allele2]['Poor'] += 1
        allele_phenotype_counts[allele2]['effects'].add(str(effect2))

# Process Indeterminate data (to get all alleles)
for _, row in indeterminate_df.iterrows():
    allele1, allele2 = parse_diplotype(row['CYP2C9 Diplotype'])
    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')

    if allele1:
        allele_phenotype_counts[allele1]['effects'].add(str(effect1))
    if allele2:
        allele_phenotype_counts[allele2]['effects'].add(str(effect2))

# ============================================================================
# CREATE COMPREHENSIVE ALLELE ANNOTATION TABLE
# ============================================================================
print("\n[3] Creating comprehensive allele annotation table...")

def get_region(position):
    if position == 0:
        return 'None', 0.3, 'No mutation'
    for region, info in FUNCTIONAL_REGIONS.items():
        if info['range'][0] <= position <= info['range'][1]:
            return region, info['score'], info['desc']
    return 'Non-SRS', 0.3, 'Outside SRS regions'

annotations = []

for allele in sorted(allele_phenotype_counts.keys(), key=lambda x: int(x.replace('*', ''))):
    counts = allele_phenotype_counts[allele]
    effects = list(counts['effects'] - {'nan', 'wild type', ''})
    primary_effect = effects[0] if effects else 'wild type'

    # Extract structural info
    position, ref_aa, alt_aa, mut_type = extract_effect_info(primary_effect)
    region, region_score, region_desc = get_region(position)

    # Grantham score
    grantham = get_grantham(ref_aa, alt_aa)

    # Physicochemical changes
    if ref_aa and alt_aa and alt_aa not in ['X', 'fs']:
        ref_props = AA_PROPERTIES.get(ref_aa, {})
        alt_props = AA_PROPERTIES.get(alt_aa, {})
        charge_change = abs(ref_props.get('charge', 0) - alt_props.get('charge', 0))
        hydro_change = abs(ref_props.get('hydrophobicity', 0) - alt_props.get('hydrophobicity', 0))
        size_change = abs(ref_props.get('size', 0) - alt_props.get('size', 0))
    elif alt_aa in ['X', 'fs']:
        charge_change = hydro_change = size_change = 1  # Severe
    else:
        charge_change = hydro_change = size_change = 0

    # Distance to heme
    distance_to_heme = abs(position - HEME_CYS) if position > 0 else PROTEIN_LENGTH

    # CPIC function if known
    cpic_info = CPIC_ALLELE_FUNCTION.get(allele, {})
    cpic_function = cpic_info.get('function', 'Unknown')
    cpic_activity = cpic_info.get('activity')
    cpic_evidence = cpic_info.get('evidence', 'None')

    # Determine phenotype association
    total_inter = counts['Intermediate']
    total_poor = counts['Poor']
    total_normal = counts['Normal']
    total = total_inter + total_poor + total_normal

    if total > 0:
        poor_ratio = total_poor / total
        inter_ratio = total_inter / total
        normal_ratio = total_normal / total
    else:
        poor_ratio = inter_ratio = normal_ratio = 0

    # Assign predominant phenotype
    if cpic_function != 'Unknown':
        if cpic_function == 'No Function':
            predominant_phenotype = 'Poor'
        elif cpic_function == 'Decreased':
            predominant_phenotype = 'Intermediate'
        else:
            predominant_phenotype = 'Normal'
    elif total_poor > total_inter and total_poor > total_normal:
        predominant_phenotype = 'Poor'
    elif total_inter > total_poor and total_inter > total_normal:
        predominant_phenotype = 'Intermediate'
    elif total_normal > 0:
        predominant_phenotype = 'Normal'
    else:
        predominant_phenotype = 'Unknown'

    # Predicted severity score (0 = Normal, 1 = Poor)
    if cpic_activity is not None:
        severity_score = 1 - cpic_activity
    elif mut_type in ['frameshift', 'truncating']:
        severity_score = 1.0
    elif region in ['SRS5', 'SRS6']:
        severity_score = 0.7 + (grantham / 1000)
    elif region == 'SRS4':
        severity_score = 0.5 + (grantham / 1000)
    elif grantham > 150:
        severity_score = 0.6
    elif grantham > 100:
        severity_score = 0.4
    else:
        severity_score = 0.3

    annotation = {
        'Allele': allele,
        'Protein_Effect': primary_effect,
        'Position': position,
        'Ref_AA': ref_aa,
        'Alt_AA': alt_aa,
        'Mutation_Type': mut_type,
        'Region': region,
        'Region_Score': region_score,
        'Region_Description': region_desc,
        'Grantham_Score': grantham,
        'Charge_Change': charge_change,
        'Hydrophobicity_Change': hydro_change,
        'Size_Change': size_change,
        'Distance_to_Heme': distance_to_heme,
        'CPIC_Function': cpic_function,
        'CPIC_Activity': cpic_activity,
        'CPIC_Evidence': cpic_evidence,
        'Count_Normal': total_normal,
        'Count_Intermediate': total_inter,
        'Count_Poor': total_poor,
        'Total_Count': total,
        'Poor_Ratio': poor_ratio,
        'Intermediate_Ratio': inter_ratio,
        'Normal_Ratio': normal_ratio,
        'Predominant_Phenotype': predominant_phenotype,
        'Severity_Score': severity_score,
    }

    annotations.append(annotation)

annot_df = pd.DataFrame(annotations)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n[4] Summary statistics...")

print(f"\n{'='*70}")
print("ALLELE ANNOTATION SUMMARY")
print(f"{'='*70}")

print(f"\nTotal alleles annotated: {len(annot_df)}")

print(f"\n--- By CPIC Function ---")
for func in ['Normal', 'Decreased', 'No Function', 'Unknown']:
    count = len(annot_df[annot_df['CPIC_Function'] == func])
    print(f"  {func}: {count}")

print(f"\n--- By Predominant Phenotype ---")
for pheno in ['Normal', 'Intermediate', 'Poor', 'Unknown']:
    count = len(annot_df[annot_df['Predominant_Phenotype'] == pheno])
    print(f"  {pheno}: {count}")

print(f"\n--- By Functional Region ---")
for region in annot_df['Region'].value_counts().index:
    count = len(annot_df[annot_df['Region'] == region])
    print(f"  {region}: {count}")

print(f"\n--- By Mutation Type ---")
for mut in annot_df['Mutation_Type'].value_counts().index:
    count = len(annot_df[annot_df['Mutation_Type'] == mut])
    print(f"  {mut}: {count}")

# ============================================================================
# DISPLAY KEY ALLELES
# ============================================================================
print(f"\n{'='*70}")
print("KEY ALLELE ANNOTATIONS")
print(f"{'='*70}")

print("\n--- NORMAL FUNCTION ALLELES ---")
normal_alleles = annot_df[annot_df['Predominant_Phenotype'] == 'Normal']
print(f"{'Allele':<8} {'Effect':<15} {'CPIC':<12} {'Region':<10}")
for _, row in normal_alleles.iterrows():
    print(f"{row['Allele']:<8} {str(row['Protein_Effect'])[:15]:<15} {row['CPIC_Function']:<12} {row['Region']:<10}")

print("\n--- INTERMEDIATE FUNCTION ALLELES (from training data) ---")
inter_alleles = annot_df[(annot_df['Predominant_Phenotype'] == 'Intermediate') & (annot_df['Count_Intermediate'] > 0)]
print(f"{'Allele':<8} {'Effect':<15} {'CPIC':<12} {'Inter_Ct':<10} {'Poor_Ct':<10}")
for _, row in inter_alleles.head(15).iterrows():
    print(f"{row['Allele']:<8} {str(row['Protein_Effect'])[:15]:<15} {row['CPIC_Function']:<12} {row['Count_Intermediate']:<10} {row['Count_Poor']:<10}")

print("\n--- POOR FUNCTION ALLELES (from training data) ---")
poor_alleles = annot_df[(annot_df['Predominant_Phenotype'] == 'Poor') | (annot_df['CPIC_Function'] == 'No Function')]
print(f"{'Allele':<8} {'Effect':<15} {'CPIC':<12} {'Inter_Ct':<10} {'Poor_Ct':<10}")
for _, row in poor_alleles.head(15).iterrows():
    print(f"{row['Allele']:<8} {str(row['Protein_Effect'])[:15]:<15} {row['CPIC_Function']:<12} {row['Count_Intermediate']:<10} {row['Count_Poor']:<10}")

print("\n--- UNKNOWN FUNCTION ALLELES (only in Indeterminate) ---")
unknown_alleles = annot_df[(annot_df['Predominant_Phenotype'] == 'Unknown') & (annot_df['Total_Count'] == 0)]
print(f"{'Allele':<8} {'Effect':<15} {'Region':<10} {'Grantham':<10} {'Severity':<10}")
for _, row in unknown_alleles.head(20).iterrows():
    print(f"{row['Allele']:<8} {str(row['Protein_Effect'])[:15]:<15} {row['Region']:<10} {row['Grantham_Score']:<10} {row['Severity_Score']:.2f}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print(f"\n[5] Saving outputs...")

annot_df.to_csv('allele_phenotype_annotations.csv', index=False)
print(f"    Saved: allele_phenotype_annotations.csv ({len(annot_df)} alleles)")

# Create lookup dictionary for feature extraction
allele_lookup = annot_df.set_index('Allele').to_dict('index')

# Save as pickle for easy loading
import pickle
with open('allele_phenotype_lookup.pkl', 'wb') as f:
    pickle.dump(allele_lookup, f)
print(f"    Saved: allele_phenotype_lookup.pkl")

print(f"\n{'='*70}")
print("ANNOTATION COMPLETE - Ready for feature extraction")
print(f"{'='*70}")
