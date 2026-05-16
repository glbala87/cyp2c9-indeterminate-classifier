"""
Annotate Indeterminate Metabolizer Alleles for CYP2C9
======================================================
Creates comprehensive annotations for all unique alleles in the
indeterminate dataset to support ML model feature engineering.

Annotations include:
- Protein position and functional region (SRS1-6)
- Mutation type (missense, truncating, splice, frameshift, deletion)
- Amino acid change properties (hydrophobicity, charge, size)
- Clinical evidence from PharmGKB/CPIC
- Predicted functional impact
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

print("="*80)
print("ANNOTATING INDETERMINATE METABOLIZER ALLELES")
print("="*80)

# ============================================================================
# PROTEIN FUNCTIONAL REGIONS
# ============================================================================
FUNCTIONAL_REGIONS = {
    'SRS1': {'range': (97, 126), 'importance': 'Substrate Recognition Site 1', 'score': 0.7},
    'SRS2': {'range': (200, 230), 'importance': 'Substrate Recognition Site 2', 'score': 0.7},
    'SRS3': {'range': (230, 250), 'importance': 'Substrate Recognition Site 3', 'score': 0.6},
    'SRS4': {'range': (290, 320), 'importance': 'I-helix (catalytic)', 'score': 0.8},
    'SRS5': {'range': (359, 390), 'importance': 'Substrate Recognition Site 5', 'score': 0.9},
    'SRS6': {'range': (430, 490), 'importance': 'Heme binding (critical)', 'score': 1.0},
}

# ============================================================================
# AMINO ACID PROPERTIES
# ============================================================================
AA_PROPERTIES = {
    # Hydrophobic
    'A': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'small', 'polarity': 'nonpolar'},
    'V': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'medium', 'polarity': 'nonpolar'},
    'I': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'large', 'polarity': 'nonpolar'},
    'L': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'large', 'polarity': 'nonpolar'},
    'M': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'large', 'polarity': 'nonpolar'},
    'F': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'large', 'polarity': 'nonpolar'},
    'W': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'large', 'polarity': 'nonpolar'},
    'P': {'hydrophobicity': 'hydrophobic', 'charge': 'neutral', 'size': 'small', 'polarity': 'nonpolar'},
    # Polar uncharged
    'G': {'hydrophobicity': 'hydrophilic', 'charge': 'neutral', 'size': 'small', 'polarity': 'nonpolar'},
    'S': {'hydrophobicity': 'hydrophilic', 'charge': 'neutral', 'size': 'small', 'polarity': 'polar'},
    'T': {'hydrophobicity': 'hydrophilic', 'charge': 'neutral', 'size': 'medium', 'polarity': 'polar'},
    'C': {'hydrophobicity': 'hydrophilic', 'charge': 'neutral', 'size': 'small', 'polarity': 'polar'},
    'Y': {'hydrophobicity': 'hydrophilic', 'charge': 'neutral', 'size': 'large', 'polarity': 'polar'},
    'N': {'hydrophobicity': 'hydrophilic', 'charge': 'neutral', 'size': 'medium', 'polarity': 'polar'},
    'Q': {'hydrophobicity': 'hydrophilic', 'charge': 'neutral', 'size': 'medium', 'polarity': 'polar'},
    # Charged
    'D': {'hydrophobicity': 'hydrophilic', 'charge': 'negative', 'size': 'medium', 'polarity': 'charged'},
    'E': {'hydrophobicity': 'hydrophilic', 'charge': 'negative', 'size': 'medium', 'polarity': 'charged'},
    'K': {'hydrophobicity': 'hydrophilic', 'charge': 'positive', 'size': 'large', 'polarity': 'charged'},
    'R': {'hydrophobicity': 'hydrophilic', 'charge': 'positive', 'size': 'large', 'polarity': 'charged'},
    'H': {'hydrophobicity': 'hydrophilic', 'charge': 'positive', 'size': 'medium', 'polarity': 'charged'},
}

# Grantham scores for amino acid substitution severity (higher = more different)
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
}

def get_grantham_score(aa1, aa2):
    """Get Grantham score for amino acid substitution"""
    if aa1 == aa2:
        return 0
    pair = (aa1, aa2) if (aa1, aa2) in GRANTHAM_SCORES else (aa2, aa1)
    return GRANTHAM_SCORES.get(pair, 100)  # Default to moderate

def classify_substitution(grantham_score):
    """Classify substitution as conservative, moderately conservative, moderately radical, or radical"""
    if grantham_score <= 50:
        return 'conservative'
    elif grantham_score <= 100:
        return 'moderately_conservative'
    elif grantham_score <= 150:
        return 'moderately_radical'
    else:
        return 'radical'

# ============================================================================
# KNOWN ALLELE INFORMATION FROM CPIC/PharmGKB
# ============================================================================
KNOWN_ALLELES = {
    '*1': {'function': 'Normal', 'activity': 1.0, 'evidence': 'Reference', 'effect': 'wild type'},
    '*2': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'CPIC Level A', 'effect': 'p.R144C', 'rs': 'rs1799853'},
    '*3': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC Level A', 'effect': 'p.I359L', 'rs': 'rs1057910'},
    '*4': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC', 'effect': 'p.I359T'},
    '*5': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC Level A', 'effect': 'p.D360E'},
    '*6': {'function': 'No Function', 'activity': 0.0, 'evidence': 'CPIC Level A', 'effect': 'p.273fs'},
    '*7': {'function': 'Unknown', 'activity': None, 'evidence': 'Limited', 'effect': 'p.L19I'},
    '*8': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'CPIC Level A', 'effect': 'p.R150H', 'rs': 'rs7900194'},
    '*9': {'function': 'Normal', 'activity': 1.0, 'evidence': 'CPIC', 'effect': 'p.H251R'},
    '*10': {'function': 'Unknown', 'activity': None, 'evidence': 'Limited', 'effect': 'p.E272G'},
    '*11': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'CPIC Level A', 'effect': 'p.R335W', 'rs': 'rs28371685'},
    '*12': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'Literature', 'effect': 'p.P489S'},
    '*13': {'function': 'No Function', 'activity': 0.0, 'evidence': 'Literature', 'effect': 'p.L90P'},
    '*14': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'Literature', 'effect': 'p.R125H'},
    '*15': {'function': 'No Function', 'activity': 0.0, 'evidence': 'Literature', 'effect': 'p.S162X'},
    '*16': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'Literature', 'effect': 'p.T299A'},
    '*17': {'function': 'Unknown', 'activity': None, 'evidence': 'Limited', 'effect': 'p.P382S'},
    '*18': {'function': 'Unknown', 'activity': None, 'evidence': 'Limited', 'effect': 'p.I359L,p.D397A'},
    '*19': {'function': 'Unknown', 'activity': None, 'evidence': 'Limited', 'effect': 'p.Q454H'},
    '*29': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'CPIC', 'effect': 'p.V279F'},
    '*31': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'Literature', 'effect': 'p.V327I'},
    '*33': {'function': 'Decreased', 'activity': 0.5, 'evidence': 'Literature', 'effect': 'p.T130R'},
    '*59': {'function': 'Decreased', 'activity': 0.25, 'evidence': 'PharmGKB', 'effect': 'p.I434F'},
    '*61': {'function': 'Decreased', 'activity': 0.25, 'evidence': 'Literature', 'effect': 'p.V467F'},
    '*72': {'function': 'Decreased', 'activity': 0.25, 'evidence': 'Literature', 'effect': 'p.R433C'},
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

indeterminate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Indeterminate')
intermediate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Intermediate_Metabolizer')
poor_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Poor_Metabolizer')

print(f"    Indeterminate: {len(indeterminate_df)} diplotypes")
print(f"    Intermediate: {len(intermediate_df)} diplotypes")
print(f"    Poor: {len(poor_df)} diplotypes")

# Load PharmGKB annotations
var_annot = pd.read_csv('variant_annotation.tsv', sep='\t')
clin_annot = pd.read_csv('clinpgx-summary-annotations.tsv', sep='\t')
print(f"    PharmGKB variant annotations: {len(var_annot)}")
print(f"    Clinical annotations: {len(clin_annot)}")

# ============================================================================
# EXTRACT UNIQUE ALLELES FROM INDETERMINATE
# ============================================================================
print("\n[2] Extracting unique alleles from indeterminate samples...")

def parse_diplotype(diplotype):
    """Parse diplotype string to extract allele numbers"""
    match = re.match(r'\*(\d+)/\*(\d+)', str(diplotype))
    if match:
        return f'*{match.group(1)}', f'*{match.group(2)}'
    return None, None

def parse_effect(effect):
    """Parse protein effect string to extract mutations"""
    if pd.isna(effect) or effect == 'wild type':
        return []

    mutations = []
    # Handle multiple mutations separated by comma or semicolon
    parts = re.split(r'[,;]\s*', str(effect))
    for part in parts:
        part = part.strip()
        # Match patterns like p.R144C, p.I359L, p.273fs, p.S162X
        match = re.search(r'p\.([A-Z])(\d+)([A-Z]|fs|X)', part, re.IGNORECASE)
        if match:
            mutations.append({
                'original': part,
                'ref_aa': match.group(1),
                'position': int(match.group(2)),
                'alt_aa': match.group(3)
            })
    return mutations

# Collect all unique alleles
allele_info = defaultdict(lambda: {
    'count': 0,
    'effects': set(),
    'diplotypes': []
})

for _, row in indeterminate_df.iterrows():
    diplotype = row['CYP2C9 Diplotype']
    allele1, allele2 = parse_diplotype(diplotype)
    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')

    if allele1:
        allele_info[allele1]['count'] += 1
        allele_info[allele1]['effects'].add(str(effect1))
        allele_info[allele1]['diplotypes'].append(diplotype)

    if allele2:
        allele_info[allele2]['count'] += 1
        allele_info[allele2]['effects'].add(str(effect2))
        allele_info[allele2]['diplotypes'].append(diplotype)

# Get unique non-*1 alleles (variants)
variant_alleles = {k: v for k, v in allele_info.items() if k != '*1'}
print(f"    Found {len(variant_alleles)} unique variant alleles (excluding *1)")

# ============================================================================
# ANNOTATE EACH ALLELE
# ============================================================================
print("\n[3] Annotating alleles...")

def get_region(position):
    """Get functional region for a position"""
    for region, info in FUNCTIONAL_REGIONS.items():
        if info['range'][0] <= position <= info['range'][1]:
            return region, info['importance'], info['score']
    return 'Non-SRS', 'Outside substrate recognition sites', 0.3

def get_mutation_type(effect_str):
    """Classify mutation type"""
    if pd.isna(effect_str) or effect_str == 'wild type':
        return 'wild_type'

    effect_lower = str(effect_str).lower()

    if 'fs' in effect_lower:
        return 'frameshift'
    if 'x' in effect_lower and re.search(r'\d+x', effect_lower):
        return 'truncating'
    if 'del' in effect_lower:
        return 'deletion'
    if 'ins' in effect_lower:
        return 'insertion'
    if 'splice' in effect_lower or 'ivs' in effect_lower:
        return 'splice'
    if re.search(r'[A-Z]\d+[A-Z]', str(effect_str)):
        return 'missense'

    return 'unknown'

def get_pharmgkb_evidence(allele):
    """Get PharmGKB clinical evidence for an allele"""
    evidence = []

    # Search variant annotations
    for _, row in var_annot.iterrows():
        variant = str(row.get('Variant', ''))
        if allele in variant:
            assoc = str(row.get('Association', ''))
            sig = str(row.get('Significance', ''))
            evidence.append({
                'type': 'variant_annotation',
                'association': assoc[:100] if len(assoc) > 100 else assoc,
                'significance': sig
            })

    # Search clinical annotations
    for _, row in clin_annot.iterrows():
        variant = str(row.get('Variant', ''))
        if allele in variant:
            level = row.get('Level', '')
            phenotype = str(row.get('Phenotype Categories', ''))
            evidence.append({
                'type': 'clinical_annotation',
                'level': level,
                'phenotype_category': phenotype
            })

    return evidence

annotations = []

for allele, info in sorted(variant_alleles.items(), key=lambda x: int(x[0].replace('*', ''))):
    # Get the most common effect
    effects = list(info['effects'])
    primary_effect = effects[0] if effects else 'unknown'

    # Parse the effect
    mutations = parse_effect(primary_effect)

    # Get known info if available
    known = KNOWN_ALLELES.get(allele, {})

    # Determine position and region
    if mutations:
        position = mutations[0]['position']
        region, region_desc, region_score = get_region(position)
        ref_aa = mutations[0]['ref_aa']
        alt_aa = mutations[0]['alt_aa']

        # Get amino acid properties
        if alt_aa not in ['fs', 'X'] and len(alt_aa) == 1:
            grantham = get_grantham_score(ref_aa, alt_aa)
            substitution_class = classify_substitution(grantham)

            ref_props = AA_PROPERTIES.get(ref_aa, {})
            alt_props = AA_PROPERTIES.get(alt_aa, {})

            charge_change = ref_props.get('charge', '') != alt_props.get('charge', '')
            hydrophobicity_change = ref_props.get('hydrophobicity', '') != alt_props.get('hydrophobicity', '')
            size_change = ref_props.get('size', '') != alt_props.get('size', '')
        else:
            grantham = None
            substitution_class = 'severe' if alt_aa in ['fs', 'X'] else 'unknown'
            charge_change = None
            hydrophobicity_change = None
            size_change = None
    else:
        position = 0
        region = 'Unknown'
        region_desc = 'Unknown'
        region_score = 0.3
        ref_aa = ''
        alt_aa = ''
        grantham = None
        substitution_class = 'unknown'
        charge_change = None
        hydrophobicity_change = None
        size_change = None

    # Get mutation type
    mut_type = get_mutation_type(primary_effect)

    # Get PharmGKB evidence
    pharmgkb = get_pharmgkb_evidence(allele)
    has_clinical_evidence = len(pharmgkb) > 0
    evidence_level = None
    for ev in pharmgkb:
        if 'level' in ev and ev['level']:
            evidence_level = ev['level']
            break

    # Determine predicted impact
    if mut_type in ['frameshift', 'truncating']:
        predicted_impact = 'Loss of Function'
        impact_score = 1.0
    elif mut_type == 'splice':
        predicted_impact = 'Likely Severe'
        impact_score = 0.9
    elif region in ['SRS5', 'SRS6']:
        if substitution_class in ['radical', 'moderately_radical']:
            predicted_impact = 'Likely Decreased'
            impact_score = 0.8
        else:
            predicted_impact = 'Possibly Decreased'
            impact_score = 0.6
    elif region == 'SRS4':
        predicted_impact = 'Possibly Decreased'
        impact_score = 0.5
    elif substitution_class == 'radical':
        predicted_impact = 'Possibly Decreased'
        impact_score = 0.5
    else:
        predicted_impact = 'Uncertain'
        impact_score = 0.3

    # Override with known function if available
    if known.get('function'):
        if known['function'] == 'No Function':
            predicted_impact = 'Loss of Function (Known)'
            impact_score = 1.0
        elif known['function'] == 'Decreased':
            predicted_impact = 'Decreased (Known)'
            impact_score = 0.7
        elif known['function'] == 'Normal':
            predicted_impact = 'Normal (Known)'
            impact_score = 0.0

    annotation = {
        'Allele': allele,
        'Count_in_Indeterminate': info['count'],
        'Protein_Effect': primary_effect,
        'Position': position,
        'Region': region,
        'Region_Description': region_desc,
        'Region_Score': region_score,
        'Mutation_Type': mut_type,
        'Ref_AA': ref_aa,
        'Alt_AA': alt_aa,
        'Grantham_Score': grantham,
        'Substitution_Class': substitution_class,
        'Charge_Change': charge_change,
        'Hydrophobicity_Change': hydrophobicity_change,
        'Size_Change': size_change,
        'Known_Function': known.get('function', 'Unknown'),
        'Known_Activity': known.get('activity'),
        'Evidence_Level': known.get('evidence', evidence_level),
        'Has_PharmGKB_Evidence': has_clinical_evidence,
        'Predicted_Impact': predicted_impact,
        'Impact_Score': impact_score,
    }

    annotations.append(annotation)

# Create DataFrame
annot_df = pd.DataFrame(annotations)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n[4] Summary statistics...")

print(f"\n{'='*70}")
print("ALLELE ANNOTATION SUMMARY")
print(f"{'='*70}")

print(f"\nTotal variant alleles annotated: {len(annot_df)}")

print(f"\n--- By Functional Region ---")
region_counts = annot_df['Region'].value_counts()
for region, count in region_counts.items():
    print(f"  {region}: {count}")

print(f"\n--- By Mutation Type ---")
mut_counts = annot_df['Mutation_Type'].value_counts()
for mut, count in mut_counts.items():
    print(f"  {mut}: {count}")

print(f"\n--- By Substitution Class ---")
sub_counts = annot_df['Substitution_Class'].value_counts()
for sub, count in sub_counts.items():
    print(f"  {sub}: {count}")

print(f"\n--- By Predicted Impact ---")
impact_counts = annot_df['Predicted_Impact'].value_counts()
for impact, count in impact_counts.items():
    print(f"  {impact}: {count}")

print(f"\n--- By Known Function ---")
func_counts = annot_df['Known_Function'].value_counts()
for func, count in func_counts.items():
    print(f"  {func}: {count}")

# ============================================================================
# HIGH-PRIORITY ALLELES (for MD simulation / further study)
# ============================================================================
print(f"\n{'='*70}")
print("HIGH-PRIORITY ALLELES FOR FURTHER CHARACTERIZATION")
print(f"{'='*70}")

# Priority: Unknown function + critical region + high frequency
high_priority = annot_df[
    (annot_df['Known_Function'] == 'Unknown') &
    (annot_df['Region'].isin(['SRS5', 'SRS6', 'SRS4'])) &
    (annot_df['Count_in_Indeterminate'] >= 10)
].sort_values('Impact_Score', ascending=False)

print(f"\nAlleles with Unknown function in critical regions (count >= 10):")
print(f"{'Allele':<10} {'Effect':<20} {'Region':<10} {'Count':<8} {'Impact':<25}")
print("-"*70)
for _, row in high_priority.iterrows():
    print(f"{row['Allele']:<10} {row['Protein_Effect'][:20]:<20} {row['Region']:<10} {row['Count_in_Indeterminate']:<8} {row['Predicted_Impact']:<25}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print(f"\n[5] Saving outputs...")

# Full annotation table
annot_df.to_csv('indeterminate_allele_annotations.csv', index=False)
print(f"    Saved: indeterminate_allele_annotations.csv ({len(annot_df)} alleles)")

# Summary by region
region_summary = annot_df.groupby('Region').agg({
    'Allele': 'count',
    'Impact_Score': 'mean',
    'Count_in_Indeterminate': 'sum'
}).rename(columns={'Allele': 'N_Alleles', 'Count_in_Indeterminate': 'Total_Occurrences'})
region_summary.to_csv('indeterminate_region_summary.csv')
print(f"    Saved: indeterminate_region_summary.csv")

# High priority alleles
high_priority.to_csv('indeterminate_high_priority_alleles.csv', index=False)
print(f"    Saved: indeterminate_high_priority_alleles.csv ({len(high_priority)} alleles)")

# ============================================================================
# FEATURE SUGGESTIONS FOR ML
# ============================================================================
print(f"\n{'='*70}")
print("SUGGESTED NEW FEATURES FOR ML MODEL")
print(f"{'='*70}")

print("""
Based on allele annotations, consider adding these features:

1. AMINO ACID CHANGE FEATURES (from Grantham scores):
   - grantham_score_max: Maximum Grantham score of mutations
   - is_radical_substitution: Grantham > 150
   - is_conservative_substitution: Grantham <= 50

2. CHARGE/HYDROPHOBICITY FEATURES:
   - has_charge_change: Change in amino acid charge
   - has_hydrophobicity_change: Change in hydrophobicity
   - has_size_change: Change in amino acid size

3. POSITION-BASED FEATURES:
   - normalized_position: Position / 490 (protein length)
   - distance_to_heme: abs(position - 436) (Cys436 is heme-coordinating)
   - distance_to_active_site: min distance to SRS regions

4. COMBINED IMPACT FEATURES:
   - predicted_impact_score: 0-1 score based on region + mutation type
   - has_known_decreased_allele: Known decreased function allele
   - has_unknown_critical_allele: Unknown function in critical region
""")

print("="*70)
print("ANNOTATION COMPLETE")
print("="*70)
