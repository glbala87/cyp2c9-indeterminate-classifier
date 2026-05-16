"""
Extract Clinically-Grounded Features for CYP2C9 Phenotype Prediction
Uses PharmGKB, CPIC, and variant annotation data
"""

import pandas as pd
import numpy as np
import re

print("="*80)
print("EXTRACTING CLINICALLY-GROUNDED FEATURES FOR CYP2C9")
print("="*80)

# ============================================================================
# STEP 1: Load Reference Data
# ============================================================================
print("\n[1] Loading reference data...")

# Load variant annotations from PharmGKB
var_annot = pd.read_csv('variant_annotation.tsv', sep='\t')
print(f"   - variant_annotation.tsv: {len(var_annot)} entries")

# Load clinical PGx summary
clin_pgx = pd.read_csv('clinpgx-summary-annotations.tsv', sep='\t')
print(f"   - clinpgx-summary-annotations.tsv: {len(clin_pgx)} entries")

# Load training data
intermediate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Intermediate_Metabolizer')
poor_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Poor_Metabolizer')
indeterminate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Indeterminate')

print(f"   - Intermediate Metabolizers: {len(intermediate_df)}")
print(f"   - Poor Metabolizers: {len(poor_df)}")
print(f"   - Indeterminate: {len(indeterminate_df)}")

# ============================================================================
# STEP 2: Build Allele Function Database from Clinical Evidence
# ============================================================================
print("\n[2] Building allele function database from clinical evidence...")

# CPIC/PharmGKB Validated Allele Functions
# Source: https://www.pharmgkb.org/page/cyp2c9RefMaterials
ALLELE_FUNCTION = {
    # Normal Function (Activity Score = 1)
    '*1': {'function': 'Normal', 'activity_score': 1.0, 'evidence': 'Reference'},
    '*9': {'function': 'Normal', 'activity_score': 1.0, 'evidence': 'CPIC'},

    # Decreased Function (Activity Score = 0.5)
    '*2': {'function': 'Decreased', 'activity_score': 0.5, 'evidence': 'CPIC Level A'},
    '*8': {'function': 'Decreased', 'activity_score': 0.5, 'evidence': 'CPIC Level A'},
    '*11': {'function': 'Decreased', 'activity_score': 0.5, 'evidence': 'CPIC Level A'},

    # No Function (Activity Score = 0)
    '*3': {'function': 'No Function', 'activity_score': 0.0, 'evidence': 'CPIC Level A'},
    '*4': {'function': 'No Function', 'activity_score': 0.0, 'evidence': 'CPIC'},
    '*5': {'function': 'No Function', 'activity_score': 0.0, 'evidence': 'CPIC Level A'},
    '*6': {'function': 'No Function', 'activity_score': 0.0, 'evidence': 'CPIC Level A'},
    '*13': {'function': 'No Function', 'activity_score': 0.0, 'evidence': 'Literature'},
    '*15': {'function': 'No Function', 'activity_score': 0.0, 'evidence': 'Literature'},

    # Likely Decreased (from variant annotations)
    '*12': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*14': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*16': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*29': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*31': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*33': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*37': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*39': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*42': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*43': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*45': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*50': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*52': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*55': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'Literature'},
    '*59': {'function': 'Likely Decreased', 'activity_score': 0.25, 'evidence': 'PharmGKB'},
}

# Add more alleles from variant annotation (decreased metabolism associations)
print("   - Parsing variant annotations for function evidence...")
decreased_alleles = set()
for _, row in var_annot.iterrows():
    assoc = str(row['Association']).lower()
    variant = str(row['Variant'])

    # Look for decreased metabolism/activity associations
    if 'decreased' in assoc and ('metabolism' in assoc or 'activity' in assoc or 'dose' in assoc):
        # Extract allele numbers
        allele_matches = re.findall(r'\*(\d+)', variant)
        for allele in allele_matches:
            if allele != '1':  # Exclude reference
                decreased_alleles.add(f'*{allele}')

# Add to function database if not already present
for allele in decreased_alleles:
    if allele not in ALLELE_FUNCTION:
        ALLELE_FUNCTION[allele] = {'function': 'Uncertain Decreased', 'activity_score': 0.5, 'evidence': 'PharmGKB Literature'}

print(f"   - Total alleles with function data: {len(ALLELE_FUNCTION)}")

# ============================================================================
# STEP 3: Define Protein Functional Regions
# ============================================================================
print("\n[3] Defining protein functional regions...")

FUNCTIONAL_REGIONS = {
    'SRS1': {'range': (97, 126), 'importance': 'substrate_recognition', 'score': 0.7},
    'SRS2': {'range': (200, 230), 'importance': 'substrate_recognition', 'score': 0.7},
    'SRS3': {'range': (230, 250), 'importance': 'substrate_recognition', 'score': 0.6},
    'SRS4': {'range': (290, 320), 'importance': 'I-helix_catalytic', 'score': 0.8},
    'SRS5': {'range': (359, 390), 'importance': 'substrate_recognition', 'score': 0.9},
    'SRS6': {'range': (430, 490), 'importance': 'heme_binding_critical', 'score': 1.0},
}

def get_region_and_score(position):
    """Get functional region and impact score for a mutation position"""
    if position is None:
        return 'Unknown', 0.3
    for region, info in FUNCTIONAL_REGIONS.items():
        if info['range'][0] <= position <= info['range'][1]:
            return region, info['score']
    return 'Non-critical', 0.3

# ============================================================================
# STEP 4: Define Mutation Type Severity
# ============================================================================
print("\n[4] Defining mutation type severity...")

MUTATION_SEVERITY = {
    'wild type': 0.0,           # No change
    'missense': 0.5,            # Amino acid substitution
    'frameshift': 1.0,          # Reading frame disruption
    'truncation': 1.0,          # Premature stop
    'nonsense': 1.0,            # Premature stop codon
    'splice': 0.9,              # Splicing defect
    'deletion': 0.8,            # Deletion
    'insertion': 0.7,           # Insertion
}

def get_mutation_severity(effect):
    """Determine mutation severity from effect description"""
    if pd.isna(effect) or effect == 'wild type':
        return 0.0
    effect_lower = str(effect).lower()

    # Check for frameshift (most severe)
    if 'fs' in effect_lower or 'frameshift' in effect_lower:
        return 1.0
    # Check for truncation/nonsense
    if 'x' in effect.upper() and 'p.' in effect:  # e.g., p.S162X
        return 1.0
    # Check for deletion
    if 'del' in effect_lower:
        return 0.8
    # Default to missense
    return 0.5

# ============================================================================
# STEP 5: Extract Position from Protein Effect
# ============================================================================
def extract_position(effect):
    """Extract amino acid position from effect string like p.R144C"""
    if pd.isna(effect) or effect == 'wild type':
        return None
    match = re.search(r'[A-Z](\d+)[A-Z]', str(effect))
    if match:
        return int(match.group(1))
    # Try alternative patterns
    match = re.search(r'(\d+)fs', str(effect))
    if match:
        return int(match.group(1))
    return None

# ============================================================================
# STEP 6: Feature Extraction Function
# ============================================================================
print("\n[5] Creating feature extraction function...")

def extract_features(row):
    """Extract clinically-grounded features from a diplotype row"""
    features = {}

    diplotype = row['CYP2C9 Diplotype']
    effect1 = row.get('Effect on protein by allele 1', 'wild type')
    effect2 = row.get('Effect on protein by allele 2', 'wild type')

    # Parse alleles
    match = re.match(r'\*(\d+)/\*(\d+)', str(diplotype))
    if not match:
        return None

    allele1 = f'*{match.group(1)}'
    allele2 = f'*{match.group(2)}'

    # ========== ALLELE-BASED FEATURES (Clinical Evidence) ==========

    # Activity scores from CPIC/PharmGKB
    a1_info = ALLELE_FUNCTION.get(allele1, {'activity_score': 0.5, 'function': 'Unknown'})
    a2_info = ALLELE_FUNCTION.get(allele2, {'activity_score': 0.5, 'function': 'Unknown'})

    features['Allele1_Activity_Score'] = a1_info['activity_score']
    features['Allele2_Activity_Score'] = a2_info['activity_score']
    features['Total_Activity_Score'] = a1_info['activity_score'] + a2_info['activity_score']
    features['Min_Activity_Score'] = min(a1_info['activity_score'], a2_info['activity_score'])

    # Function categories
    features['Allele1_No_Function'] = 1 if a1_info['function'] == 'No Function' else 0
    features['Allele2_No_Function'] = 1 if a2_info['function'] == 'No Function' else 0
    features['Has_No_Function_Allele'] = 1 if features['Allele1_No_Function'] or features['Allele2_No_Function'] else 0

    features['Allele1_Decreased'] = 1 if 'Decreased' in a1_info['function'] else 0
    features['Allele2_Decreased'] = 1 if 'Decreased' in a2_info['function'] else 0
    features['Both_Decreased'] = 1 if features['Allele1_Decreased'] and features['Allele2_Decreased'] else 0

    features['Allele1_Normal'] = 1 if a1_info['function'] == 'Normal' else 0
    features['Allele2_Normal'] = 1 if a2_info['function'] == 'Normal' else 0
    features['Wildtype_Count'] = features['Allele1_Normal'] + features['Allele2_Normal']

    # ========== MUTATION SEVERITY FEATURES ==========

    features['Mutation1_Severity'] = get_mutation_severity(effect1)
    features['Mutation2_Severity'] = get_mutation_severity(effect2)
    features['Max_Mutation_Severity'] = max(features['Mutation1_Severity'], features['Mutation2_Severity'])
    features['Combined_Mutation_Severity'] = features['Mutation1_Severity'] + features['Mutation2_Severity']

    # ========== PROTEIN POSITION FEATURES ==========

    pos1 = extract_position(effect1)
    pos2 = extract_position(effect2)

    region1, score1 = get_region_and_score(pos1)
    region2, score2 = get_region_and_score(pos2)

    features['Position1'] = pos1 if pos1 else 0
    features['Position2'] = pos2 if pos2 else 0
    features['Region1_Score'] = score1
    features['Region2_Score'] = score2
    features['Max_Region_Score'] = max(score1, score2)

    # Critical region flags
    features['Has_SRS5_Mutation'] = 1 if region1 == 'SRS5' or region2 == 'SRS5' else 0
    features['Has_SRS6_Mutation'] = 1 if region1 == 'SRS6' or region2 == 'SRS6' else 0
    features['Has_Heme_Region_Mutation'] = features['Has_SRS6_Mutation']
    features['Has_SRS4_Mutation'] = 1 if region1 == 'SRS4' or region2 == 'SRS4' else 0

    # ========== SPECIFIC ALLELE FLAGS (Clinical Evidence) ==========

    # CPIC Level A alleles
    features['Has_Star2'] = 1 if allele1 == '*2' or allele2 == '*2' else 0
    features['Has_Star3'] = 1 if allele1 == '*3' or allele2 == '*3' else 0
    features['Has_Star5'] = 1 if allele1 == '*5' or allele2 == '*5' else 0
    features['Has_Star6'] = 1 if allele1 == '*6' or allele2 == '*6' else 0
    features['Has_Star8'] = 1 if allele1 == '*8' or allele2 == '*8' else 0
    features['Has_Star11'] = 1 if allele1 == '*11' or allele2 == '*11' else 0

    # ========== COMBINED IMPACT SCORE ==========

    # Weighted combination of all factors
    features['Combined_Impact_Score'] = (
        (1 - features['Total_Activity_Score'] / 2) * 0.4 +  # Activity loss
        features['Max_Mutation_Severity'] * 0.3 +            # Mutation severity
        features['Max_Region_Score'] * 0.3                   # Region importance
    )

    # ========== CPIC Phenotype Prediction Score ==========
    # Based on CPIC activity score thresholds
    # Poor: AS <= 0.5, Intermediate: 0.5 < AS < 1.5, Normal: AS >= 1.5
    features['CPIC_Phenotype_Score'] = features['Total_Activity_Score']

    return features

# ============================================================================
# STEP 7: Apply Feature Extraction to All Data
# ============================================================================
print("\n[6] Extracting features from all datasets...")

def process_dataset(df, label):
    features_list = []
    for _, row in df.iterrows():
        feat = extract_features(row)
        if feat:
            feat['Original_Phenotype'] = label
            feat['Diplotype'] = row['CYP2C9 Diplotype']
            features_list.append(feat)
    return pd.DataFrame(features_list)

intermediate_features = process_dataset(intermediate_df, 'Intermediate')
poor_features = process_dataset(poor_df, 'Poor')
indeterminate_features = process_dataset(indeterminate_df, 'Indeterminate')

print(f"   - Intermediate features: {len(intermediate_features)}")
print(f"   - Poor features: {len(poor_features)}")
print(f"   - Indeterminate features: {len(indeterminate_features)}")

# ============================================================================
# STEP 8: Analyze Feature Distributions
# ============================================================================
print("\n[7] Analyzing feature distributions by phenotype...")

# Combine labeled data
labeled_data = pd.concat([intermediate_features, poor_features])

# Key discriminating features
key_features = [
    'Total_Activity_Score',
    'Min_Activity_Score',
    'Has_No_Function_Allele',
    'Both_Decreased',
    'Combined_Mutation_Severity',
    'Max_Region_Score',
    'Has_SRS5_Mutation',
    'Has_SRS6_Mutation',
    'Combined_Impact_Score',
    'CPIC_Phenotype_Score',
    'Has_Star3',
    'Has_Star2',
]

print("\n" + "="*80)
print("FEATURE COMPARISON: INTERMEDIATE vs POOR")
print("="*80)
print(f"\n{'Feature':<35} {'Intermediate Mean':>18} {'Poor Mean':>18} {'Difference':>12}")
print("-"*80)

for feat in key_features:
    int_mean = intermediate_features[feat].mean()
    poor_mean = poor_features[feat].mean()
    diff = poor_mean - int_mean
    print(f"{feat:<35} {int_mean:>18.3f} {poor_mean:>18.3f} {diff:>+12.3f}")

# ============================================================================
# STEP 9: Save Feature Data
# ============================================================================
print("\n[8] Saving feature data...")

# Save training data
training_data = pd.concat([intermediate_features, poor_features])
training_data.to_csv('clinical_features_training.csv', index=False)
print(f"   - Saved: clinical_features_training.csv ({len(training_data)} samples)")

# Save indeterminate data
indeterminate_features.to_csv('clinical_features_indeterminate.csv', index=False)
print(f"   - Saved: clinical_features_indeterminate.csv ({len(indeterminate_features)} samples)")

# ============================================================================
# STEP 10: Summary of Clinical Features
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDED CLINICAL FEATURES FOR PHENOTYPE PREDICTION")
print("="*80)

recommended_features = """
FEATURE CATEGORIES AND CLINICAL RATIONALE:

1. ACTIVITY SCORE FEATURES (CPIC Evidence Level A)
   - Total_Activity_Score: Sum of allele activity scores (CPIC standard)
   - Min_Activity_Score: Minimum activity (determines worst-case function)
   - CPIC_Phenotype_Score: Direct CPIC scoring system

2. ALLELE FUNCTION FLAGS (Clinical Guidelines)
   - Has_No_Function_Allele: Presence of null allele (*3, *5, *6)
   - Both_Decreased: Both alleles have reduced function
   - Wildtype_Count: Number of normal *1 alleles

3. MUTATION SEVERITY (Functional Impact)
   - Combined_Mutation_Severity: Based on mutation type
   - Max_Mutation_Severity: Most severe mutation present

4. PROTEIN REGION FEATURES (Structural Biology)
   - Has_SRS5_Mutation: Position 359-390 (substrate recognition)
   - Has_SRS6_Mutation: Position 430-490 (heme binding - critical)
   - Max_Region_Score: Functional importance of affected region

5. SPECIFIC ALLELE FLAGS (CPIC Level A Evidence)
   - Has_Star3: *3 allele (I359L) - most studied no-function
   - Has_Star2: *2 allele (R144C) - most studied decreased function
   - Has_Star5, Has_Star6: Other no-function alleles

PHENOTYPE PREDICTION LOGIC:
   - Poor: Total_Activity_Score <= 1.0 OR Has_No_Function_Allele with any decreased
   - Intermediate: 1.0 < Total_Activity_Score < 2.0 OR one decreased allele
   - Normal: Total_Activity_Score = 2.0 (both *1)
"""

print(recommended_features)

# Save feature documentation
with open('clinical_features_documentation.txt', 'w') as f:
    f.write(recommended_features)
print("\nSaved: clinical_features_documentation.txt")

print("\n" + "="*80)
print("FEATURE EXTRACTION COMPLETE")
print("="*80)
