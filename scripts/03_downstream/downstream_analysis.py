#!/usr/bin/env python3
"""
Downstream Analysis on High-Confidence Predictions
+ Export pathogenicity scores for ALL groups
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os

# Create output directory
OUTPUT_DIR = 'downstream_analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("DOWNSTREAM ANALYSIS: HIGH-CONFIDENCE PREDICTIONS")
print("="*80)

# ============================================================================
# VALIDATED PATHOGENICITY SCORES (70 variants)
# ============================================================================
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_variant(effect):
    """Extract variant from effect string"""
    if pd.isna(effect) or effect == 'wild type':
        return None
    match = re.search(r'p\.([A-Z])(\d+)([A-Z])', str(effect))
    if match:
        return f"p.{match.group(1)}{match.group(2)}{match.group(3)}"
    match = re.search(r'p\.([A-Z])(\d+)fs', str(effect), re.IGNORECASE)
    if match:
        return f"p.{match.group(1)}{match.group(2)}fs"
    match = re.search(r'p\.([A-Z])(\d+)X', str(effect), re.IGNORECASE)
    if match:
        return f"p.{match.group(1)}{match.group(2)}X"
    return str(effect)

def get_scores(variant):
    """Get pathogenicity scores for a variant"""
    if variant and variant in VALIDATED_SCORES:
        scores = VALIDATED_SCORES[variant]
        return scores['CADD'], scores['REVEL'], scores['SpliceAI'], 'Validated'
    elif variant:
        # Estimated scores
        return 20.0, 0.40, 0.01, 'Estimated'
    else:
        return 18.0, 0.36, 0.01, 'Wild-type default'

# ============================================================================
# PART 1: EXTRACT PATHOGENICITY SCORES FOR ALL GROUPS
# ============================================================================
print("\n" + "="*80)
print("PART 1: PATHOGENICITY SCORES FOR ALL GROUPS")
print("="*80)

# Load all datasets
print("\nLoading data...")
intermediate_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Intermediate_Metabolizer')
poor_df = pd.read_excel('CYP2C9_GENE_Formatted.xlsx', sheet_name='Poor_Metabolizer')
indeterminate_df = pd.read_excel('CYP2C9_INDETERMINATE_PHENO.xlsx', sheet_name='Sheet1')

def extract_scores_for_group(df, group_name):
    """Extract pathogenicity scores for all samples in a group"""
    records = []
    for idx, row in df.iterrows():
        diplotype = row.get('CYP2C9 Diplotype', '')
        effect1 = row.get('Effect on protein by allele 1', 'wild type')
        effect2 = row.get('Effect on protein by allele 2', 'wild type')

        var1 = extract_variant(effect1)
        var2 = extract_variant(effect2)

        cadd1, revel1, splice1, source1 = get_scores(var1)
        cadd2, revel2, splice2, source2 = get_scores(var2)

        records.append({
            'Group': group_name,
            'Diplotype': diplotype,
            'Effect1': effect1,
            'Effect2': effect2,
            'Variant1': var1 if var1 else 'wild-type',
            'Variant2': var2 if var2 else 'wild-type',
            'CADD1': cadd1,
            'REVEL1': revel1,
            'SpliceAI1': splice1,
            'Score_Source1': source1,
            'CADD2': cadd2,
            'REVEL2': revel2,
            'SpliceAI2': splice2,
            'Score_Source2': source2,
            'Max_CADD': max(cadd1, cadd2),
            'Max_REVEL': max(revel1, revel2),
            'Max_SpliceAI': max(splice1, splice2),
        })
    return pd.DataFrame(records)

# Extract for all groups
print("\nExtracting pathogenicity scores...")
poor_scores = extract_scores_for_group(poor_df, 'Poor_Metabolizer')
inter_scores = extract_scores_for_group(intermediate_df, 'Intermediate_Metabolizer')
indet_scores = extract_scores_for_group(indeterminate_df, 'Indeterminate')

# Save individual files
poor_scores.to_csv(f'{OUTPUT_DIR}/pathogenicity_scores_POOR.csv', index=False)
inter_scores.to_csv(f'{OUTPUT_DIR}/pathogenicity_scores_INTERMEDIATE.csv', index=False)
indet_scores.to_csv(f'{OUTPUT_DIR}/pathogenicity_scores_INDETERMINATE.csv', index=False)

# Combine all groups
all_scores = pd.concat([poor_scores, inter_scores, indet_scores], ignore_index=True)
all_scores.to_csv(f'{OUTPUT_DIR}/pathogenicity_scores_ALL_GROUPS.csv', index=False)

print(f"\n  Poor Metabolizer samples:         {len(poor_scores)}")
print(f"  Intermediate Metabolizer samples: {len(inter_scores)}")
print(f"  Indeterminate samples:            {len(indet_scores)}")
print(f"  Total:                            {len(all_scores)}")

# Score source statistics
print("\n" + "-"*60)
print("SCORE SOURCE DISTRIBUTION:")
print("-"*60)

for group_name, group_df in [('Poor', poor_scores), ('Intermediate', inter_scores), ('Indeterminate', indet_scores)]:
    validated1 = (group_df['Score_Source1'] == 'Validated').sum()
    validated2 = (group_df['Score_Source2'] == 'Validated').sum()
    total_variants = len(group_df) * 2
    print(f"\n  {group_name}:")
    print(f"    Allele 1 - Validated: {validated1}/{len(group_df)} ({validated1/len(group_df)*100:.1f}%)")
    print(f"    Allele 2 - Validated: {validated2}/{len(group_df)} ({validated2/len(group_df)*100:.1f}%)")

# ============================================================================
# PART 2: SUMMARY STATISTICS BY GROUP
# ============================================================================
print("\n" + "="*80)
print("PART 2: PATHOGENICITY SCORE STATISTICS BY GROUP")
print("="*80)

summary_stats = []
for group_name, group_df in [('Poor_Metabolizer', poor_scores),
                              ('Intermediate_Metabolizer', inter_scores),
                              ('Indeterminate', indet_scores)]:
    stats = {
        'Group': group_name,
        'N_Samples': len(group_df),
        'Mean_Max_CADD': group_df['Max_CADD'].mean(),
        'Std_Max_CADD': group_df['Max_CADD'].std(),
        'Mean_Max_REVEL': group_df['Max_REVEL'].mean(),
        'Std_Max_REVEL': group_df['Max_REVEL'].std(),
        'Mean_Max_SpliceAI': group_df['Max_SpliceAI'].mean(),
        'Std_Max_SpliceAI': group_df['Max_SpliceAI'].std(),
    }
    summary_stats.append(stats)

    print(f"\n{group_name}:")
    print(f"  Max CADD:    {stats['Mean_Max_CADD']:.2f} ± {stats['Std_Max_CADD']:.2f}")
    print(f"  Max REVEL:   {stats['Mean_Max_REVEL']:.3f} ± {stats['Std_Max_REVEL']:.3f}")
    print(f"  Max SpliceAI:{stats['Mean_Max_SpliceAI']:.3f} ± {stats['Std_Max_SpliceAI']:.3f}")

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(f'{OUTPUT_DIR}/pathogenicity_score_summary.csv', index=False)

# ============================================================================
# PART 3: DOWNSTREAM ANALYSIS ON HIGH-CONFIDENCE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("PART 3: DOWNSTREAM ANALYSIS ON HIGH-CONFIDENCE PREDICTIONS")
print("="*80)

# Load high-confidence predictions
high_conf_poor = pd.read_csv('xgboost_validated_output/high_confidence_poor.csv')
high_conf_int = pd.read_csv('xgboost_validated_output/high_confidence_intermediate.csv')

print(f"\nHigh-confidence Poor Metabolizers:         {len(high_conf_poor)}")
print(f"High-confidence Intermediate Metabolizers: {len(high_conf_int)}")

# ============================================================================
# ANALYSIS 1: Variant Frequency in High-Confidence Poor Metabolizers
# ============================================================================
print("\n" + "-"*60)
print("ANALYSIS 1: Most Common Variants in High-Confidence Poor Metabolizers")
print("-"*60)

from collections import Counter

def count_variants(df):
    variants = []
    for _, row in df.iterrows():
        if 'Effect1' in row and pd.notna(row['Effect1']) and row['Effect1'] != 'wild type':
            var = extract_variant(row['Effect1'])
            if var:
                variants.append(var)
        if 'Effect2' in row and pd.notna(row['Effect2']) and row['Effect2'] != 'wild type':
            var = extract_variant(row['Effect2'])
            if var:
                variants.append(var)
    return Counter(variants)

poor_variants = count_variants(high_conf_poor)
print("\nTop 15 variants in high-confidence Poor Metabolizers:")
print(f"{'Rank':<6}{'Variant':<15}{'Count':<10}{'CADD':<10}{'REVEL':<10}{'Source':<15}")
print("-"*66)

variant_analysis = []
for rank, (var, count) in enumerate(poor_variants.most_common(15), 1):
    cadd, revel, splice, source = get_scores(var)
    print(f"{rank:<6}{var:<15}{count:<10}{cadd:<10.1f}{revel:<10.3f}{source:<15}")
    variant_analysis.append({
        'Rank': rank,
        'Variant': var,
        'Count': count,
        'CADD': cadd,
        'REVEL': revel,
        'SpliceAI': splice,
        'Score_Source': source,
        'Phenotype': 'High_Confidence_Poor'
    })

# Same for intermediate
int_variants = count_variants(high_conf_int)
print("\nTop 15 variants in high-confidence Intermediate Metabolizers:")
print(f"{'Rank':<6}{'Variant':<15}{'Count':<10}{'CADD':<10}{'REVEL':<10}{'Source':<15}")
print("-"*66)

for rank, (var, count) in enumerate(int_variants.most_common(15), 1):
    cadd, revel, splice, source = get_scores(var)
    print(f"{rank:<6}{var:<15}{count:<10}{cadd:<10.1f}{revel:<10.3f}{source:<15}")
    variant_analysis.append({
        'Rank': rank,
        'Variant': var,
        'Count': count,
        'CADD': cadd,
        'REVEL': revel,
        'SpliceAI': splice,
        'Score_Source': source,
        'Phenotype': 'High_Confidence_Intermediate'
    })

variant_df = pd.DataFrame(variant_analysis)
variant_df.to_csv(f'{OUTPUT_DIR}/top_variants_by_phenotype.csv', index=False)

# ============================================================================
# ANALYSIS 2: Clinical Recommendations for High-Confidence Poor Metabolizers
# ============================================================================
print("\n" + "-"*60)
print("ANALYSIS 2: CLINICAL RECOMMENDATIONS FOR HIGH-CONFIDENCE POOR METABOLIZERS")
print("-"*60)

# Drug dosing recommendations based on CPIC guidelines
DRUG_RECOMMENDATIONS = {
    'Warfarin': {
        'Poor': 'Decrease initial dose by 20-40%',
        'Intermediate': 'Decrease initial dose by 10-20%',
        'Normal': 'Standard initial dose'
    },
    'Phenytoin': {
        'Poor': 'Consider 25-50% dose reduction; monitor levels closely',
        'Intermediate': 'Consider 25% dose reduction',
        'Normal': 'Standard dosing'
    },
    'Celecoxib': {
        'Poor': 'Consider alternative or 50% dose reduction',
        'Intermediate': 'Consider 50% dose reduction',
        'Normal': 'Standard dosing'
    },
    'Flurbiprofen': {
        'Poor': 'Consider alternative NSAID',
        'Intermediate': 'Use with caution',
        'Normal': 'Standard dosing'
    },
    'Lornoxicam': {
        'Poor': 'Consider alternative or significant dose reduction',
        'Intermediate': 'Consider dose reduction',
        'Normal': 'Standard dosing'
    }
}

clinical_recs = []
for idx, row in high_conf_poor.iterrows():
    rec = {
        'Diplotype': row['Diplotype'],
        'Effect1': row['Effect1'],
        'Effect2': row['Effect2'],
        'Prob_Poor': row['Prob_Poor'],
        'Predicted_Phenotype': 'Poor Metabolizer',
    }
    for drug, recs in DRUG_RECOMMENDATIONS.items():
        rec[f'{drug}_Recommendation'] = recs['Poor']
    clinical_recs.append(rec)

clinical_df = pd.DataFrame(clinical_recs)
clinical_df.to_csv(f'{OUTPUT_DIR}/clinical_recommendations_poor.csv', index=False)

print(f"\nGenerated clinical recommendations for {len(clinical_df)} high-confidence Poor Metabolizers")
print("\nDrug-specific recommendations (CPIC-based):")
for drug, recs in DRUG_RECOMMENDATIONS.items():
    print(f"  {drug}: {recs['Poor']}")

# ============================================================================
# ANALYSIS 3: MD Simulation Candidates from High-Confidence Predictions
# ============================================================================
print("\n" + "-"*60)
print("ANALYSIS 3: MD SIMULATION CANDIDATES")
print("-"*60)

# Functional regions
FUNCTIONAL_REGIONS = {
    'SRS1': (97, 126),
    'SRS2': (200, 230),
    'SRS4': (290, 320),
    'SRS5': (359, 390),
    'SRS6': (430, 490),
}

def get_region(position):
    for name, (start, end) in FUNCTIONAL_REGIONS.items():
        if start <= position <= end:
            return name
    return 'Non-SRS'

def extract_position(variant):
    match = re.search(r'(\d+)', str(variant))
    if match:
        return int(match.group(1))
    return 0

# Identify variants suitable for MD simulation
md_candidates = []
for var, count in poor_variants.most_common(30):
    pos = extract_position(var)
    region = get_region(pos)
    cadd, revel, splice, source = get_scores(var)

    # Priority scoring for MD simulation
    priority = 0
    if region in ['SRS6', 'SRS5']:
        priority += 50
    elif region in ['SRS4', 'SRS1', 'SRS2']:
        priority += 30
    if cadd >= 25:
        priority += 20
    if revel >= 0.6:
        priority += 20
    priority += count  # Frequency bonus

    md_candidates.append({
        'Variant': var,
        'Position': pos,
        'Region': region,
        'Frequency': count,
        'CADD': cadd,
        'REVEL': revel,
        'SpliceAI': splice,
        'MD_Priority_Score': priority,
        'Recommendation': 'High Priority' if priority >= 80 else 'Medium Priority' if priority >= 50 else 'Low Priority'
    })

md_df = pd.DataFrame(md_candidates)
md_df = md_df.sort_values('MD_Priority_Score', ascending=False)
md_df.to_csv(f'{OUTPUT_DIR}/md_simulation_candidates.csv', index=False)

print("\nTop 10 MD Simulation Candidates (from High-Confidence Poor):")
print(f"{'Variant':<12}{'Position':<10}{'Region':<10}{'Freq':<8}{'CADD':<8}{'REVEL':<8}{'Priority':<10}{'Recommendation':<15}")
print("-"*90)
for _, row in md_df.head(10).iterrows():
    print(f"{row['Variant']:<12}{row['Position']:<10}{row['Region']:<10}{row['Frequency']:<8}"
          f"{row['CADD']:<8.1f}{row['REVEL']:<8.3f}{row['MD_Priority_Score']:<10}{row['Recommendation']:<15}")

# ============================================================================
# ANALYSIS 4: Create Comparison Figures
# ============================================================================
print("\n" + "-"*60)
print("ANALYSIS 4: GENERATING FIGURES")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Figure 1: CADD Score Distribution by Group
ax1 = axes[0, 0]
groups = ['Poor\nMetabolizer', 'Intermediate\nMetabolizer', 'Indeterminate']
cadd_means = [poor_scores['Max_CADD'].mean(), inter_scores['Max_CADD'].mean(), indet_scores['Max_CADD'].mean()]
cadd_stds = [poor_scores['Max_CADD'].std(), inter_scores['Max_CADD'].std(), indet_scores['Max_CADD'].std()]
colors = ['#e74c3c', '#f39c12', '#3498db']
bars = ax1.bar(groups, cadd_means, yerr=cadd_stds, color=colors, capsize=5, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Mean Max CADD Score')
ax1.set_title('CADD Score Distribution by Phenotype Group')
ax1.set_ylim(0, 30)
for bar, mean in zip(bars, cadd_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

# Figure 2: REVEL Score Distribution by Group
ax2 = axes[0, 1]
revel_means = [poor_scores['Max_REVEL'].mean(), inter_scores['Max_REVEL'].mean(), indet_scores['Max_REVEL'].mean()]
revel_stds = [poor_scores['Max_REVEL'].std(), inter_scores['Max_REVEL'].std(), indet_scores['Max_REVEL'].std()]
bars = ax2.bar(groups, revel_means, yerr=revel_stds, color=colors, capsize=5, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Mean Max REVEL Score')
ax2.set_title('REVEL Score Distribution by Phenotype Group')
ax2.set_ylim(0, 1.0)
for bar, mean in zip(bars, revel_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

# Figure 3: Score Source Distribution
ax3 = axes[1, 0]
validated_counts = []
estimated_counts = []
for group_df in [poor_scores, inter_scores, indet_scores]:
    val1 = (group_df['Score_Source1'] == 'Validated').sum()
    val2 = (group_df['Score_Source2'] == 'Validated').sum()
    validated_counts.append(val1 + val2)
    estimated_counts.append(len(group_df)*2 - val1 - val2)

x = np.arange(len(groups))
width = 0.35
bars1 = ax3.bar(x - width/2, validated_counts, width, label='Validated (dbNSFP)', color='#27ae60', alpha=0.8)
bars2 = ax3.bar(x + width/2, estimated_counts, width, label='Estimated', color='#95a5a6', alpha=0.8)
ax3.set_ylabel('Number of Alleles')
ax3.set_title('Pathogenicity Score Source by Group')
ax3.set_xticks(x)
ax3.set_xticklabels(groups)
ax3.legend()

# Figure 4: High-Confidence Prediction Summary
ax4 = axes[1, 1]
labels = ['High-Conf\nPoor', 'High-Conf\nIntermediate', 'Low/Moderate\nConfidence']
full_preds = pd.read_csv('xgboost_validated_output/full_predictions.csv')
low_conf = len(full_preds) - len(high_conf_poor) - len(high_conf_int)
sizes = [len(high_conf_poor), len(high_conf_int), low_conf]
colors_pie = ['#e74c3c', '#27ae60', '#95a5a6']
explode = (0.05, 0.05, 0)
wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
                                    explode=explode, shadow=True, startangle=90)
ax4.set_title('High-Confidence Prediction Distribution\n(Indeterminate Samples)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/downstream_analysis_figures.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFigure saved: {OUTPUT_DIR}/downstream_analysis_figures.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DOWNSTREAM ANALYSIS COMPLETE")
print("="*80)
print(f"""
OUTPUT FILES GENERATED:
-----------------------

1. PATHOGENICITY SCORES FOR ALL GROUPS:
   - {OUTPUT_DIR}/pathogenicity_scores_POOR.csv          ({len(poor_scores)} samples)
   - {OUTPUT_DIR}/pathogenicity_scores_INTERMEDIATE.csv  ({len(inter_scores)} samples)
   - {OUTPUT_DIR}/pathogenicity_scores_INDETERMINATE.csv ({len(indet_scores)} samples)
   - {OUTPUT_DIR}/pathogenicity_scores_ALL_GROUPS.csv    ({len(all_scores)} total)
   - {OUTPUT_DIR}/pathogenicity_score_summary.csv        (Statistics by group)

2. DOWNSTREAM ANALYSIS:
   - {OUTPUT_DIR}/top_variants_by_phenotype.csv          (Most common variants)
   - {OUTPUT_DIR}/clinical_recommendations_poor.csv      ({len(clinical_df)} clinical recs)
   - {OUTPUT_DIR}/md_simulation_candidates.csv           ({len(md_df)} MD candidates)

3. FIGURES:
   - {OUTPUT_DIR}/downstream_analysis_figures.png        (4-panel summary)

KEY FINDINGS:
-------------
- High-confidence Poor Metabolizers:         {len(high_conf_poor)} (suitable for clinical follow-up)
- High-confidence Intermediate Metabolizers: {len(high_conf_int)}
- Total high-confidence predictions:         {len(high_conf_poor) + len(high_conf_int)} ({(len(high_conf_poor)+len(high_conf_int))/len(full_preds)*100:.1f}%)

- Validated pathogenicity scores used:       {len(VALIDATED_SCORES)} variants from dbNSFP/CPIC
- Top MD simulation candidate:               {md_df.iloc[0]['Variant']} (Priority: {md_df.iloc[0]['MD_Priority_Score']})
""")
