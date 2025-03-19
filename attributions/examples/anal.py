import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def load_data(filepath):
    """Load and preprocess the annotator comparison data."""
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Convert string values to float where necessary
    for col in df.columns:
        if col not in ['mechanism', 'train_annotator', 'test_annotator']:
            df[col] = pd.to_numeric(df[col])

    return df

def verify_attributions(df):
    """Verify that Shapley values sum to performance differences."""

    # Group by train_annotator and test_annotator
    grouped = df.groupby(['train_annotator', 'test_annotator'])

    results = []
    for (train, test), group in grouped:
        # Check if both mechanisms exist for this pair
        mechanisms = group['mechanism'].unique()
        if len(mechanisms) < 2:
            print(f"Warning: Missing mechanism for train={train}, test={test}")
            continue

        # Sum the Shapley values
        sum_dice = group['value_Dice'].sum()
        sum_f1 = group['value_F1'].sum()

        # Get the performance differences (should be the same for all rows in group)
        perf_diff_dice = group['performance_difference_Dice'].iloc[0]
        perf_diff_f1 = group['performance_difference_F1'].iloc[0]

        # Calculate error
        error_dice = abs(sum_dice - perf_diff_dice)
        error_f1 = abs(sum_f1 - perf_diff_f1)

        results.append({
            'train_annotator': train,
            'test_annotator': test,
            'sum_dice': sum_dice,
            'perf_diff_dice': perf_diff_dice,
            'error_dice': error_dice,
            'sum_f1': sum_f1,
            'perf_diff_f1': perf_diff_f1,
            'error_f1': error_f1
        })

    verification_df = pd.DataFrame(results)

    # Print summary
    print("\n== Verification Summary ==")
    print(f"Max Dice Error: {verification_df['error_dice'].max():.6f}")
    print(f"Max F1 Error: {verification_df['error_f1'].max():.6f}")
    print(f"Mean Dice Error: {verification_df['error_dice'].mean():.6f}")
    print(f"Mean F1 Error: {verification_df['error_f1'].mean():.6f}")

    return verification_df

def analyze_mechanism_contribution(df):
    """Analyze the contribution of each mechanism to performance differences."""

    # Calculate relative contribution (as percentage of total difference)
    # Group by train_annotator, test_annotator, and mechanism
    grouped = df.groupby(['train_annotator', 'test_annotator', 'mechanism'])

    results = []
    for (train, test, mechanism), group in grouped:
        # Get the performance differences
        perf_diff_dice = group['performance_difference_Dice'].iloc[0]
        perf_diff_f1 = group['performance_difference_F1'].iloc[0]

        # Get the Shapley values
        value_dice = group['value_Dice'].iloc[0]
        value_f1 = group['value_F1'].iloc[0]

        # Calculate relative contribution (%)
        relative_contrib_dice = (value_dice / perf_diff_dice) * 100 if perf_diff_dice != 0 else 0
        relative_contrib_f1 = (value_f1 / perf_diff_f1) * 100 if perf_diff_f1 != 0 else 0

        results.append({
            'train_annotator': train,
            'test_annotator': test,
            'mechanism': mechanism,
            'value_dice': value_dice,
            'value_f1': value_f1,
            'relative_contrib_dice': relative_contrib_dice,
            'relative_contrib_f1': relative_contrib_f1
        })

    contribution_df = pd.DataFrame(results)

    # Calculate average contribution by mechanism
    avg_contrib = contribution_df.groupby('mechanism').agg({
        'relative_contrib_dice': 'mean',
        'relative_contrib_f1': 'mean'
    }).reset_index()

    print("\n== Average Mechanism Contribution ==")
    print(avg_contrib)

    return contribution_df, avg_contrib

def plot_heatmap_matrices(df, metric='Dice', output_dir='results'):
    """
    Create heatmap matrices for:
    1. Initial performance in train environment
    2. Initial performance in inference environment
    3. Performance difference
    4. Contribution of P(labels|images)
    5. Contribution of P(images)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique annotators
    annotators = sorted(set(

def plot_mechanism_comparison(contribution_df, output_dir='results'):
    """Plot comparison between P(labels|images) and P(images) contributions."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    plot_data = []
    for _, row in contribution_df.iterrows():
        plot_data.append({
            'train_annotator': row['train_annotator'],
            'test_annotator': row['test_annotator'],
            'mechanism': row['mechanism'],
            'value_dice': row['value_dice'],
            'value_f1': row['value_f1']
        })

    plot_df = pd.DataFrame(plot_data)

    # Pivot the data to have mechanisms as columns
    dice_pivot = plot_df.pivot_table(
        index=['train_annotator', 'test_annotator'],
        columns='mechanism',
        values='value_dice'
    ).reset_index()

    f1_pivot = plot_df.pivot_table(
        index=['train_annotator', 'test_annotator'],
        columns='mechanism',
        values='value_f1'
    ).reset_index()

    # Plot scatter comparison for Dice
    plt.figure(figsize=(10, 8))
    plt.scatter(dice_pivot['P(labels|images)'], dice_pivot['P(images)'],
                alpha=0.7, s=100)

    # Add labels for each point
    for i, row in dice_pivot.iterrows():
        plt.annotate(f"{int(row['train_annotator'])}->{int(row['test_annotator'])}",
                    (row['P(labels|images)'], row['P(images)']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')

    # Add diagonal line
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

    plt.title('Comparison of Mechanism Contributions (Dice)')
    plt.xlabel('P(labels|images) contribution')
    plt.ylabel('P(images) contribution')
    plt.grid(True, alpha=0.3)

    # Add text box showing average contribution percentages
    avg_data = plot_df.groupby('mechanism').agg({
        'value_dice': 'mean'
    }).reset_index()

    total_avg = abs(avg_data['value_dice'].sum())
    percentages = ""
    for _, row in avg_data.iterrows():
        pct = (abs(row['value_dice']) / total_avg) * 100 if total_avg != 0 else 0
        percentages += f"{row['mechanism']}: {pct:.1f}%\n"

    plt.figtext(0.15, 0.15, f"Avg. Contribution:\n{percentages}",
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mechanism_comparison_dice.png'), dpi=300)
    plt.close()

    # Similar plot for F1
    plt.figure(figsize=(10, 8))
    plt.scatter(f1_pivot['P(labels|images)'], f1_pivot['P(images)'],
                alpha=0.7, s=100)

    # Add labels for each point
    for i, row in f1_pivot.iterrows():
        plt.annotate(f"{int(row['train_annotator'])}->{int(row['test_annotator'])}",
                    (row['P(labels|images)'], row['P(images)']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')

    # Add diagonal line
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

    plt.title('Comparison of Mechanism Contributions (F1)')
    plt.xlabel('P(labels|images) contribution')
    plt.ylabel('P(images) contribution')
    plt.grid(True, alpha=0.3)

    # Add text box showing average contribution percentages
    avg_data = plot_df.groupby('mechanism').agg({
        'value_f1': 'mean'
    }).reset_index()

    total_avg = abs(avg_data['value_f1'].sum())
    percentages = ""
    for _, row in avg_data.iterrows():
        pct = (abs(row['value_f1']) / total_avg) * 100 if total_avg != 0 else 0
        percentages += f"{row['mechanism']}: {pct:.1f}%\n"

    plt.figtext(0.15, 0.15, f"Avg. Contribution:\n{percentages}",
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mechanism_comparison_f1.png'), dpi=300)
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data('annotator-comparisons-7.csv')  # Replace with your actual file path

    # Verify attributions
    print("\nVerifying Shapley attributions...")
    verification_df = verify_attributions(df)

    # Analyze mechanism contributions
    print("\nAnalyzing mechanism contributions...")
    contribution_df, avg_contrib = analyze_mechanism_contribution(df)

    # Plot heatmap matrices
    print("\nCreating heatmap visualizations...")
    plot_heatmap_matrices(df, metric='Dice')
    plot_heatmap_matrices(df, metric='F1')

    # Plot mechanism comparison
    print("\nCreating mechanism comparison plots...")
    plot_mechanism_comparison(contribution_df)

    print("\nAnalysis complete! Results saved to 'results' directory.")

if __name__ == "__main__":
    main()