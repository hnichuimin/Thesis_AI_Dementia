import pandas as pd

df = pd.read_csv('Working_dataset_8.7.25.csv')

def create_keyword_year_table(df, keyword_column='ai_term_found'):

    # Split keywords and explode the DataFrame
    exploded = df.assign(
        keyword=df[keyword_column].str.split(';')
    ).explode('keyword')

    # Clean whitespace and filter empty keywords
    exploded['keyword'] = exploded['keyword'].str.strip()
    exploded = exploded[exploded['keyword'] != '']

    # Create count table
    count_table = pd.crosstab(
        index=exploded['keyword'],
        columns=exploded['year']
    )

    # Sort years chronologically
    count_table = count_table.reindex(sorted(count_table.columns), axis=1)

    # Sort keywords by total frequency (descending)
    count_table = count_table.loc[count_table.sum(axis=1).sort_values(ascending=False).index]

    return count_table

# Example usage:
ai_table = create_keyword_year_table(df, 'ai_term_found')
dementia_table = create_keyword_year_table(df, 'dementia_term_found')
print(ai_table)
print(dementia_table)


# heatmaps
import seaborn as sns
import matplotlib.pyplot as plt


def create_heatmap(table, title, figsize=(12, 8)):
    plt.figure(figsize=figsize)

    # Create heatmap with blue-white-red color scale
    ax = sns.heatmap(
        table,
        cmap='coolwarm',
        annot=True,  # Show numbers in cells
        fmt='d',  # Integer formatting
        linewidths=.5,  # Add grid lines
        cbar_kws={'label': 'Number of Papers'}
    )

    # Customize appearance
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Keywords', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return ax


# Create and save AI terms heatmap
create_heatmap(ai_table, "AI Term Occurrences by Year")
plt.savefig('ai_terms_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create and save Dementia terms heatmap
create_heatmap(dementia_table, "Dementia Term Occurrences by Year")
plt.savefig('dementia_terms_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a combined frequency table
combined_table = pd.crosstab(
    index=df['year'],
    columns=[df['ai_term_found'].str.split(';').explode().str.strip(),
             df['dementia_term_found'].str.split(';').explode().str.strip()]
)

# Heatmap for top combinations
top_combinations = combined_table.sum().sort_values(ascending=False).head(20).index
plt.figure(figsize=(15, 8))
sns.heatmap(
    combined_table[top_combinations].T,
    cmap='YlOrRd',
    annot=True,
    fmt='d'
)
plt.title("Top AI-Dementia Term Co-occurrences by Year")
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_combined_heatmap(df, top_n=20, figsize=(15, 10)):

    # Process both term columns
    exploded = df.assign(
        ai_terms=df['ai_term_found'].str.split(';'),
        dementia_terms=df['dementia_term_found'].str.split(';')
    ).explode('ai_terms').explode('dementia_terms')

    # Clean terms
    exploded['ai_terms'] = exploded['ai_terms'].str.strip()
    exploded['dementia_terms'] = exploded['dementia_terms'].str.strip()
    exploded = exploded[(exploded['ai_terms'] != '') &
                        (exploded['dementia_terms'] != '')]

    # Create term pairs
    exploded['term_pair'] = exploded['ai_terms'] + " + " + exploded['dementia_terms']

    # Get top N term pairs by frequency
    top_pairs = exploded['term_pair'].value_counts().head(top_n).index

    # Create cross-tab of year vs term pairs
    cooc_table = pd.crosstab(
        index=exploded['year'],
        columns=exploded['term_pair']
    )[top_pairs]  # Keep only top pairs

    # Sort years chronologically
    cooc_table = cooc_table.sort_index()

    # Create the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cooc_table.T,  # Transpose to have term pairs as rows
        cmap='coolwarm',
        annot=True,
        fmt='d',
        linewidths=0.5,
        linecolor='lightgray',
        cbar_kws={'label': 'Co-occurrence Count'},
        center=cooc_table.values.mean()
    )

    # Customize appearance
    ax.set_title(
        f"Top {top_n} AI-Dementia Term Co-occurrences by Year",
        fontsize=14,
        pad=20,
        fontweight='bold'
    )
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Term Pairs', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Number of Papers', rotation=270, labelpad=20)

    plt.tight_layout()
    return ax


# Usage
create_combined_heatmap(df, top_n=10)
plt.savefig('combined_terms_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

