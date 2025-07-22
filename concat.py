import pandas as pd

# Read the files

df_ss = pd.read_csv('semantic_scholar_results.csv')
df_pm = pd.read_csv('PubMed_results.csv')
df_epmc = pd.read_csv('europepmc_results.csv')

# Bring them together

combined_df = pd.concat([
    df_ss.assign(source='ss'), # keep track of the source of each row
    df_pm.assign(source='pm'),
    df_epmc.assign(source='epmc')
]).set_index('id') # set index to id row

# Save

combined_df.to_csv(
    'combined_data_final.csv',
    index=True,          # Keep the 'id' index column in the CSV
    header=True,        # Include column names
    encoding='utf-8'    # Ensure proper encoding
)


