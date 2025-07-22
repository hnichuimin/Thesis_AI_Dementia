import pandas as pd
import re


df = pd.read_csv("Working_dataset_8.7.25.csv") 

prediction_keywords = ['predict', 'prediction', 'forecast', 'classifier']
diagnosis_keywords = ['diagnosis', 'diagnose', 'diagnostic', 'detect']
progression_keywords = ['progression', 'deterioration', 'trajectory', 'track']
treatment_keywords = ['treatment', 'therapy', 'intervention']
prognosis_keywords = ['prognosis', 'outcome', 'survival', 'mortality']



def keyword_present(text, keywords):
    text = text.lower()
    return any(re.search(rf'\b{re.escape(kw)}\b', text) for kw in keywords)

results = {
    'prediction': [],
    'diagnosis': [],
    'progression': [],
    'treatment': [],
    'prognosis': []
}


for abstract in df['abstract']:
    abstract_lower = abstract.lower()
    results['prediction'].append(keyword_present(abstract_lower, prediction_keywords))
    results['diagnosis'].append(keyword_present(abstract_lower, diagnosis_keywords))
    results['progression'].append(keyword_present(abstract_lower, progression_keywords))
    results['treatment'].append(keyword_present(abstract_lower, treatment_keywords))
    results['prognosis'].append(keyword_present(abstract_lower, prognosis_keywords))


df = df.assign(**results)


overview = df[['prediction', 'diagnosis', 'progression', 'treatment', 'prognosis']].sum().to_frame(name='count')
overview['percentage'] = (overview['count'] / len(df)) * 100

print(overview)


df.to_csv("annotated_ai_papers.csv", index=False)
