import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Load file
df = pd.read_csv("Working_dataset_8.7.25.csv")
text_column = df['ai_term_found'].dropna().astype(str)

# Extract phrases, split on semicolon
phrases = []
for entry in text_column:
    for phrase in entry.split(';'):
        cleaned = phrase.strip().lower()
        if cleaned:
            phrases.append(cleaned)

# Count frequencies with phrases
phrase_freq = Counter(phrases)

# Generate word cloud with phrases
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    collocations=False,  # Prevent auto-combining word pairs
    prefer_horizontal=0.8,  # 80% horizontal, 20% vertical
    max_font_size=80,  # Cap size so big ones don't dominate space
    min_font_size=10,
    colormap='viridis'
).generate_from_frequencies(phrase_freq)

# Show the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of AI Terms")
plt.show()


