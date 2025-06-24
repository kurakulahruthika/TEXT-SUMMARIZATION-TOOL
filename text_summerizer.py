import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re

# File path
file_name = r"C:\Users\KALYAN\PycharmProjects\sem2\examplefile.txt"

# Function to read and preprocess the article
def read_article(file_name):
    with open(file_name, 'r') as file:
        filedata = file.read()
    
    # Split into sentences based on periods
    article = filedata.split(". ")
    sentences = []

    for sentence in article:
        cleaned_sentence = re.sub("[^a-zA-Z]", " ", sentence)  # Keep only letters
        words = cleaned_sentence.split()
        if words:  # Avoid adding empty lists
            sentences.append(words)

    return sentences

# Function to calculate cosine similarity between two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w not in stopwords:
            vector1[all_words.index(w)] += 1

    for w in sent2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

# Function to generate similarity matrix
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

# Main function to generate summary
def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences = read_article(file_name)

    if len(sentences) == 0:
        print("No sentences found in the input file.")
        return

    similarity_matrix = gen_sim_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # ✅ FIX: Ensure we only access available sentences
    for i in range(min(top_n, len(ranked_sentences))):
        summarize_text.append(" ".join(ranked_sentences[i][1]))

    print("Summary:\n", ". ".join(summarize_text))


# Run the summarizer
generate_summary(file_name, top_n=5)
