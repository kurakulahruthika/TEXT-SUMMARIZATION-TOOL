# TEXT-SUMMARIZATION-TOOL

*company*:codtech it solutions

*NAME:KURAKULA HRUTHIKA*

*INTERN ID*:CT04DF664

*DOMAIN*: AI

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

## DESCRIPTION OF MY WORK ##

## 📝 Text Summarization Tool – Project Description

###  Overview

This project focuses on developing a *Text Summarization Tool* that can automatically condense lengthy articles or documents into shorter, coherent summaries while preserving key information. The goal was to build an intelligent system that helps users save time by quickly grasping the core meaning of large text content.

I implemented both *extractive* and *abstractive* approaches to summarization using Natural Language Processing (NLP) and deep learning techniques. This project allowed me to understand and apply advanced language models and token-based summarization techniques.

---

###  Objectives

* Load and process raw textual data (e.g., articles, news, blogs, etc.)
* Implement extractive summarization using NLP techniques
* Explore abstractive summarization using Transformer-based models
* Allow user input via text box or file and display summarized output
* Keep the summary fluent, relevant, and concise

---

### ⚙ How I Did It – Step-by-Step Breakdown

####  Step 1: Understanding the Summarization Types

Before building the tool, I researched two main types of summarization:

* *Extractive Summarization:* Selects key sentences from the original text
* *Abstractive Summarization:* Rewrites the main idea in a new way, similar to how humans summarize

This helped me decide which models and libraries to use based on the type.

####  Step 2: Preprocessing the Text

I built a pipeline to clean and prepare the input text:

* Removed punctuation, stopwords, and extra whitespace
* Tokenized sentences and words using nltk and spacy
* Used TF-IDF scores and similarity metrics for extractive summaries

For longer documents, I also added chunking to divide the text into manageable parts for deep learning models like BART or T5.

####  Step 3: Implementing Extractive Summarization

I started with a basic extractive approach using:

* *Frequency-based sentence scoring* (Term Frequency-Inverse Document Frequency)
* *Cosine similarity with sentence embeddings*
* *LexRank / TextRank algorithm*

The process involved ranking each sentence based on its importance and selecting the top n sentences as the summary. I used Python libraries such as:

* nltk
* gensim
* sklearn
* sumy

This method worked well for factual or news articles.

####  Step 4: Abstractive Summarization with Transformers

To generate more human-like summaries, I integrated pre-trained models like:

* *BART (Facebook)*
* *T5 (Text-To-Text Transfer Transformer)*
* Using Hugging Face’s transformers library

Sample code:

python
from transformers import pipeline
summarizer = pipeline("summarization")
summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)


These models generated summaries that weren’t just copied sentences, but newly formed ones using language understanding.

####  Step 5: Testing and Interface

I tested the tool using:

* News articles (CNN, BBC)
* Wikipedia pages
* Research abstracts

I also built a simple interface using Python’s Tkinter and a command-line version where users can paste text or upload .txt files.

---

###  Tools and Libraries Used

* *Python 3*
* *NLTK, SpaCy, Gensim*
* *Scikit-learn*
* *Hugging Face Transformers*
* *Sumy, LexRank, TextRank*
* *Tkinter (for GUI)*

---

###  Challenges Faced

* Pre-trained models like BART or T5 require significant memory and GPU; I optimized input chunk size to prevent crashes
* Summarizing long documents meant truncation; I had to chunk the input while maintaining context
* Keeping extractive summaries grammatically correct and non-redundant was tricky
* Balancing summary length vs. relevance required manual tuning

---

###  What I Learned

* The core differences between extractive and abstractive summarization
* How Transformer-based models work in a text-to-text format
* How to clean, preprocess, and tokenize large text data
* How sentence embeddings and similarity scoring improve extractive techniques
* How to design NLP applications that are both practical and efficient

---

###  Output Example

*Input Text:* A 1000-word news article on climate change

*Extractive Summary:*

> "The global climate has warmed significantly in recent decades. Greenhouse gases like CO₂ are the primary cause. Scientists warn immediate action is required to avoid catastrophic impacts."

*Abstractive Summary:*

> "Scientists report that rising greenhouse gases are driving global warming, urging urgent environmental reforms to prevent severe consequences."

---

###  Conclusion

The Text Summarization Tool project gave me real-world experience working with textual data, building intelligent NLP pipelines, and applying cutting-edge models like BART and T5. This tool has practical applications in journalism, education, research, and more. In the future, I plan to deploy it as a web app with support for multiple languages and file types (PDF, DOCX, etc.).

---
