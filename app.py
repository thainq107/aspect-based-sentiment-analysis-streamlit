import streamlit as st
from transformers import pipeline

token_classifier = pipeline(
    model="thainq107/abte-restaurants-distilbert-base-uncased", 
    aggregation_strategy="simple"
)

classifier = pipeline(
    model="thainq107/absa-restaurants-distilbert-base-uncased"
)

def main():
    st.title('Aspect-based Sentiment Analysis')
    st.header('Model: DistilBERT. Dataset: SemEval4 Restaurants')
    text_input = st.text_input("Sentence: ", "The bread is top notch as well")
    results = token_classifier(text_input)
    sentence_tags = " ".join([result['word'] for result in results])
    pred_label = classifier(f'{text_input} [SEP] {sentence_tags}')
    st.success(f'Sentence: {text_input} === Term: {sentence_tags} === Sentiment: {pred_label[0]["label"]}')

if __name__ == '__main__':
     main() 
