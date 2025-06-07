import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

def show_pie_chart(df):
    fig, ax = plt.subplots()
    df['sentimen'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

def generate_wordcloud(df):
    all_words = ' '.join(df[df['sentimen'] == 'positif']['cleaned'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    st.subheader("Wordcloud Sentimen Positif")
    st.image(wordcloud.to_array())
