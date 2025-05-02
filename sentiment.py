import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import matplotlib.pyplot as plt

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stopwords and sentiment analyzer
stopwords = nltk.corpus.stopwords.words('english')
sid = SentimentIntensityAnalyzer()

# Streamlit app UI
st.title('🛍️ Product Review Sentiment Analyzer')
st.subheader('Quick NLP App for E-commerce Reviews')
st.write('Analyze individual or bulk product reviews to classify them as Positive, Negative, or Neutral.')

# TEXT Analysis Section
with st.expander('🔍 Analyze a Single Review'):
    text = st.text_area('Enter a product review')
    if text and st.button('Analyze Sentiment'):
        tokens = nltk.word_tokenize(text)
        
        # Ensure "not" is kept in negation handling
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords or token.lower() == "not"]
        
        normalized_text = ' '.join(filtered_tokens)
        score = sid.polarity_scores(normalized_text)

        st.write("Sentiment Scores:", score)

        if score['compound'] >= 0.05:
            st.success("The review has a Positive sentiment.")
        elif score['compound'] <= -0.05:
            st.error("The review has a Negative sentiment.")
        else:
            st.warning("The review has a Neutral sentiment.")

# CSV Upload and Analysis Section
with st.expander('📂 Analyze Bulk Reviews from CSV'):
    st.warning('CSV should have a column with product reviews (e.g., "review", "Review", "text").')
    uploaded_file = st.file_uploader('Upload a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # ✅ Correct indentation here!

        # Allow different column names (case-insensitive matching)
        possible_columns = ['review', 'Review', 'text', 'Text', 'comment', 'feedback']
        review_column = next((col for col in df.columns if col in possible_columns), None)

        if review_column is None:
            st.error('CSV must contain one of the following column names: "review", "Review", "text", "Text", "comment", "feedback".')
        else:
            df.dropna(subset=[review_column], inplace=True)
            df['score'] = df[review_column].apply(lambda x: sid.polarity_scores(str(x))['compound'])
            df['sentiment'] = np.where(df['score'] > 0.1, 'Positive',
                                       np.where(df['score'] < -0.1, 'Negative', 'Neutral'))

            st.success('Sentiment analysis completed.')
            st.write(df.head(5))

            # 📊 Sentiment Distribution with Custom Colors
    

            st.subheader("📊 Sentiment Distribution")
    
            # Define custom colors for each sentiment category
            colors = {'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'red'}

            # Count occurrences of each sentiment category
            sentiment_counts = df["sentiment"].value_counts()

            # Create a Matplotlib figure
            fig, ax = plt.subplots()
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=[colors[s] for s in sentiment_counts.index])
            ax.set_xlabel("Sentiment Category")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution")

           # Display the plot in Streamlit
            st.pyplot(fig)

            csv = df.to_csv(index=False)
            st.download_button('📥 Download Analyzed CSV', data=csv, file_name='analyzed_reviews.csv', mime='text/csv')
