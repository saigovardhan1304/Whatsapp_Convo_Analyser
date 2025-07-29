import streamlit as st
import re
import pandas as pd
import numpy as np
import emoji
import plotly.express as px # For interactive plots
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image # Required for ImageColorGenerator and custom masks
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import codecs
import regex # Used for emoji splitting in the user's provided code
import os # Import os module to check for font paths
import matplotlib.font_manager as fm # Import font_manager for more robust font finding

# --- Configuration and Setup ---
# Set stdout to use utf-8 encoding for emojis (especially on Windows)
# This is crucial for Streamlit's console output if running locally, though Streamlit handles UTF-8 well internally.
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Download VADER lexicon if not already downloaded
# Streamlit will run this once per session, but NLTK handles idempotency.
try:
    # Attempt to find the resource
    nltk.data.find('sentiment/vader_lexicon.zip')
# Catch LookupError, which is the base class for resource not found errors
except LookupError:
    st.info("Just a moment! We're downloading some language data for sentiment analysis. This only happens once. 🚀")
    nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sentiments = SentimentIntensityAnalyzer()

# --- Font Detection Helper Function ---
@st.cache_resource # Cache the font path lookup for performance
def _get_font_path():
    """
    Attempts to find a suitable TrueType font for WordCloud across different OS.
    Returns the font path if found, otherwise None.
    """
    font_path = None
    
    # --- START MANUAL FONT PATH OVERRIDE (If you know a specific font file works) ---
    # If you found a font in C:\Windows\Fonts\ or similar, uncomment ONE of the lines below
    # and replace the path with your actual font file path (e.g., "C:/Windows/Fonts/arial.ttf").
    # font_path = "C:/Windows/Fonts/arial.ttf"
    # font_path = "C:/Windows/Fonts/arialn.ttf" # If you specifically want Arial Narrow
    # --- END MANUAL FONT PATH OVERRIDE ---

    # If not manually set, try common system font paths first
    if font_path is None:
        if sys.platform == "win32": # Windows
            possible_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/times.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/arialn.ttf", # Added Arial Narrow specific file
                "C:/Windows/Fonts/ARIALUNI.TTF" # Arial Unicode MS, good for broader character support
            ]
        elif sys.platform == "darwin": # macOS
            possible_paths = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc", # .ttc files can sometimes be problematic, but worth a try
                "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
            ]
        else: # Linux or others
            possible_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                font_path = path
                break

    # Fallback using matplotlib's font manager (more generic and often reliable)
    if font_path is None:
        try:
            # Try to find a generic sans-serif font that matplotlib knows about
            font_path = fm.findfont(fm.FontProperties(family='sans-serif'))
            if not os.path.exists(font_path): # Double check if the path actually exists
                font_path = None
        except Exception:
            pass # If font_manager fails for any reason

    return font_path

# Get the font path once at the start of the application
GLOBAL_FONT_PATH = _get_font_path()
if GLOBAL_FONT_PATH is None:
    st.warning("Heads up! We couldn't find a suitable TrueType font on your system. Word clouds might not render correctly or may show blank spaces instead of text. Please ensure you have common fonts like Arial, Times New Roman, DejaVu Sans, or Liberation Sans installed.")


# --- Data Extraction and Preprocessing Functions ---

# Regex pattern for WhatsApp chat lines (DD/MM/YYYY, HH:MM AM/PM - Author: Message)
# This pattern is more robust and handles optional AM/PM
def date_time_pattern_match(line):
    # Matches: DD/MM/YYYY, HH:MM - (optional AM/PM)
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?: [APap][Mm])?) -'
    return re.match(pattern, line)

# Extracts date, time, author, and message from a chat line
def parse_chat_line(line):
    split_line = line.split(' - ', 1) # Split only on the first ' - '
    if len(split_line) < 2:
        return None, None, None, line # Handle lines that don't match pattern

    date_time_str = split_line[0]
    message_content = split_line[1]

    # Extract date and time
    date_part, time_part = date_time_str.split(', ', 1)

    # Determine author and actual message
    author = None
    message = message_content

    # Check if the message_content starts with an author name followed by ':'
    # This is a common pattern for messages, excluding system messages or multi-line messages
    author_message_split = message_content.split(': ', 1)
    if len(author_message_split) > 1:
        author = author_message_split[0]
        message = author_message_split[1]

    return date_part, time_part, author, message

# Main function to load and clean WhatsApp data
@st.cache_data # Cache data loading for performance
def load_and_clean_whatsapp_data(uploaded_file):
    data_list = []
    # Read file content as string
    file_content = uploaded_file.getvalue().decode("utf-8")
    
    # Split content into lines and skip the first line (header)
    lines = file_content.split('\n')[1:] 

    message_buffer = []
    current_date, current_time, current_author = None, None, None

    for line in lines:
        line = line.strip()
        if date_time_pattern_match(line):
            # If a new message starts, append the buffered message
            if message_buffer:
                data_list.append([current_date, current_time, current_author, ' '.join(message_buffer)])
            
            # Parse the new line
            current_date, current_time, current_author, message = parse_chat_line(line)
            message_buffer = [message] # Start new buffer with current message
        else:
            # If it's a continuation of the previous message
            message_buffer.append(line)
    
    # Append the last buffered message after loop ends
    if message_buffer:
        data_list.append([current_date, current_time, current_author, ' '.join(message_buffer)])

    df = pd.DataFrame(data_list, columns=["Date", 'Time', 'Author', 'Message'])

    # Drop rows where Author is None (often system messages or malformed lines)
    df.dropna(subset=['Author'], inplace=True)

    # Convert 'Date' column to datetime objects
    # Explicitly set dayfirst=True to handle DD/MM/YYYY format and suppress warning
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') # coerce invalid dates to NaT

    # Drop rows with NaT dates if any failed to parse
    df.dropna(subset=['Date'], inplace=True)

    # --- Additional Data Cleaning for Messages ---
    # Create a copy to avoid SettingWithCopyWarning for subsequent modifications
    data_clean = df.copy()

    # Remove '<Media omitted>' messages
    data_clean = data_clean[data_clean['Message'] != '<Media omitted>']

    # Remove URLs (basic regex)
    URLPATTERN = r'(https?://\S+)'
    data_clean.loc[:, 'URL_Count'] = data_clean['Message'].apply(lambda x: len(re.findall(URLPATTERN, x)))
    data_clean.loc[:, 'Message'] = data_clean['Message'].apply(lambda x: re.sub(URLPATTERN, '', x))

    # Remove emojis (using the emoji library) for sentiment analysis and word clouds
    data_clean.loc[:, 'Message_Clean'] = data_clean['Message'].apply(lambda x: emoji.replace_emoji(x, replace=''))
    
    # Convert to lowercase for consistent text analysis (e.g., word clouds)
    data_clean.loc[:, 'Message_Clean'] = data_clean['Message_Clean'].str.lower()

    # Add Letter_Count and Word_Count
    data_clean.loc[:, 'Letter_Count'] = data_clean['Message_Clean'].apply(lambda s : len(s))
    data_clean.loc[:, 'Word_Count'] = data_clean['Message_Clean'].apply(lambda s : len(s.split(' ')))
    data_clean.loc[:, "MessageCount"] = 1 # For easy counting

    return df, data_clean # Return both original (for emoji analysis) and cleaned DFs

# --- Sentiment Analysis ---
def calculate_sentiment_scores(df_cleaned):
    # Use .loc to avoid SettingWithCopyWarning
    df_cleaned.loc[:, "Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df_cleaned["Message_Clean"]]
    df_cleaned.loc[:, "Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df_cleaned["Message_Clean"]]
    df_cleaned.loc[:, "Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df_cleaned["Message_Clean"]]
    df_cleaned.loc[:, "Compound"] = [sentiments.polarity_scores(i)["compound"] for i in df_cleaned["Message_Clean"]] # Compound score for overall sentiment

    # Determine individual message sentiment label
    def get_message_sentiment_label(row):
        if row['Positive'] > row['Negative'] and row['Positive'] > row['Neutral']:
            return 'Positive'
        elif row['Negative'] > row['Positive'] and row['Negative'] > row['Neutral']:
            return 'Negative'
        else:
            return 'Neutral'

    df_cleaned.loc[:, 'Sentiment_Label'] = df_cleaned.apply(get_message_sentiment_label, axis=1)

    return df_cleaned

# --- Overall Sentiment Determination ---
def display_overall_sentiment(df_with_sentiment):
    st.subheader("Overall Sentiment Distribution 📊")
    st.write("This chart shows the general emotional tone of your chat messages.")
    # Sum up the scores
    x = df_with_sentiment["Positive"].sum()
    y = df_with_sentiment["Negative"].sum()
    z = df_with_sentiment["Neutral"].sum()

    # Calculate percentages for better interpretation
    total_sum = x + y + z
    if total_sum == 0: # Handle case with no messages after cleaning
        st.write("Hmm, it looks like there aren't enough valid messages to determine sentiment. Try exporting a chat with more text messages!")
        return

    pos_percent = (x / total_sum) * 100
    neg_percent = (y / total_sum) * 100
    neu_percent = (z / total_sum) * 100

    st.markdown(f"**Positive Messages:** {pos_percent:.2f}%")
    st.markdown(f"**Negative Messages:** {neg_percent:.2f}%")
    st.markdown(f"**Neutral Messages:** {neu_percent:.2f}%")

    if (x > y) and (x > z):
        st.success("The overall vibe of this chat is **Positive!** 😊 Keep up the good spirits!")
    elif (y > x) and (y > z):
        st.error("The overall vibe of this chat is **Negative.** 😠 Maybe it's time for some positive vibes?")
    else:
        st.info("The overall vibe of this chat is **Neutral.** 🙂 Just the facts, ma'am!")

    # Plotting sentiment distribution
    sentiment_counts = pd.Series({'Positive': x, 'Negative': y, 'Neutral': z})
    
    if sentiment_counts.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#99ff99','#ff9999','#66b3ff'])
        ax.set_title('Overall Sentiment Distribution')
        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
    else:
        st.write("Can't plot sentiment distribution - no data found.")


# --- Analysis and Visualization Functions ---

def display_user_activity(df_cleaned):
    st.subheader("Who's Chatting the Most? 🗣️")
    st.write("Discover the most active participants in your chat group!")
    message_counts = df_cleaned['Author'].value_counts()
    
    st.markdown("**Top 10 Most Active Chatters:**")
    st.dataframe(message_counts.head(10))

    # Plotting messages by author
    fig, ax = plt.subplots(figsize=(12, 6))
    message_counts.head(10).plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Top 10 Most Active Users by Message Count')
    ax.set_xlabel('Author')
    ax.set_ylabel('Number of Messages Sent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Dive D...eeper: Individual Chatter Stats 🕵️")
    st.write("Select a person from your chat to see their specific activity.")
    unique_authors = df_cleaned['Author'].unique()
    selected_author = st.selectbox("Choose a chatter:", unique_authors)

    if selected_author:
        req_df = df_cleaned[df_cleaned["Author"] == selected_author]
        
        st.markdown(f'**Stats for {selected_author}:**')
        st.write(f'Total Messages Sent: **{req_df.shape[0]}**')
        
        if req_df.shape[0] > 0:
            words_per_message = (np.sum(req_df['Word_Count'])) / req_df.shape[0]
            st.write(f'Average Words per Message: **{words_per_message:.2f}**')
        else:
            st.write('No messages found for this author.')

        # Ensure df_original_full is accessible (it's a global from the main script scope)
        global df_original_full 
        media_messages_count = df_original_full[df_original_full['Author'] == selected_author][df_original_full['Message'] == '<Media omitted>'].shape[0]
        st.write(f'Media Messages Sent: **{media_messages_count}**')

        # Emoji count for individual author
        author_emojis = []
        for msg in df_original_full[df_original_full['Author'] == selected_author]['Message'].dropna():
            author_emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])
        st.write(f'Emojis Sent: **{len(author_emojis)}**')

        links_sent = np.sum(req_df["URL_Count"])
        st.write(f'Links Shared: **{links_sent}**')

        # Individual Author Word Cloud
        st.subheader(f"What does {selected_author} talk about? (Word Cloud) ☁️")
        text_author = " ".join(review for review in req_df.Message_Clean)
        if text_author:
            stopwords = set(STOPWORDS)
            stopwords.update(["media", "omitted", "message", "link"])
            
            # Use the globally detected font path for individual author word cloud
            wordcloud_kwargs = {"stopwords": stopwords, "background_color": "white", "width": 800, "height": 400}
            if GLOBAL_FONT_PATH: # Only add font_path if it was successfully found
                wordcloud_kwargs["font_path"] = GLOBAL_FONT_PATH

            wordcloud_author = WordCloud(**wordcloud_kwargs).generate(text_author)

            fig_wc_author, ax_wc_author = plt.subplots(figsize=(10, 5))
            ax_wc_author.imshow(wordcloud_author, interpolation='bilinear')
            ax_wc_author.axis("off")
            ax_wc_author.set_title(f'Common Words by {selected_author}')
            st.pyplot(fig_wc_author)
        else:
            st.write(f"Not enough text data for {selected_author}'s word cloud. They might be a person of few words! 😉")


def display_emoji_analysis(df_original):
    st.subheader("Emoji Power! 💪")
    st.write("See which emojis reign supreme in your chat.")
    all_emojis = []
    for message in df_original['Message'].dropna(): # Use original message for emoji extraction
        all_emojis.extend([c for c in message if c in emoji.EMOJI_DATA]) # Use EMOJI_DATA for efficiency
    
    if all_emojis:
        emoji_counts = Counter(all_emojis)
        st.markdown("**Top 10 Most Used Emojis:**")
        emoji_df = pd.DataFrame(emoji_counts.most_common(10), columns=['Emoji', 'Count'])
        st.dataframe(emoji_df)

        # Plotting emoji distribution
        fig = px.bar(emoji_df, x='Emoji', y='Count', title='Top 10 Most Used Emojis')
        st.plotly_chart(fig)
    else:
        st.write("Looks like this chat is emoji-free! 🧐")

def display_word_clouds(df_with_sentiment):
    st.subheader("What's the Buzz? Word Clouds! ☁️")
    st.write("Word clouds visually highlight the most frequently used words in your chat. Bigger words mean more frequent use!")
    stopwords = set(STOPWORDS)
    stopwords.update(["media", "omitted", "message", "link"]) # Add common WhatsApp chat words

    # Use the globally detected font path for all word clouds in this function
    base_wordcloud_kwargs = {"stopwords": stopwords, "background_color": "white", "width": 800, "height": 400}
    if GLOBAL_FONT_PATH: # Only add font_path if it was successfully found
        base_wordcloud_kwargs["font_path"] = GLOBAL_FONT_PATH


    # Overall Word Cloud
    st.markdown("### Overall Chat Topics")
    text_overall = " ".join(review for review in df_with_sentiment.Message_Clean)
    if text_overall:
        wordcloud_overall = WordCloud(**base_wordcloud_kwargs).generate(text_overall)
        fig_wc_overall, ax_wc_overall = plt.subplots(figsize=(10, 5))
        ax_wc_overall.imshow(wordcloud_overall, interpolation='bilinear')
        ax_wc_overall.axis("off")
        ax_wc_overall.set_title('Overall Chat Word Cloud')
        st.pyplot(fig_wc_overall)
    else:
        st.write("Not enough text data for an overall word cloud.")

    # Positive Word Cloud
    st.markdown("### Positive Conversations")
    text_positive = " ".join(review for review in df_with_sentiment[df_with_sentiment['Positive'] > 0.5].Message_Clean)
    if text_positive:
        wordcloud_positive = WordCloud(**base_wordcloud_kwargs).generate(text_positive)
        fig_wc_pos, ax_wc_pos = plt.subplots(figsize=(10, 5))
        ax_wc_pos.imshow(wordcloud_positive, interpolation='bilinear')
        ax_wc_pos.axis("off")
        ax_wc_pos.set_title('Positive Messages Word Cloud')
        st.pyplot(fig_wc_pos)
    else:
        st.write("Not enough positive messages for a word cloud.")

    # Negative Word Cloud
    st.markdown("### Negative Conversations")
    text_negative = " ".join(review for review in df_with_sentiment[df_with_sentiment['Negative'] > 0.5].Message_Clean)
    if text_negative:
        wordcloud_negative = WordCloud(**base_wordcloud_kwargs).generate(text_negative)
        fig_wc_neg, ax_wc_neg = plt.subplots(figsize=(10, 5))
        ax_wc_neg.imshow(wordcloud_negative, interpolation='bilinear')
        ax_wc_neg.axis("off")
        ax_wc_neg.set_title('Negative Messages Word Cloud')
        st.pyplot(fig_wc_neg)
    else:
        st.write("Not enough negative messages for a word cloud.")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="WhatsApp Chat Analyzer")

st.title("Your WhatsApp Chat, Uncovered! 🕵️‍♀️�")
st.markdown("Upload your chat export (.txt file) to explore its sentiment, activity, and hidden patterns. Let's dive in!")

with st.expander("🤔 How to export your WhatsApp chat?"):
    st.markdown("""
    1.  Open the individual or group chat in WhatsApp.
    2.  Tap on the **three dots (⋮)** in the top right corner (Android) or the **contact/group name** at the top (iOS).
    3.  Select **More** (Android) or scroll down to **Export Chat** (iOS).
    4.  Choose **Export Chat**.
    5.  Select **Without Media** (this keeps the file size small and focuses on text).
    6.  Save or share the `.txt` file to your device.
    7.  Upload that `.txt` file below! 👇
    """)

uploaded_file = st.file_uploader("Ready? Upload your WhatsApp chat export (.txt file) here!", type=["txt"])

if uploaded_file is not None:
    with st.spinner("Analyzing your chat... This might take a few moments, depending on chat size. Hang tight! ✨"):
        df_original_full, cleaned_data_df = load_and_clean_whatsapp_data(uploaded_file)
        
        if cleaned_data_df.empty:
            st.warning("Oops! It looks like we couldn't find any valid messages in your chat file. Please ensure it's a standard WhatsApp export and try again. 🙏")
            st.stop()

        df_with_sentiment = calculate_sentiment_scores(cleaned_data_df)
        
        total_messages = len(df_original_full)
        media_messages_count = df_original_full[df_original_full['Message'] == '<Media omitted>'].shape[0]
        links_count = np.sum(df_with_sentiment['URL_Count'])

        # Total emoji count (using the original df before cleaning emojis for sentiment)
        all_emojis_total = []
        for message in df_original_full['Message'].dropna():
            all_emojis_total.extend([c for c in message if c in emoji.EMOJI_DATA])
        emojis_count = len(all_emojis_total)

    st.sidebar.header("Choose Your Journey 🗺️")
    analysis_method = st.sidebar.radio(
        "What would you like to explore?",
        ("Overview", "Sentiment Analysis", "User Activity", "Emoji Analysis", "Word Clouds")
    )

    st.success(f"Great! We've successfully loaded **{total_messages}** messages from your chat. Let's explore!")
    st.info(f"We'll analyze **{len(df_with_sentiment)}** messages after filtering out media, links, and empty lines for a clearer picture.")

    if analysis_method == "Overview":
        st.header("Chat at a Glance 🌟")
        st.markdown("Here's a quick summary of your chat's key statistics:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", total_messages)
            st.metric("Media Messages", media_messages_count)
        with col2:
            st.metric("Emojis Sent", emojis_count)
            st.metric("Links Shared", links_count)
        with col3:
            st.metric("Chat Participants", df_original_full['Author'].nunique())
            st.metric("Chat Duration", f"{df_original_full['Date'].min().strftime('%Y-%m-%d')} to {df_original_full['Date'].max().strftime('%Y-%m-%d')}")

        st.subheader("A Peek into Your Chat 🧐")
        st.write("Here are the first few messages with their detected sentiment:")
        st.dataframe(df_with_sentiment[['Date', 'Time', 'Author', 'Message', 'Sentiment_Label']].head(10))


    elif analysis_method == "Sentiment Analysis":
        st.header("Feeling the Vibe: Sentiment Analysis 💖💔🤔")
        st.write("Our powerful AI (VADER) has analyzed the emotional tone of each message. Let's see the overall mood!")
        display_overall_sentiment(df_with_sentiment)
        
        st.markdown("---")
        st.subheader("Curious about specific messages?")
        with st.expander("Show me the Negative Messages 😠"):
            negative_messages_df = df_with_sentiment[df_with_sentiment['Negative'] > 0].copy()
            if not negative_messages_df.empty:
                st.dataframe(negative_messages_df[['Date', 'Author', 'Message', 'Negative', 'Sentiment_Label']])
            else:
                st.info("Great news! No messages with a negative sentiment score > 0 were found in your chat. 🎉")

        with st.expander("Show me the Positive Messages 😊"):
            positive_messages_df = df_with_sentiment[df_with_sentiment['Positive'] > 0].copy()
            if not positive_messages_df.empty:
                st.dataframe(positive_messages_df[['Date', 'Author', 'Message', 'Positive', 'Sentiment_Label']])
            else:
                st.info("Looks like your chat is full of positive vibes! No messages with a positive sentiment score > 0 were found. (This might happen if all positive messages were perfectly neutral or had very low scores.)")

    elif analysis_method == "User Activity":
        st.header("Who's the Chat Star? 🏆")
        st.write("Let's uncover the most talkative members and their chatting habits!")
        display_user_activity(df_with_sentiment)

    elif analysis_method == "Emoji Analysis":
        st.header("Emoji Story! 😂😭👍")
        st.write("Emojis add so much flavor to our chats! See which ones are used the most.")
        display_emoji_analysis(df_original_full) # Pass the original df here for full emoji data

    elif analysis_method == "Word Clouds":
        st.header("What's Everyone Talking About? 💭")
        st.write("Word clouds give you a quick visual summary of the most common words. The bigger the word, the more frequently it appears!")
        display_word_clouds(df_with_sentiment)

else:
    st.info("Ready to explore your chat? Just upload your WhatsApp chat export file (.txt) above! ⬆️")

st.markdown("---")
st.markdown("Made with ❤️ by your friend Saigovardhan.")

