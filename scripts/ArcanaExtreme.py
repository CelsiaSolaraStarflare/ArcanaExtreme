import streamlit as st
import openai  # Ensure you have this package installed
import uuid
import socket
import nltk
import os
import time
from dotenv import load_dotenv

from arcana.finder import files_page
from scripts.chatbot import chatbot_page
from arcana.settings import settings_page
from arcana.mixup import mixup_page
from arcana.longresponse import longresponse_page
from arcana.editor import editor_page
from arcana.speech_to_text import speech_to_text_page
from scripts.config import APP_TITLE, CACHE_DIR, INDEX_FILE
from arcana.fiber import FiberDBMS
from arcana.theme import apply_theme

# --- Application Setup ---

def initialize_app():
    """
    Sets up the application on its first run, including downloading necessary
    NLTK data, creating directories, and initializing the database.
    """
    # 1. Load environment variables from .env file
    load_dotenv()

    # 2. Set page config
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide"
    )

    # 2. Initialize theme
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"

    # 3. Ensure NLTK data is available
    import nltk_setup  # This will automatically download required NLTK data

    # 4. Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 4b. Build initial index if it does not exist yet
    if not os.path.exists(INDEX_FILE):
        from indexing import indexing as build_index  # local import to avoid circular
        with st.spinner("First-time setup: building document index… this may take a while"):
            build_index(CACHE_DIR)
            st.success("Initial indexing complete!")

    # 5. Initialize or load the database into session state
    if 'dbms' not in st.session_state:
        dbms = FiberDBMS()
        if os.path.exists(INDEX_FILE):
            print(f"Loading existing database from {INDEX_FILE}...")
            dbms.load_from_file(INDEX_FILE)
        else:
            print("No existing database found, starting with an empty one.")
        st.session_state.dbms = dbms

    # 6. Apply the selected theme
    apply_theme(st.session_state.theme)

# --- Page Routing ---

def main():
    """
    The main function that routes between different pages of the application
    based on user selection.
    """
    # Initialize the app on first run
    if 'initialized' not in st.session_state:
        initialize_app()
        st.session_state.initialized = True

    # Sidebar for navigation
    st.sidebar.title(APP_TITLE)
    page = st.sidebar.radio("Go to", (
        "Chatbot",
        "Files",
        "Settings",
        "Mixup",
        "Long Response",
        "Editor",
        "Speech to Text",
    ))

    # Show the selected page
    if page == "Chatbot":
        chatbot_page()
    elif page == "Files":
        files_page()
    elif page == "Settings":
        settings_page()
    elif page == "Mixup":
        mixup_page()
    elif page == "Long Response":
        longresponse_page()
    elif page == "Editor":
        editor_page()
    elif page == "Speech to Text":
        speech_to_text_page()

if __name__ == "__main__":
    main()
