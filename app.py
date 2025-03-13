import streamlit as st
import cv2
import numpy as np
from fer import FER
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# --- Spotify API Setup ---
SPOTIPY_CLIENT_ID = "your_spotify_client_id"
SPOTIPY_CLIENT_SECRET = "your_spotify_client_secret"

def init_spotify():
    """Initialize Spotify API with error handling."""
    try:
        return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
            client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
        ))
    except Exception as e:
        st.error(f"Error connecting to Spotify: {e}")
        return None

spotify = init_spotify()

# --- Emotion-to-Genre Mapping ---
EMOTION_GENRE_MAP = {
    "happy": ["pop", "dance", "party"],
    "sad": ["acoustic", "piano", "blues"],
    "angry": ["rock", "metal", "hardcore"],
    "surprised": ["indie", "jazz", "soul"],
    "neutral": ["chill", "ambient", "lofi"]
}

LANGUAGE_MARKET_MAP = {
    "English": "US",
    "Hindi": "IN",
    "Kannada": "IN",
    "Punjabi": "IN"
}

@st.cache_data
def detect_emotion(image):
    """Detects emotion from an image using the FER library."""
    try:
        detector = FER(mtcnn=False)
        emotions = detector.detect_emotions(image)
        if not emotions:
            return "neutral"
        return max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
    except Exception as e:
        st.error(f"Error detecting emotion: {e}")
        return "neutral"

@st.cache_data
def get_songs(emotion, language, offset=0):
    """Fetch songs based on mood and language using Spotify API."""
    if not spotify:
        return []
    genres = EMOTION_GENRE_MAP.get(emotion, ["chill"])
    market = LANGUAGE_MARKET_MAP.get(language, "US")
    query = f"genre:{genres[0]}"
    try:
        results = spotify.search(q=query, type="track", limit=10, offset=offset, market=market)
        return [(track["name"], track["artists"][0]["name"], track["external_urls"]["spotify"])
                for track in results["tracks"]["items"]]
    except Exception as e:
        st.error(f"Error fetching songs: {e}")
        return []

# --- Streamlit UI ---
st.title("🎭 Mood-Based Song Recommender 🎶")
st.sidebar.write("**Upload an image or capture from webcam**")

img_file = st.file_uploader("📂 Upload an image", type=["png", "jpg", "jpeg"])
camera_image = st.camera_input("📷 Capture from Webcam")
image = None

if img_file or camera_image:
    img_source = img_file if img_file else camera_image
    file_bytes = np.frombuffer(img_source.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))
        st.image(image, caption="Uploaded/Captured Image", width=300)
    else:
        st.error("Failed to load image! Try again.")

if image is not None:
    detected_emotion = detect_emotion(image)
    st.subheader(f"🎭 **Detected Mood:** {detected_emotion.capitalize()}")
    
    language = st.selectbox("🌐 Choose your language", ["English", "Kannada", "Hindi", "Punjabi"])
    songs = get_songs(detected_emotion, language)
    if songs:
        st.write("🎵 **Recommended Songs:**")
        for name, artist, url in songs:
            st.markdown(f"- [{name} by {artist}]({url})")
    else:
        st.warning("No songs found! Try again.")
    
    if st.button("🔄 More Songs"):
        more_songs = get_songs(detected_emotion, language, offset=10)
        if more_songs:
            for name, artist, url in more_songs:
                st.markdown(f"- [{name} by {artist}]({url})")
        else:
            st.warning("No more songs found!")

st.markdown("---")
st.markdown("📌 **Tip:** Ensure your face is well-lit for better mood detection!")
