import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
TMDB_API_KEY = st.secrets.get("tmdb_api_key", "")
PLACEHOLDER_POSTER = "https://placehold.co/500x750/0b0e14/5ec9c9?text=No+Poster"
DEFAULT_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 3
MAX_RECOMMENDATIONS = 20

# Hugging Face repo that hosts both large pickle files
HF_REPO_ID = "alokyadav310703/Similarity"
HF_MOVIE_DICT_FILENAME = "movie_dict.pkl"
HF_SIMILARITY_FILENAME = "similarity.pkl"

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling — dark slate + cyan/teal accent theme
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B0E14;
        --bg-card: #131722;
        --accent: #5EC9C9;
        --accent-soft: #8EDCDC;
        --cream: #E7ECF3;
        --muted: #7C8698;
        --divider: #232A38;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--cream);
    }

    .stApp {
        background: radial-gradient(circle at 50% -10%, #161B29 0%, #0B0E14 55%) fixed;
    }

    /* Hide default Streamlit chrome for a cleaner marquee feel,
       but keep the sidebar collapse/expand arrow usable */
    #MainMenu, footer {visibility: hidden;}
    header {
        visibility: hidden;
        height: 0;
    }
    div[data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        position: fixed;
        top: 0.6rem;
        left: 0.6rem;
        z-index: 999999;
    }
    div[data-testid="collapsedControl"] svg {
        fill: var(--accent) !important;
    }

    /* ---------- Marquee header ---------- */
    .marquee-wrap {
        text-align: center;
        padding: 1.6rem 0 1.2rem 0;
        border-bottom: 1px solid var(--divider);
        margin-bottom: 2rem;
    }
    .marquee-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4.2rem;
        letter-spacing: 0.12em;
        color: var(--accent-soft);
        text-shadow: 0 0 18px rgba(94, 201, 201, 0.35);
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .marquee-sub {
        font-size: 0.95rem;
        color: var(--muted);
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0F131C;
        border-right: 1px solid var(--divider);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p {
        color: var(--cream) !important;
    }

    /* ---------- Selectbox ---------- */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--divider) !important;
        color: var(--cream) !important;
        border-radius: 8px;
    }

    /* ---------- Button ---------- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--accent) 0%, #2E9E9E 100%);
        color: #071012;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.9rem;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(94, 201, 201, 0.35);
        color: #071012;
    }

    /* ---------- Movie card ---------- */
    .movie-card {
        background: var(--bg-card);
        border: 1px solid var(--divider);
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        border-color: var(--accent);
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.55);
    }
    .movie-card img {
        width: 100%;
        aspect-ratio: 2 / 3;
        object-fit: cover;
        display: block;
        border-bottom: 1px solid var(--divider);
    }
    .movie-card-body {
        padding: 0.85rem 0.9rem 1rem 0.9rem;
    }
    .movie-title {
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.25;
        margin-bottom: 0.35rem;
        min-height: 2.4em;
    }
    .movie-rating {
        color: var(--accent);
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }

    .section-label {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.4rem;
        letter-spacing: 0.08em;
        color: var(--accent-soft);
        border-left: 3px solid var(--accent);
        padding-left: 0.6rem;
        margin: 0.5rem 0 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_hf_file(filename: str) -> Path:
    """
    Download a file from the Hugging Face Hub the first time the app runs,
    then reuse the cached local copy on every rerun. hf_hub_download stores
    it under the HF cache dir (~/.cache/huggingface by default) and skips
    re-downloading if it's already there.
    """
    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            # repo_type="model" is the default; change to "dataset" if this
            # repo is re-uploaded as a HF *dataset* repo instead.
        )
    )


@st.cache_data(show_spinner="Loading movie catalog…")
def load_data():
    movie_dict_path = get_hf_file(HF_MOVIE_DICT_FILENAME)
    with open(movie_dict_path, "rb") as f:
        movies_dict = pickle.load(f)
    movies_df = pd.DataFrame(movies_dict)

    similarity_path = get_hf_file(HF_SIMILARITY_FILENAME)
    with open(similarity_path, "rb") as f:
        similarity_matrix = pickle.load(f)

    return movies_df, similarity_matrix


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return PLACEHOLDER_POSTER
        return "https://image.tmdb.org/t/p/w500" + poster_path
    except (requests.RequestException, ValueError):
        return PLACEHOLDER_POSTER


def star_rating(score_out_of_10):
    """Render a 0-10 score as a 5-star unicode string."""
    if score_out_of_10 is None:
        return "—"
    stars = round((score_out_of_10 / 10) * 5)
    return "★" * stars + "☆" * (5 - stars) + f"  {score_out_of_10:.1f}"


def recommend(movie, movies_df, similarity_matrix, n=DEFAULT_RECOMMENDATIONS):
    movie_index = movies_df[movies_df["title"] == movie].index[0]
    distances = similarity_matrix[movie_index]
    ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : n + 1]

    results = []
    for i, _ in ranked:
        row = movies_df.iloc[i]
        rating = row["weighted_rating"] if "weighted_rating" in movies_df.columns else row.get("vote_average")
        results.append(
            {
                "title": row["title"],
                "movie_id": row["movie_id"],
                "rating": rating,
            }
        )
    return results


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
movies, similarity = load_data()

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown(
        "A content-based recommender that finds movies similar to one you already love, "
        "using overview text, genres, cast, and crew."
    )
    st.markdown("---")
    st.markdown(f"**Titles in catalog:** {len(movies):,}")
    st.markdown("**Method:** TF-IDF / cosine similarity")
    st.markdown("---")
    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=MIN_RECOMMENDATIONS,
        max_value=MAX_RECOMMENDATIONS,
        value=DEFAULT_RECOMMENDATIONS,
        step=1,
    )
    st.markdown("---")
    st.caption("Built with Streamlit · Posters via TMDB API")

st.markdown(
    """
    <div class="marquee-wrap">
        <div class="marquee-title">🎬 CINEMATCH</div>
        <div class="marquee-sub">Find your next favorite film</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">PICK A MOVIE YOU LIKE</div>', unsafe_allow_html=True)

col_select, col_button = st.columns([4, 1])
with col_select:
    selected_movie_name = st.selectbox(
        "Select movie", movies["title"].values, label_visibility="collapsed"
    )
with col_button:
    recommend_clicked = st.button("Recommend")

if recommend_clicked:
    with st.spinner("Rolling the film reel..."):
        picks = recommend(selected_movie_name, movies, similarity, n=num_recommendations)
        for p in picks:
            p["poster"] = fetch_poster(p["movie_id"])

    st.markdown('<div class="section-label">BECAUSE YOU WATCHED THIS</div>', unsafe_allow_html=True)

    # Render in rows of 5 so 10 recommendations wrap onto two rows
    row_size = 5
    for row_start in range(0, len(picks), row_size):
        row_picks = picks[row_start : row_start + row_size]
        cols = st.columns(row_size)
        for col, movie in zip(cols, row_picks):
            with col:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <img src="{movie['poster']}" alt="{movie['title']}" />
                        <div class="movie-card-body">
                            <div class="movie-title">{movie['title']}</div>
                            <div class="movie-rating">{star_rating(movie['rating'])}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    st.info("Pick a movie above and hit **Recommend** to see similar titles.")

# # import pickle

# # import pandas as pd
# # import requests
# # import streamlit as st

# # # ----------------------------------------------------------------------------
# # # Config
# # # ----------------------------------------------------------------------------
# # TMDB_API_KEY = "33e1f99be03fafa06033cca28a4c8e47"  # move to st.secrets["tmdb_api_key"] before sharing this repo
# # PLACEHOLDER_POSTER = "https://placehold.co/500x750/1a0f12/d4a24c?text=No+Poster"

# # st.set_page_config(
# #     page_title="CineMatch — Movie Recommender",
# #     page_icon="🎬",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )

# # # ----------------------------------------------------------------------------
# # # Styling — cinematic marquee theme (deep burgundy + brushed gold)
# # # ----------------------------------------------------------------------------
# # st.markdown(
# #     """
# #     <style>
# #     @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

# #     :root {
# #         --bg: #150C0F;
# #         --bg-card: #201317;
# #         --gold: #D4A24C;
# #         --gold-soft: #E8C784;
# #         --cream: #F5EDE4;
# #         --muted: #A08E88;
# #         --divider: #3A2429;
# #     }

# #     html, body, [class*="css"] {
# #         font-family: 'Inter', sans-serif;
# #         color: var(--cream);
# #     }

# #     .stApp {
# #         background: radial-gradient(circle at 50% -10%, #2A1519 0%, #150C0F 55%) fixed;
# #     }

# #     /* Hide default Streamlit chrome for a cleaner marquee feel */
# #     #MainMenu, footer, header {visibility: hidden;}

# #     /* ---------- Marquee header ---------- */
# #     .marquee-wrap {
# #         text-align: center;
# #         padding: 1.6rem 0 1.2rem 0;
# #         border-bottom: 1px solid var(--divider);
# #         margin-bottom: 2rem;
# #     }
# #     .marquee-title {
# #         font-family: 'Bebas Neue', sans-serif;
# #         font-size: 4.2rem;
# #         letter-spacing: 0.12em;
# #         color: var(--gold-soft);
# #         text-shadow: 0 0 18px rgba(212, 162, 76, 0.35);
# #         line-height: 1;
# #         margin-bottom: 0.3rem;
# #     }
# #     .marquee-sub {
# #         font-size: 0.95rem;
# #         color: var(--muted);
# #         letter-spacing: 0.04em;
# #         text-transform: uppercase;
# #     }

# #     /* ---------- Sidebar ---------- */
# #     section[data-testid="stSidebar"] {
# #         background: #1A1013;
# #         border-right: 1px solid var(--divider);
# #     }
# #     section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p {
# #         color: var(--cream) !important;
# #     }

# #     /* ---------- Selectbox ---------- */
# #     div[data-baseweb="select"] > div {
# #         background-color: var(--bg-card) !important;
# #         border: 1px solid var(--divider) !important;
# #         color: var(--cream) !important;
# #         border-radius: 8px;
# #     }

# #     /* ---------- Button ---------- */
# #     .stButton > button {
# #         width: 100%;
# #         background: linear-gradient(135deg, var(--gold) 0%, #B8823A 100%);
# #         color: #1A0F12;
# #         font-weight: 700;
# #         letter-spacing: 0.06em;
# #         text-transform: uppercase;
# #         font-size: 0.9rem;
# #         border: none;
# #         border-radius: 8px;
# #         padding: 0.7rem 0;
# #         transition: transform 0.15s ease, box-shadow 0.15s ease;
# #     }
# #     .stButton > button:hover {
# #         transform: translateY(-2px);
# #         box-shadow: 0 8px 20px rgba(212, 162, 76, 0.35);
# #         color: #1A0F12;
# #     }

# #     /* ---------- Movie card ---------- */
# #     .movie-card {
# #         background: var(--bg-card);
# #         border: 1px solid var(--divider);
# #         border-radius: 12px;
# #         overflow: hidden;
# #         transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
# #         height: 100%;
# #     }
# #     .movie-card:hover {
# #         transform: translateY(-6px);
# #         border-color: var(--gold);
# #         box-shadow: 0 14px 28px rgba(0, 0, 0, 0.45);
# #     }
# #     .movie-card img {
# #         width: 100%;
# #         aspect-ratio: 2 / 3;
# #         object-fit: cover;
# #         display: block;
# #         border-bottom: 1px solid var(--divider);
# #     }
# #     .movie-card-body {
# #         padding: 0.85rem 0.9rem 1rem 0.9rem;
# #     }
# #     .movie-title {
# #         font-weight: 600;
# #         font-size: 0.95rem;
# #         line-height: 1.25;
# #         margin-bottom: 0.35rem;
# #         min-height: 2.4em;
# #     }
# #     .movie-rating {
# #         color: var(--gold);
# #         font-size: 0.85rem;
# #         letter-spacing: 0.02em;
# #     }

# #     .section-label {
# #         font-family: 'Bebas Neue', sans-serif;
# #         font-size: 1.4rem;
# #         letter-spacing: 0.08em;
# #         color: var(--gold-soft);
# #         border-left: 3px solid var(--gold);
# #         padding-left: 0.6rem;
# #         margin: 0.5rem 0 1.2rem 0;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True,
# # )

# # # ----------------------------------------------------------------------------
# # # Data loading
# # # ----------------------------------------------------------------------------
# # @st.cache_data
# # def load_data():
# #     movies_dict = pickle.load(open("./models/movie_dict.pkl", "rb"))
# #     movies_df = pd.DataFrame(movies_dict)
# #     similarity_matrix = pickle.load(open("./models/similarity.pkl", "rb"))
# #     return movies_df, similarity_matrix


# # @st.cache_data(show_spinner=False)
# # def fetch_poster(movie_id):
# #     try:
# #         url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
# #         response = requests.get(url, timeout=10)
# #         response.raise_for_status()
# #         data = response.json()
# #         poster_path = data.get("poster_path")
# #         if not poster_path:
# #             return PLACEHOLDER_POSTER
# #         return "https://image.tmdb.org/t/p/w500" + poster_path
# #     except (requests.RequestException, ValueError):
# #         return PLACEHOLDER_POSTER


# # def star_rating(score_out_of_10):
# #     """Render a 0-10 score as a 5-star unicode string."""
# #     if score_out_of_10 is None:
# #         return "—"
# #     stars = round((score_out_of_10 / 10) * 5)
# #     return "★" * stars + "☆" * (5 - stars) + f"  {score_out_of_10:.1f}"


# # def recommend(movie, movies_df, similarity_matrix, n=5):
# #     movie_index = movies_df[movies_df["title"] == movie].index[0]
# #     distances = similarity_matrix[movie_index]
# #     ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : n + 1]

# #     results = []
# #     for i, _ in ranked:
# #         row = movies_df.iloc[i]
# #         rating = row["weighted_rating"] if "weighted_rating" in movies_df.columns else row.get("vote_average")
# #         results.append(
# #             {
# #                 "title": row["title"],
# #                 "movie_id": row["movie_id"],
# #                 "rating": rating,
# #             }
# #         )
# #     return results


# # # ----------------------------------------------------------------------------
# # # App
# # # ----------------------------------------------------------------------------
# # movies, similarity = load_data()

# # with st.sidebar:
# #     st.markdown("## 🎬 CineMatch")
# #     st.markdown(
# #         "A content-based recommender that finds movies similar to one you already love, "
# #         "using overview text, genres, cast, and crew."
# #     )
# #     st.markdown("---")
# #     st.markdown(f"**Titles in catalog:** {len(movies):,}")
# #     st.markdown("**Method:** TF-IDF / cosine similarity")
# #     st.markdown("---")
# #     st.caption("Built with Streamlit · Posters via TMDB API")

# # st.markdown(
# #     """
# #     <div class="marquee-wrap">
# #         <div class="marquee-title">🎬 CINEMATCH</div>
# #         <div class="marquee-sub">Find your next favorite film</div>
# #     </div>
# #     """,
# #     unsafe_allow_html=True,
# # )

# # st.markdown('<div class="section-label">PICK A MOVIE YOU LIKE</div>', unsafe_allow_html=True)

# # col_select, col_button = st.columns([4, 1])
# # with col_select:
# #     selected_movie_name = st.selectbox(
# #         "Select movie", movies["title"].values, label_visibility="collapsed"
# #     )
# # with col_button:
# #     recommend_clicked = st.button("Recommend")

# # if recommend_clicked:
# #     with st.spinner("Rolling the film reel..."):
# #         picks = recommend(selected_movie_name, movies, similarity, n=5)
# #         for p in picks:
# #             p["poster"] = fetch_poster(p["movie_id"])

# #     st.markdown('<div class="section-label">BECAUSE YOU WATCHED THIS</div>', unsafe_allow_html=True)

# #     cols = st.columns(5)
# #     for col, movie in zip(cols, picks):
# #         with col:
# #             st.markdown(
# #                 f"""
# #                 <div class="movie-card">
# #                     <img src="{movie['poster']}" alt="{movie['title']}" />
# #                     <div class="movie-card-body">
# #                         <div class="movie-title">{movie['title']}</div>
# #                         <div class="movie-rating">{star_rating(movie['rating'])}</div>
# #                     </div>
# #                 </div>
# #                 """,
# #                 unsafe_allow_html=True,
# #             )
# # else:
# #     st.info("Pick a movie above and hit **Recommend** to see similar titles.")



# import pickle

# import pandas as pd
# import requests
# import streamlit as st

# # ----------------------------------------------------------------------------
# # Config
# # ----------------------------------------------------------------------------
# TMDB_API_KEY = "33e1f99be03fafa06033cca28a4c8e47"  # move to st.secrets["tmdb_api_key"] before sharing this repo
# PLACEHOLDER_POSTER = "https://placehold.co/500x750/0b0e14/5ec9c9?text=No+Poster"
# DEFAULT_RECOMMENDATIONS = 10
# MIN_RECOMMENDATIONS = 3
# MAX_RECOMMENDATIONS = 20

# st.set_page_config(
#     page_title="CineMatch — Movie Recommender",
#     page_icon="🎬",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ----------------------------------------------------------------------------
# # Styling — dark slate + cyan/teal accent theme
# # ----------------------------------------------------------------------------
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

#     :root {
#         --bg: #0B0E14;
#         --bg-card: #131722;
#         --accent: #5EC9C9;
#         --accent-soft: #8EDCDC;
#         --cream: #E7ECF3;
#         --muted: #7C8698;
#         --divider: #232A38;
#     }

#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#         color: var(--cream);
#     }

#     .stApp {
#         background: radial-gradient(circle at 50% -10%, #161B29 0%, #0B0E14 55%) fixed;
#     }

#     /* Hide default Streamlit chrome for a cleaner marquee feel,
#        but keep the sidebar collapse/expand arrow usable */
#     #MainMenu, footer {visibility: hidden;}
#     header {
#         visibility: hidden;
#         height: 0;
#     }
#     div[data-testid="collapsedControl"] {
#         visibility: visible !important;
#         display: flex !important;
#         position: fixed;
#         top: 0.6rem;
#         left: 0.6rem;
#         z-index: 999999;
#     }
#     div[data-testid="collapsedControl"] svg {
#         fill: var(--accent) !important;
#     }

#     /* ---------- Marquee header ---------- */
#     .marquee-wrap {
#         text-align: center;
#         padding: 1.6rem 0 1.2rem 0;
#         border-bottom: 1px solid var(--divider);
#         margin-bottom: 2rem;
#     }
#     .marquee-title {
#         font-family: 'Bebas Neue', sans-serif;
#         font-size: 4.2rem;
#         letter-spacing: 0.12em;
#         color: var(--accent-soft);
#         text-shadow: 0 0 18px rgba(94, 201, 201, 0.35);
#         line-height: 1;
#         margin-bottom: 0.3rem;
#     }
#     .marquee-sub {
#         font-size: 0.95rem;
#         color: var(--muted);
#         letter-spacing: 0.04em;
#         text-transform: uppercase;
#     }

#     /* ---------- Sidebar ---------- */
#     section[data-testid="stSidebar"] {
#         background: #0F131C;
#         border-right: 1px solid var(--divider);
#     }
#     section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p {
#         color: var(--cream) !important;
#     }

#     /* ---------- Selectbox ---------- */
#     div[data-baseweb="select"] > div {
#         background-color: var(--bg-card) !important;
#         border: 1px solid var(--divider) !important;
#         color: var(--cream) !important;
#         border-radius: 8px;
#     }

#     /* ---------- Button ---------- */
#     .stButton > button {
#         width: 100%;
#         background: linear-gradient(135deg, var(--accent) 0%, #2E9E9E 100%);
#         color: #071012;
#         font-weight: 700;
#         letter-spacing: 0.06em;
#         text-transform: uppercase;
#         font-size: 0.9rem;
#         border: none;
#         border-radius: 8px;
#         padding: 0.7rem 0;
#         transition: transform 0.15s ease, box-shadow 0.15s ease;
#     }
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 20px rgba(94, 201, 201, 0.35);
#         color: #071012;
#     }

#     /* ---------- Movie card ---------- */
#     .movie-card {
#         background: var(--bg-card);
#         border: 1px solid var(--divider);
#         border-radius: 12px;
#         overflow: hidden;
#         transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
#         height: 100%;
#     }
#     .movie-card:hover {
#         transform: translateY(-6px);
#         border-color: var(--accent);
#         box-shadow: 0 14px 28px rgba(0, 0, 0, 0.55);
#     }
#     .movie-card img {
#         width: 100%;
#         aspect-ratio: 2 / 3;
#         object-fit: cover;
#         display: block;
#         border-bottom: 1px solid var(--divider);
#     }
#     .movie-card-body {
#         padding: 0.85rem 0.9rem 1rem 0.9rem;
#     }
#     .movie-title {
#         font-weight: 600;
#         font-size: 0.95rem;
#         line-height: 1.25;
#         margin-bottom: 0.35rem;
#         min-height: 2.4em;
#     }
#     .movie-rating {
#         color: var(--accent);
#         font-size: 0.85rem;
#         letter-spacing: 0.02em;
#     }

#     .section-label {
#         font-family: 'Bebas Neue', sans-serif;
#         font-size: 1.4rem;
#         letter-spacing: 0.08em;
#         color: var(--accent-soft);
#         border-left: 3px solid var(--accent);
#         padding-left: 0.6rem;
#         margin: 0.5rem 0 1.2rem 0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ----------------------------------------------------------------------------
# # Data loading
# # ----------------------------------------------------------------------------
# @st.cache_data
# def load_data():
#     movies_dict = pickle.load(open("./models/movie_dict.pkl", "rb"))
#     movies_df = pd.DataFrame(movies_dict)
#     similarity_matrix = pickle.load(open("./models/similarity.pkl", "rb"))
#     return movies_df, similarity_matrix


# @st.cache_data(show_spinner=False)
# def fetch_poster(movie_id):
#     try:
#         url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
#         poster_path = data.get("poster_path")
#         if not poster_path:
#             return PLACEHOLDER_POSTER
#         return "https://image.tmdb.org/t/p/w500" + poster_path
#     except (requests.RequestException, ValueError):
#         return PLACEHOLDER_POSTER


# def star_rating(score_out_of_10):
#     """Render a 0-10 score as a 5-star unicode string."""
#     if score_out_of_10 is None:
#         return "—"
#     stars = round((score_out_of_10 / 10) * 5)
#     return "★" * stars + "☆" * (5 - stars) + f"  {score_out_of_10:.1f}"


# def recommend(movie, movies_df, similarity_matrix, n=DEFAULT_RECOMMENDATIONS):
#     movie_index = movies_df[movies_df["title"] == movie].index[0]
#     distances = similarity_matrix[movie_index]
#     ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : n + 1]

#     results = []
#     for i, _ in ranked:
#         row = movies_df.iloc[i]
#         rating = row["weighted_rating"] if "weighted_rating" in movies_df.columns else row.get("vote_average")
#         results.append(
#             {
#                 "title": row["title"],
#                 "movie_id": row["movie_id"],
#                 "rating": rating,
#             }
#         )
#     return results


# # ----------------------------------------------------------------------------
# # App
# # ----------------------------------------------------------------------------
# movies, similarity = load_data()

# with st.sidebar:
#     st.markdown("## 🎬 CineMatch")
#     st.markdown(
#         "A content-based recommender that finds movies similar to one you already love, "
#         "using overview text, genres, cast, and crew."
#     )
#     st.markdown("---")
#     st.markdown(f"**Titles in catalog:** {len(movies):,}")
#     st.markdown("**Method:** TF-IDF / cosine similarity")
#     st.markdown("---")
#     num_recommendations = st.slider(
#         "Number of recommendations",
#         min_value=MIN_RECOMMENDATIONS,
#         max_value=MAX_RECOMMENDATIONS,
#         value=DEFAULT_RECOMMENDATIONS,
#         step=1,
#     )
#     st.markdown("---")
#     st.caption("Built with Streamlit · Posters via TMDB API")

# st.markdown(
#     """
#     <div class="marquee-wrap">
#         <div class="marquee-title">🎬 CINEMATCH</div>
#         <div class="marquee-sub">Find your next favorite film</div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown('<div class="section-label">PICK A MOVIE YOU LIKE</div>', unsafe_allow_html=True)

# col_select, col_button = st.columns([4, 1])
# with col_select:
#     selected_movie_name = st.selectbox(
#         "Select movie", movies["title"].values, label_visibility="collapsed"
#     )
# with col_button:
#     recommend_clicked = st.button("Recommend")

# if recommend_clicked:
#     with st.spinner("Rolling the film reel..."):
#         picks = recommend(selected_movie_name, movies, similarity, n=num_recommendations)
#         for p in picks:
#             p["poster"] = fetch_poster(p["movie_id"])

#     st.markdown('<div class="section-label">BECAUSE YOU WATCHED THIS</div>', unsafe_allow_html=True)

#     # Render in rows of 5 so 10 recommendations wrap onto two rows
#     row_size = 5
#     for row_start in range(0, len(picks), row_size):
#         row_picks = picks[row_start : row_start + row_size]
#         cols = st.columns(row_size)
#         for col, movie in zip(cols, row_picks):
#             with col:
#                 st.markdown(
#                     f"""
#                     <div class="movie-card">
#                         <img src="{movie['poster']}" alt="{movie['title']}" />
#                         <div class="movie-card-body">
#                             <div class="movie-title">{movie['title']}</div>
#                             <div class="movie-rating">{star_rating(movie['rating'])}</div>
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )
# else:
#     st.info("Pick a movie above and hit **Recommend** to see similar titles.")
