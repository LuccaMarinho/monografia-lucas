import networkx as nx
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import streamlit as st
import webbrowser
import time
import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components


# Replace with your Spotify API credentials
client_id = '7f639bf9d989414aa6af202b0b27edff'
client_secret = 'ea33b989a84841949af25f8fa7bca64a'
redirect_uri = 'https://monografia-ufmg-lucas.streamlit.app/' # Replace with your redirect URI

scope = 'user-read-private user-library-read playlist-modify-public playlist-modify-private'

def get_token(oauth, code):

    token = oauth.get_access_token(code, as_dict=False, check_cache=False)
    # remove cached token saved in directory
    os.remove(".cache")
    
    # return the token
    return token


def sign_in(token):
    sp = spotipy.Spotify(auth=token)
    return sp


def app_get_token():
    try:
        token = get_token(st.session_state["oauth"], st.session_state["code"])
    except Exception as e:
        st.error("An error occurred during token retrieval!")
        st.write("The error is as follows:")
        st.write(e)
    else:
        st.session_state["cached_token"] = token


def app_sign_in():
    try:
        sp = sign_in(st.session_state["cached_token"])
    except Exception as e:
        st.error("An error occurred during sign-in!")
        st.write("The error is as follows:")
        st.write(e)
    else:
        st.session_state["signed_in"] = True
        app_display_welcome()
        st.success("Sign in success!")
        
    return sp


def app_display_welcome():
    
    # import secrets from streamlit deployment
    cid = client_id
    csecret = client_secret
    uri = redirect_uri

    # set scope and establish connection
    scopes = scope

    # create oauth object
    oauth = SpotifyOAuth(scope=scopes, redirect_uri=uri, client_id=cid, client_secret=csecret)
    # store oauth in session
    st.session_state["oauth"] = oauth

    # retrieve auth url
    auth_url = oauth.get_authorize_url()
    
    # this SHOULD open the link in the same tab when Streamlit Cloud is updated
    # via the "_self" target
    link_html = " <a target=\"_self\" href=\"{url}\" >{msg}</a> ".format(
        url=auth_url,
        msg="Click me to authenticate!"
    )
    
    # define welcome
    welcome_msg = """
    Welcome! :wave: This app uses the Spotify API to interact with general 
    music info and your playlists! In order to view and modify information 
    associated with your account, you must log in. You only need to do this 
    once.
    """
    
    # define temporary note
    note_temp = """
    _Note: Unfortunately, the current version of Streamlit will not allow for
    staying on the same page, so the authorization and redirection will open in a 
    new tab. This has already been addressed in a development release, so it should
    be implemented in Streamlit Cloud soon!_
    """

    st.title("Spotify Playlist Generator")

    if not st.session_state["signed_in"]:
        st.markdown(welcome_msg)
        st.write(" ".join(["No tokens found for this session. Please log in by",
                          "clicking the link below."]))
        st.markdown(link_html, unsafe_allow_html=True)
        st.markdown(note_temp)

def load_track_names(filename):
    df = pd.read_csv(filename, sep=";")
    return df['track_name'].to_list()

def get_track_id_from_df(track_name, df):
    track_id = df[df['track_name'] == track_name]['track_id'].values
    return track_id[0] if len(track_id) > 0 else None

def load_graph():
    return nx.read_graphml('final_spotify_graph.graphml')

def get_track_info(track_id, sp):
    track = sp.track(track_id)
    return (track['name'], track['artists'][0]['name']) if track else (None, None)

def find_closest_songs_weighted(G, song_A_id, song_B_id, X, sp, dfa):
    def dijkstra_distances(graph, start_node):
        distances = {node: float('infinity') for node in graph}
        distances[start_node] = 0
        pq = [(0, start_node)]
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_distance > distances[current_node]:
                continue
            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight['weight']
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        return distances

    distances_A = dijkstra_distances(G, song_A_id)
    distances_B = dijkstra_distances(G, song_B_id)

    common_neighbors = set(distances_A.keys()) & set(distances_B.keys())
    bridge_songs = sorted([(neighbor, distances_A[neighbor] + distances_B[neighbor]) 
                           for neighbor in common_neighbors], key=lambda x: x[1])
    bridge_songs = [song_id for song_id, dist in bridge_songs]

    similar_songs = []
    avg_similarities = None
    if len(bridge_songs) < X:
        features_A = sp.audio_features(song_A_id)[0]
        features_B = sp.audio_features(song_B_id)[0]
        all_features = [sp.audio_features(track_id)[0] for track_id in dfa['track_id']]
        similarities_A = cosine_similarity([features_A], all_features)
        similarities_B = cosine_similarity([features_B], all_features)
        avg_similarities = (similarities_A + similarities_B) / 2
        similar_song_indices = avg_similarities.argsort()[0][::-1]
        similar_songs = [dfa['track_id'].iloc[i] for i in similar_song_indices]

    all_songs = bridge_songs + similar_songs
    ranked_songs = []
    for song_id in all_songs:
        dist = distances_A.get(song_id) or distances_B.get(song_id) or float('inf')
        similarity_score = avg_similarities[0][dfa['track_id'].tolist().index(song_id)] if (
            song_id in dfa['track_id'].values and avg_similarities is not None
        ) else 0
        combined_score = 0.2 * dist + 0.8 * (1 - similarity_score)  # Adjusted weights
        ranked_songs.append((song_id, combined_score))

    ranked_songs.sort(key=lambda x: x[1])
    return [song_id for song_id, score in ranked_songs[:X]]

def create_spotify_playlist(user_id, playlist_name, track_ids, sp):
    try:
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
        sp.playlist_add_items(playlist_id=playlist['id'], items=track_ids)
        return playlist['id']
    except Exception as e:
        st.error(f"Error creating playlist: {e}")
        return None

if "signed_in" not in st.session_state:
    st.session_state["signed_in"] = False
if "cached_token" not in st.session_state:
    st.session_state["cached_token"] = ""
if "code" not in st.session_state:
    st.session_state["code"] = ""
if "oauth" not in st.session_state:
    st.session_state["oauth"] = None
    
    # %% authenticate with response stored in url
    
    
    # attempt sign in with cached token
if st.session_state["cached_token"] != "":
    sp = app_sign_in()
    # if no token, but code in url, get code, parse token, and sign in
else:
    app_display_welcome()


def main():
    if st.session_state["signed_in"]:
        sp = spotipy.Spotify(auth=st.session_state['access_token'])
        current_user = sp.current_user()
        user_id = current_user['id']
        
        dfa = pd.read_csv('track_names.csv', sep=";")
        G = load_graph()
        track_names = load_track_names('track_names.csv')
    
        start_track = st.selectbox('Start Track', track_names)
        end_track = st.selectbox('End Track', track_names)
        num_songs = st.number_input('Number of Songs', min_value=2)
    
        if st.button('Find Playlist Tracks'):
            if start_track and end_track and num_songs >= 2:
                start_track_id = get_track_id_from_df(start_track, dfa)
                end_track_id = get_track_id_from_df(end_track, dfa)
    
                if start_track_id and end_track_id:
                    try:
                        closest_songs = find_closest_songs_weighted(
                                G, start_track_id, end_track_id, num_songs - 2, sp, dfa
                            )
                        closest_songs.insert(0, start_track_id)  
                        closest_songs.append(end_track_id)
                        track_names = [get_track_info(track_id, sp)[0] for track_id in closest_songs]
                        st.write('Playlist Tracks:', track_names)
                        playlist_id = create_spotify_playlist(user_id, 'Generated Playlist', closest_songs, sp)
                        st.write(f'Playlist created successfully: https://open.spotify.com/playlist/{playlist_id}')
    
                    except Exception as e:
                        st.error(f"Error finding path or creating playlist: {e}")
                else:
                    st.error("Track not found")
            else:
                st.error('Please fill in all required fields and ensure "Number of Songs" is at least 2.')
    else:
        st.write('Waiting for authentication...')


if __name__ == '__main__':
    st.title('Spotify Playlist Generator')
    main()
