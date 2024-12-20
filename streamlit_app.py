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
from streamlit.components.v1 import html


# Replace with your Spotify API credentials
client_id = '7f639bf9d989414aa6af202b0b27edff'
client_secret = 'ea33b989a84841949af25f8fa7bca64a'
redirect_uri = 'https://monografia-ufmg-lucas.streamlit.app/'  # Replace with your redirect URI

scope = 'user-read-private user-library-read playlist-modify-public playlist-modify-private'


def authenticate():
    sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        show_dialog=True  # Show dialog for each user
    )

    auth_url = sp_oauth.get_authorize_url()

    st.button("Connect Spotify Account", on_click=open_page, args=auth_url)  # Add a button

    code = st.query_params.get("code")  # Use experimental_get_query_params
    if code:
        token_info = sp_oauth.get_access_token(code[0])  # Access the first element of the list
        st.session_state['access_token'] = token_info['access_token']
        st.session_state['token_expiry'] = time.time() + token_info['expires_in']
        st.success("Successfully authenticated with Spotify!")

def open_page(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)

def is_token_expired():
    if 'token_expiry' in st.session_state:
        return time.time() > st.session_state['token_expiry']
    return True

def refresh_token():
    # Implement logic to refresh the token using the refresh token 
    # stored in session state
    # ... (e.g., using the `sp_oauth.refresh_access_token()` method)
    # Update the access token and expiration time in the session state
    st.session_state['access_token'] = new_access_token  # Replace with the actual new token
    st.session_state['token_expiry'] = new_expiration_time  # Replace with the actual new expiration time

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

def main():
    authenticate()

    if 'access_token' in st.session_state:
        if is_token_expired():
            refresh_token()

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
