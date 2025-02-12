import networkx as nx
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import streamlit as st
import time
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components


# Replace with your Spotify API credentials
client_id = 'e32ae9edd98c4442b14831a2650b8149'
client_secret = '9bb07b7f2b9f4f6498f781c69700b1f6'
redirect_uri = 'https://monografia-ufmg-lucas.streamlit.app/' # Replace with your redirect URI

scope = 'user-read-private user-library-read playlist-modify-public playlist-modify-private'


def authenticate():
    sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        show_dialog=True  # Show dialog for each user
    )

    if 'access_token' not in st.session_state:
        st.write("To use this app, you need to log in with your Spotify account.")
        auth_url = sp_oauth.get_authorize_url()
            # Pass the auth_url to the open_page function using st.button's on_click
        st.button("Connect Spotify Account", on_click=open_page, args=(auth_url,))  

    code = st.experimental_get_query_params().get("code")
    if code:
        try:
            token_info = sp_oauth.get_access_token(code[0])
            st.session_state['access_token'] = token_info['access_token']
            st.session_state['refresh_token'] = token_info['refresh_token']
            st.session_state['token_expiry'] = time.time() + token_info['expires_in']
            st.success("Successfully authenticated with Spotify!")
        except spotipy.oauth2.SpotifyOauthError as e:
            st.error(f"Error authenticating with Spotify: {e}")

def open_page(url):
    """Opens a URL in a new tab using JavaScript."""
    open_script = f"""
        <script type="text/javascript">
            window.open('{url}', '_blank').focus();
        </script>
    """
    components.html(open_script)

def is_token_expired():
    if 'token_expiry' in st.session_state:
        return time.time() > st.session_state['token_expiry']
    return True

def refresh_token():
    """Refreshes the Spotify access token using the refresh token."""

    sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
    )

    if 'refresh_token' in st.session_state:
        refresh_token = st.session_state['refresh_token']

        try:
            # Refresh the access token
            new_token_info = sp_oauth.refresh_access_token(refresh_token)

            # Update session state with new token info
            st.session_state['access_token'] = new_token_info['access_token']
            st.session_state['token_expiry'] = time.time() + new_token_info['expires_in']

        except spotipy.oauth2.SpotifyOauthError as e:
            st.error(f"Error refreshing token: {e}")
            # You might want to handle the error by re-authenticating the user
            # For example:
            # del st.session_state['access_token']
            # del st.session_state['refresh_token']
            # st.experimental_rerun() 

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
        """Calculates shortest path distances from a starting node."""
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
        if song_id == song_B_id:
            dist = float('inf')  # Force end song to have the lowest rank
        else:
            dist = distances_A.get(song_id) or distances_B.get(song_id) or float('inf')

        # Calculate similarity score for song_id 
        if song_id in dfa['track_id'].values and avg_similarities is not None:
            song_index = dfa['track_id'].tolist().index(song_id)
            similarity_score = avg_similarities[0][song_index]
        else:
            similarity_score = 0  # Assign 0 similarity if song not found in dfa or avg_similarities is not calculated

        # Combine distance and similarity (adjust weights as needed)
        combined_score = 0.2 * dist + 0.8 * (1 - similarity_score)

        ranked_songs.append((song_id, combined_score))

    ranked_songs.sort(key=lambda x: x[1])  # Sort by combined score

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
                
                try:
                    closest_songs = find_closest_songs_weighted(G, start_track_id, end_track_id, num_songs - 2, sp, dfa)
                    closest_songs.insert(0, start_track_id)   # Add start track here 
                    closest_songs.append(end_track_id)
                    track_names = [get_track_info(track_id, sp)[0] for track_id in closest_songs]
                    st.write('Playlist Tracks:', track_names)
                    playlist_name = "{} - {} - {}".format(start_track, end_track, num_songs)
                    playlist_id = create_spotify_playlist(user_id, playlist_name, closest_songs, sp)
                    st.write(f'Playlist created successfully: https://open.spotify.com/playlist/{playlist_id}')

                except Exception as e:
                    st.error(f"Error finding path or creating playlist: {e}")
            else:
                st.error('Please fill in all required fields and ensure "Number of Songs" is at least 2.')
    else:
        st.write('Waiting for authentication...')

if __name__ == '__main__':
    st.title('Spotify Playlist Generator')
    main()
