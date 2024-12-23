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
client_id = '7f639bf9d989414aa6af202b0b27edff'
client_secret = 'ea33b989a84841949af25f8fa7bca64a'
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

def find_exact_path(graph, start, end, num_songs, current_path=[]):
    """
    Finds the exact path between two nodes in a graph, considering the number of desired songs.

    Args:
        graph: The graph to search.
        start: The starting node.
        end: The ending node.
        num_songs: The desired number of songs in the path.
        current_path: The current path being explored.

    Returns:
        A list of nodes representing the path, or None if no path is found.
    """

    current_path.append(start)

    if len(current_path) == num_songs and current_path[-1] == end:
        return current_path

    for neighbor in graph.neighbors(start):
        if neighbor not in current_path:
            new_path = find_exact_path(graph, neighbor, end, num_songs, current_path.copy())
            if new_path:
                return new_path

    # If no path is found, try to find the closest nodes to the start and end
    if len(current_path) < num_songs:
        distances_A = nx.shortest_path_length(graph, source=start, weight='weight')
        distances_B = nx.shortest_path_length(graph, source=end, weight='weight')

        sorted_nodes_A = sorted(distances_A.items(), key=lambda x: x[1])[:num_songs // 2]
        sorted_nodes_B = sorted(distances_B.items(), key=lambda x: x[1])[:num_songs // 2]

        closest_nodes = [node for node, _ in sorted_nodes_A] + [node for node, _ in sorted_nodes_B]

        return [start] + closest_nodes + [end]

    current_path.pop()
    return None

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
                    closest_songs = find_exact_path(G, start_track_id, end_track_id, num_songs)
                    track_names = [get_track_info(track_id, sp)[0] for track_id in closest_songs]
                    st.write('Playlist Tracks:', track_names)
                    playlist_id = create_spotify_playlist(user_id, 'Generated Playlist', closest_songs, sp)
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
