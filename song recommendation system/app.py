from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your DataFrame and cosine similarity matrix
df = pd.read_csv(r"C:\Users\rocke\python\english_spotify_tracks_clusters.csv")  # Update with your actual CSV path
cosine_sim = np.load(r"C:\Users\rocke\python\cosine_similarity_matrix.npy")  # Update with your actual cosine similarity file path

def recommend_songs(song_name, df, cosine_sim, top_n=10):
    if song_name not in df['name'].values:
        return {"error": f"Song '{song_name}' not found in the DataFrame."}
    
    idx = df.index[df['name'] == song_name][0]
    song_cluster = df.loc[idx, 'cluster']
    cluster_indices = df[df['cluster'] == song_cluster].index
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [score for score in sim_scores if score[0] in cluster_indices]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    if len(sim_scores) == 0:
        return {"recommendations": []}
    
    song_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[song_indices][['name', 'artist', 'genre']].to_dict(orient='records')
    
    return {"recommendations": recommendations}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    song_name = request.args.get('song_name')
    top_n = int(request.args.get('top_n', 10))
    
    result = recommend_songs(song_name, df, cosine_sim, top_n)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)