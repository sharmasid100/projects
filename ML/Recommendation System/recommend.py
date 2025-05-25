import pickle

# Load data
with open('model/similarity.pkl', 'rb') as f:
    sim = pickle.load(f)

with open('model/movies.pkl', 'rb') as f:
    df = pickle.load(f)

def recommend(movie):
    try:
        index = df[df['title'] == movie].index[0]
    except IndexError:
        return []

    rec = []
    _dist = sorted(list(enumerate(sim[index])), key=lambda x: x[1], reverse=True)
    for i in _dist[1:6]:  # top 5 recommendations excluding itself
        rec.append(df.iloc[i[0]].title)
    return rec
