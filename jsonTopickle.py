import pickle, json

with open('/Users/haro/works/handwriting_authenticate/data/signature_data.json', 'r') as f:
    data = json.load(f)

with open('/Users/haro/works/handwriting_authenticate/data/signature_data.pickle', 'wb') as f:
    pickle.dump(data, f)