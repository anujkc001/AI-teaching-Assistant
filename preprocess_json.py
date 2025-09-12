# import requests 
# import os
# import json


# def create_embeddings(text_list):
#     r=requests.post("http://localhost:11434/api/embedings", json={
#         "model":"bge-m3",
#         "input":text_list
#     })

#     embedding=r.json()['embedding']
#     return embedding



# # jsons=os.listdir("jsons")
# # print(jsons)
# # my_dicts=[]
# # chunk_id=0
# # for json_file in jsons:
# #     with open(f"jsons/{json_file}") as f:
# #         content=json.load(f)
# #     for chunk in content['chunks']:
# #         print(chunk)
# #         chunk['chunk_id']=chunk_id
# #         chunk_id +=1
# #         my_dicts.append(chunk)
# #     break



# #print(my_dicts)

# a=create_embeddings("cat sat on the mat")
# print(a)


import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list, model="bge-m3"):
    url = "http://localhost:11434/api/embed"
    r = requests.post(url, json={"model": model, "input": text_list})
    r.raise_for_status()
    response = r.json()
    return response.get("embeddings", [])

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    if "chunks" not in content:
        print(f"Skipping {json_file}, no 'chunks' found")
        continue

    print(f"Creating Embeddings for {json_file}")
    texts = [c.get('text', '') for c in content['chunks']]
    embeddings = create_embedding(texts)

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i] if i < len(embeddings) else None
        chunk_id += 1
        my_dicts.append(chunk)
    


df = pd.DataFrame.from_records(my_dicts) 
# save  this data frame 
joblib.dump(df,'embeddings.joblib')
incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
print(max_indx)
new_df = df.loc[max_indx] 
print(new_df[["number", "text"]])
# df = pd.DataFrame.from_records(my_dicts)
# print(df)
