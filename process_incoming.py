# import requests
# import os
# import json
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import requests


# def create_embedding(text_list, model="bge-m3"):
#     url = "http://localhost:11434/api/embed"
#     r = requests.post(url, json={"model": model, "input": text_list})
#     r.raise_for_status()
#     response = r.json()
#     return response.get("embeddings", [])

# df=joblib.load('embeddings.joblib')

# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 

# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# # print(similarities)
# top_results = 3
# max_indx = similarities.argsort()[::-1][0:top_results]
# # print(max_indx)
# new_df = df.loc[max_indx] 
# # print(new_df[["number", "text"]])
# prompt=f'''i am teaching webdevlopment using sigma web devlopment course.here are video
# subtitle chunks containg video title , video number , start time in seconds, end time in 
# seconds, the text at that time
#  {new_df[["number","start","end","text"]].to_json()}
# -----------------------------------------------------
# "{incoming_query}"
# user askes]d this question related to the video chunks ,you havr to answer where and how much 
# content is taught in which video(in which video and at what timestam) and guide the user to go 
# to that particular video .if user asks unrealeted question,tell him that you can only answer
# questioons related to the couurse
# '''
# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# response = inference(prompt)["response"]
# print(response)
# # df = pd.DataFrame.from_records(my_dicts)
# # print(df)

# # for index ,item in new_df.iterrows():
# #     print(index,item["number"], item["text"],item["start"],item["end"])
    
# with open("prompt.txt","w")as f:
#     f.write(prompt)

import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])