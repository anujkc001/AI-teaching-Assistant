# 🎓 RAG-Based AI Teaching Assistant

This project is a **Retrieval-Augmented Generation (RAG)** pipeline that converts your own lecture videos into an **AI-powered teaching assistant**.  
The workflow extracts transcripts, embeds them for semantic search, and uses an LLM to answer queries based on your content.
## 📂 Project Structure

## Step 1- Collect our videos
Move all video files to the videos folder

## Step 2- Convert to mp3
Convert all the video files to mp3 by running video_to_mp3 

## Step 3- Convert mp3 to json
Convert all the mp3 files to json by running mp3_to_json

## Step 4- Convert the json files to vectors
Use the file preprocess_json files to a dataframe with Embeddungs and save it as a joblib pickle


## Step 5- Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create relevant prompt as per the user query and feed it to the LLM






## I am using only two mp3 files because the CPU and GPU processes are very slow






RAG_based_ai/
│── videos/                # Place your video files here
│── video_to_mp3.py        # Convert videos → mp3
│── mp3_to_json.py         # Convert mp3 → text/json
│── preprocess_json.py     # Convert json → embeddings (joblib)
│── process_incoming.py    # Main script (handles Q&A)
│── embeddings.joblib      # Saved embeddings
│── prompt.txt             # Last generated prompt
│── response.txt           # Last AI response
