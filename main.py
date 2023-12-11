import uvicorn
import numpy as np
import os
from sentence_transformers.cross_encoder import CrossEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import regex as re
import pandas as pd
import nltk
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine

nltk.download('punkt')
app = FastAPI()

tokenizer = BertTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = TFBertModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model_trained = CrossEncoder('pahri/sts-trained-lokergo')
tfidf_vectorizer1 = TfidfVectorizer(stop_words='english')
tfidf_vectorizer2 = TfidfVectorizer(stop_words='english')
scaler = MinMaxScaler()

db_username = 'root'
db_password = ''
db_host = 'localhost'
db_port = '3306'
db_name = 'test'

# Create a connection string
connection_string = f'mysql+mysqlconnector://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create a SQLAlchemy engine
con = create_engine(connection_string).connect()

# [CC] nanti sumber vacancy data nya di ubah oleh cc utk nyambungin ke DB
vacancy_data_raw = pd.read_sql_table('pekerjaan', con)

vacancy_data_raw.fillna('', inplace=True)
vacancy_data = vacancy_data_raw[vacancy_data_raw['study_requirement'] != 'BRD, FSD, UAT']
job_title = vacancy_data['job_title']
job_title_embedding = tokenizer(list(job_title), padding=True, truncation=True, return_tensors="tf")
job_title_embedding = model(**job_title_embedding)
job_title_embedding = tf.reduce_mean(job_title_embedding.last_hidden_state, axis=1)

skills_req = pd.DataFrame(vacancy_data, columns=['skills'])
job_skill_embedding = tfidf_vectorizer1.fit_transform(skills_req['skills'])


class User(BaseModel):
    user_id: str
    user_preference: str
    user_skill: str
    user_study_level: str
    user_loc: str


class User_colab(BaseModel):
    user_id: str
    user_preference: str


@app.post('/')
async def root(item: User):
    def first_rec_cosSim(job_title_embedding, job_skill_embedding, preference, skillset):
        preference_embedding = tokenizer([preference], padding=True, truncation=True, return_tensors="tf")
        preference_embedding = model(**preference_embedding)
        preference_embedding = tf.reduce_mean(preference_embedding.last_hidden_state, axis=1)
        similarities_preference = cosine_similarity(job_title_embedding, preference_embedding).flatten()

        skillset_embedding = tfidf_vectorizer1.transform([skillset])
        similarities_skill = cosine_similarity(skillset_embedding, job_skill_embedding).flatten()
        similarities_skill_dict = dict(zip(vacancy_data['job_id'], similarities_skill))
        return similarities_preference, similarities_skill, similarities_skill_dict

    def combine_similarity(similarities_preference, similarities_skill, weights):
        normalized_preference = scaler.fit_transform(similarities_preference.reshape(-1, 1))
        normalized_skill = scaler.fit_transform(similarities_skill.reshape(-1, 1))
        combined_similarity = weights[0] * normalized_preference.flatten() + weights[1] * normalized_skill.flatten()
        return combined_similarity

    def match_study_level(study_level, vacancy_data, combined_similarity):
        pattern_SD = fr".*(SD).*"
        pattern_SMP = fr".*(SMP).*"
        pattern_SMA = fr".*(SMA).*"
        pattern_Diploma = fr".*(Diploma).*"
        pattern_S1 = fr".*(S1).*"
        pattern_S2 = fr".*(S2).*"
        pattern_S3 = fr".*(S3).*"

        if re.search(pattern_SD, study_level):
            matched_rows = vacancy_data[
                vacancy_data['study_requirement'].str.match(pattern_SMA, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_Diploma, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_S1, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_S2, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_S3, case=False, na=False)
                ]
        elif re.search(pattern_SMP, study_level):
            matched_rows = vacancy_data[
                vacancy_data['study_requirement'].str.match(pattern_Diploma, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_S1, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_S2, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_S3, case=False, na=False)
                ]
        elif re.search(pattern_SMA, study_level):
            matched_rows = vacancy_data[
                vacancy_data['study_requirement'].str.match(pattern_S2, case=False, na=False) |
                vacancy_data['study_requirement'].str.match(pattern_S3, case=False, na=False)
                ]
        elif re.search(pattern_Diploma, study_level) or re.search(pattern_S1, study_level):
            matched_rows = vacancy_data[
                vacancy_data['study_requirement'].str.match(pattern_S3, case=False, na=False)
            ]

        similarity_dict = dict(zip(vacancy_data['job_id'], combined_similarity))
        selected_ids = matched_rows['job_id'].tolist()
        return selected_ids, similarity_dict

    def first_rec(weights, user_preference, user_skill, study_level, vacancy_data, top_n=20):
        similarities_preference, similarities_skill, similarities_skill_dict = first_rec_cosSim(job_title_embedding,
                                                                                                job_skill_embedding,
                                                                                                user_preference,
                                                                                                user_skill)
        combined_similarity = combine_similarity(
            similarities_preference, similarities_skill, weights
        )
        selected_ids, similarity_dict = match_study_level(study_level, vacancy_data, combined_similarity)

        # model1 first rec without sorting
        filtered_dict = {key: value for key, value in similarity_dict.items() if key not in selected_ids}

        # sorting
        filtered_dict_sorted = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))

        # ambil top N
        model1_rec = {key: filtered_dict_sorted[key] for key in list(filtered_dict_sorted)[:100]}

        # Mengambil seluruh data dari kolom 'job_id' berdasarkan kunci dalam dictionary
        selected_data = vacancy_data[vacancy_data['job_id'].isin(model1_rec.keys())]
        pairs = [[item.user_preference, job_name] for job_name in list(selected_data['job_title'])]
        similarities_preference_model2 = model_trained.predict(pairs)

        normalized_preference_model2 = scaler.fit_transform(similarities_preference_model2.reshape(-1, 1))
        similarities_skill = {key: similarities_skill_dict[key] for key in model1_rec.keys()}
        similarities_skill = np.array(list(similarities_skill.values()))

        combined_similarity2 = weights[0] * normalized_preference_model2.flatten() + weights[
            1] * similarities_skill.flatten()

        # Mengganti nilai model1_rec dengan nilai dari list combined_similarity2
        model2_rec = {key: value for key, value in zip(model1_rec.keys(), combined_similarity2)}

        # sorting
        model2_rec = dict(sorted(model2_rec.items(), key=lambda item: item[1], reverse=True))

        # ambil top N
        model2_rec = {key: model2_rec[key] for key in list(model2_rec)[:top_n]}

        return filtered_dict, model2_rec

    def display_loc_rec(keyword, vacancy_data, filtered_dict, n=20):
        pattern = fr".*({keyword}).*"
        vacancy_loc = vacancy_data[['job_id', 'location']]
        selected_rows = vacancy_loc[vacancy_loc['job_id'].isin(filtered_dict.keys())]
        matched_rows = selected_rows[selected_rows['location'].str.match(pattern, case=False, na=False)]
        filtered_dict_matched_rows = {key: filtered_dict[key] for key in matched_rows['job_id']}
        filtered_dict_sorted = dict(sorted(filtered_dict_matched_rows.items(), key=lambda item: item[1], reverse=True))
        top_n = {key: filtered_dict_sorted[key] for key in list(filtered_dict_sorted)[:n]}
        return top_n

    filtered_dict, first_rec_result = first_rec([0.5, 0.5], item.user_preference, item.user_skill,
                                                item.user_study_level,
                                                vacancy_data, 20)
    loc_rec = display_loc_rec(item.user_loc, vacancy_data, filtered_dict)
    return item.user_id, first_rec_result, loc_rec


@app.post('/colab')
async def colab(item: User_colab):
    # [CC] nanti sumber user_data nya di ubah oleh cc utk nyambungin ke DB
    user_data = pd.read_csv(
        "https://raw.githubusercontent.com/YustafKusuma/kerjago-vacancy-recommendation-system/master/data/user_data.csv")
    global_preference_embedding = tfidf_vectorizer2.fit_transform(user_data['Preference'])
    preference_embedding = tfidf_vectorizer2.transform([item.user_preference])
    similarities_preference = cosine_similarity(global_preference_embedding, preference_embedding).flatten()

    # Ambil 5 teratas
    top_n_colab = similarities_preference.argsort()[:-5 - 1:-1]

    # Buat kamus dari pasangan nilai kunci
    result_dict = dict(zip(user_data.iloc[top_n_colab]['User_ID'].tolist(), similarities_preference[top_n_colab]))

    return item.user_id, result_dict


# Starting the server
# Your can check the API documentation easily using /docs after the server is running
uvicorn.run(app, host='0.0.0.0', port=8080)
