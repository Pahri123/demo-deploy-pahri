from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import re
import pandas as pd
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk

app = FastAPI()

tokenizer = BertTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = TFBertModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
nltk.download('punkt')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

#nanti sumber vacancy data nya di ubah oleh cc utk nyambungin ke DB
vacancy_data_raw = pd.read_csv("https://raw.githubusercontent.com/aldrianaliv/CapstoneProject/main/glints-data%20(new).csv")

vacancy_data_raw.fillna('', inplace=True)
vacancy_data = vacancy_data_raw[vacancy_data_raw['Study_requirement'] != 'BRD, FSD, UAT']
job_title = vacancy_data['Job_title']
job_title_embedding = tokenizer(list(job_title), padding=True, truncation=True, return_tensors="tf")
job_title_output = model(**job_title_embedding)
job_title_embedding = tf.reduce_mean(job_title_output.last_hidden_state, axis=1)

skills_req = pd.DataFrame(vacancy_data, columns=['Skills'])
job_skill_embedding = tfidf_vectorizer.fit_transform(skills_req['Skills'])


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
        preference_output = model(**preference_embedding)
        preference_embedding = tf.reduce_mean(preference_output.last_hidden_state, axis=1)
        similarities_preference = cosine_similarity(job_title_embedding, preference_embedding)

        skillset_embedding = tfidf_vectorizer.transform([skillset])
        similarities_skill = linear_kernel(skillset_embedding, job_skill_embedding).flatten()
        return similarities_preference, similarities_skill

    def combine_similarity(similarities_preference, similarities_skill, weights):
        scaler = MinMaxScaler()
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
                vacancy_data['Study_requirement'].str.match(pattern_SMA, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_Diploma, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_S1, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_S2, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_S3, case=False, na=False)
                ]
        elif re.search(pattern_SMP, study_level):
            matched_rows = vacancy_data[
                vacancy_data['Study_requirement'].str.match(pattern_Diploma, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_S1, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_S2, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_S3, case=False, na=False)
                ]
        elif re.search(pattern_SMA, study_level):
            matched_rows = vacancy_data[
                vacancy_data['Study_requirement'].str.match(pattern_S2, case=False, na=False) |
                vacancy_data['Study_requirement'].str.match(pattern_S3, case=False, na=False)
                ]
        elif re.search(pattern_Diploma, study_level) or re.search(pattern_S1, study_level):
            matched_rows = vacancy_data[
                vacancy_data['Study_requirement'].str.match(pattern_S3, case=False, na=False)
            ]

        similarity_dict = dict(zip(vacancy_data['id'], combined_similarity))
        selected_ids = matched_rows['id'].tolist()
        return selected_ids, similarity_dict

    def exclude_and_display(similarity_dict, selected_ids, vacancy_data, n):
        filtered_dict = {key: value for key, value in similarity_dict.items() if key not in selected_ids}
        filtered_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
        filtered_dict_sorted = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
        top_n = {key: filtered_dict_sorted[key] for key in list(filtered_dict_sorted)[:n]}
        return filtered_dict, top_n

    def first_rec(weights, user_preference, user_skill, study_level, vacancy_data, top_n=20):
        similarities_preference, similarities_skill = first_rec_cosSim(job_title_embedding, job_skill_embedding,
                                                                       user_preference, user_skill)
        combined_similarity = combine_similarity(
            similarities_preference, similarities_skill, weights
        )
        selected_ids, similarity_dict = match_study_level(study_level, vacancy_data, combined_similarity)
        filtered_dict, first_rec = exclude_and_display(similarity_dict, selected_ids, vacancy_data, top_n)
        return filtered_dict, first_rec

    def display_loc_rec(keyword, vacancy_data, filtered_dict, n=20):
        pattern = fr".*({keyword}).*"
        vacancy_loc = vacancy_data[['id', 'Location']]
        selected_rows = vacancy_loc[vacancy_loc['id'].isin(filtered_dict.keys())]
        matched_rows = selected_rows[selected_rows['Location'].str.match(pattern, case=False, na=False)]
        filtered_dict_matched_rows = {key: filtered_dict[key] for key in matched_rows['id']}
        filtered_dict_sorted = dict(sorted(filtered_dict_matched_rows.items(), key=lambda item: item[1], reverse=True))
        top_n = {key: filtered_dict_sorted[key] for key in list(filtered_dict_sorted)[:n]}
        return top_n

    filtered_dict, first_rec_result = first_rec([0.5, 0.5], item.user_preference, item.user_skill, item.user_study_level,
                                                vacancy_data, 20)
    loc_rec = display_loc_rec(item.user_loc, vacancy_data, filtered_dict)
    return first_rec_result, loc_rec


@app.post('/colab')
async def colab(item: User_colab):
    # nanti sumber user_data nya di ubah oleh cc utk nyambungin ke DB
    user_data = pd.read_csv(
        "https://raw.githubusercontent.com/YustafKusuma/kerjago-vacancy-recommendation-system/master/data/user_data.csv")
    global_preference_embedding = tfidf_vectorizer.fit_transform(user_data['Preference'])
    preference_embedding = tfidf_vectorizer.transform([item.user_preference])
    similarities_preference = linear_kernel(global_preference_embedding, preference_embedding).flatten()

    # Ambil 5 teratas
    top_n_colab = similarities_preference.argsort()[:-5 - 1:-1]

    # Buat kamus dari pasangan nilai kunci
    result_dict = dict(zip(user_data.iloc[top_n_colab]['User_ID'].tolist(), similarities_preference[top_n_colab]))

    return item.user_id, result_dict

