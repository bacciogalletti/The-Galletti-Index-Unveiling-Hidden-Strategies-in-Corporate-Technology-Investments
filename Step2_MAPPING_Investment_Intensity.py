import os
import json
import pandas as pd
import spacy
from fuzzywuzzy import fuzz
import openai
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt
from multiprocessing import Pool, cpu_count
import sys

# Global variables to be shared across worker processes
definition_embeddings = None
tech_definitions_df = None
nlp = None

# Function to initialize worker processes
def init_worker(definition_embeddings_param, tech_definitions_df_param, openai_api_key):
    global definition_embeddings
    global tech_definitions_df
    global nlp

    definition_embeddings = definition_embeddings_param
    tech_definitions_df = pd.DataFrame(tech_definitions_df_param)
    nlp = spacy.load('en_core_web_md')
    openai.api_key = openai_api_key

# Function to cache OpenAI embeddings
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_openai_embedding(text, cache_file="openai_embeddings_cache.json"):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            cache = {}
    else:
        cache = {}

    if text in cache:
        return cache[text]

    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )

    embedding = response['data'][0]['embedding']

    cache[text] = embedding
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    return embedding

# Function to compute combined similarity
def compute_combined_similarity(row_dict):
    try:
        row = pd.Series(row_dict)

        top_keywords = row['Top_Keywords']
        keyword_list = eval(top_keywords)
        keyword_text = ' '.join([kw[0] for kw in keyword_list])
        keyword_doc = nlp(keyword_text)

        similarities = {}

        # Getting OpenAI embedding for the patent keywords
        keyword_embedding = get_openai_embedding(keyword_text)

        for tech, definition in zip(tech_definitions_df['Technology'], tech_definitions_df['Definition']):
            # Computing SpaCy semantic similarity
            definition_doc = nlp(definition)
            spacy_similarity = keyword_doc.similarity(definition_doc)

            # Computing FuzzyWuzzy similarity
            fuzzy_similarity = fuzz.token_set_ratio(keyword_text, definition) / 100.0

            # Getting precomputed OpenAI embedding for the technology definition
            definition_embedding = definition_embeddings[definition]

            # Computing cosine similarity between OpenAI embeddings
            openai_similarity = cosine_similarity([keyword_embedding], [definition_embedding])[0][0]

            # Combining all three scores using specified weights
            combined_similarity = (
                (0.33 * spacy_similarity) +
                (0.33 * fuzzy_similarity) +
                (0.33 * openai_similarity)
            )

            similarities[tech] = combined_similarity

        result = {
            'Patent Title': row['Title']
        }

        # Adding each technology and its similarity score to the result
        result.update(similarities)

        return result

    except Exception as e:
        logging.error(f"Error processing row {row.get('Title', 'Unknown')}: {e}")
        return None

def precompute_openai_embeddings(definitions):
    embedding_cache = {}
    for definition in tqdm(definitions, desc="Precomputing OpenAI embeddings"):
        embedding_cache[definition] = get_openai_embedding(definition)
    return embedding_cache

def main():
    # Logging any errors encountered during processing
    logging.basicConfig(level=logging.ERROR)

    # Setting OpenAI API key (hidden for confidentiality)
    openai.api_key = "XXX"

    # Loading the dataset of patents
    print("Loading patents dataset...")
    patents_file_path = '/Users/baccio.galletti/Desktop/Thesis material/2024/patents_with_keywords_2024.csv'
    df = pd.read_csv(patents_file_path, encoding='ISO-8859-1')
    print(f"Loaded {len(df)} patents.")
    sys.stdout.flush()

    # Loading the technology definitions
    print("Loading technology definitions...")
    definitions_file_path = '/Users/baccio.galletti/Desktop/Thesis material/2024/technology_definitions_2024.csv'
    tech_definitions_df_local = pd.read_csv(definitions_file_path)
    print(f"Loaded {len(tech_definitions_df_local)} technology definitions.")
    sys.stdout.flush()

    # Precomputing and cache OpenAI embeddings for all technology definitions
    print("Precomputing OpenAI embeddings for all technology definitions...")
    definition_embeddings_local = precompute_openai_embeddings(tech_definitions_df_local['Definition'].tolist())
    print("Completed precomputing OpenAI embeddings.")
    sys.stdout.flush()

    # Preparing data for multiprocessing
    patents_to_process = [row.to_dict() for _, row in df.iterrows()]

    # Converting tech_definitions_df_local to dictionary to make it picklable
    tech_definitions_df_dict = tech_definitions_df_local.to_dict(orient='list')

    # Setting up multiprocessing pool
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for multiprocessing.")
    sys.stdout.flush()

    with Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(definition_embeddings_local, tech_definitions_df_dict, openai.api_key)
    ) as pool:
        # Processing patents in parallel with a progress bar
        similarity_results = []
        for result in tqdm(
            pool.imap_unordered(compute_combined_similarity, patents_to_process),
            total=len(patents_to_process),
            desc="Processing Patents"
        ):
            if result:
                similarity_results.append(result)

    # Converting the results to a DataFrame
    similarity_df = pd.DataFrame(similarity_results)

    # Saving the results to a CSV file
    output_file_path = '/Users/baccio.galletti/Desktop/Thesis material/2024/MAPPING_COMPLETE_2024.csv'
    similarity_df.to_csv(output_file_path, index=False)

    print(f"Mapping completed and saved to {output_file_path}")

if __name__ == '__main__':
    main()