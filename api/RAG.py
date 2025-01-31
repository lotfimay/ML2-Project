import torch
from pinecone import Pinecone
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModel
)
from dotenv import load_dotenv

load_dotenv()
# Configuration Pinecone
PINECONE_API_KEY = "pcsk_5zmUoA_BTvfm3rkJd6H4i7youfn8nQSdZYryD2bgeGpKEMTRhm6wQ6dKFQ7x6bAMkAuci7"
index_name = "ml2"                         

# Initialisation de Pinecone avec  API
pc = Pinecone(api_key=PINECONE_API_KEY)


# Nom de l'index déjà créé par votre ami
index_name = "ml2"

# Connexion à l'index existant
index = pc.Index(index_name)

model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
camembert_model = AutoModel.from_pretrained(model_name)

def get_embedding(text: str):
    """
    Convertit un texte en embedding de taille 768
    via le [CLS] token de CamemBERT.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = camembert_model(**inputs)
    emb = output.last_hidden_state[:, 0, :].squeeze().numpy()  # (768,)
    return emb

def rerank_with_msmarco(query, retrieved_results, base_path, max_length=512):
    import os, re, torch
    from tqdm import tqdm

    rerank_scores = []

    for result in tqdm(retrieved_results, desc="Reranking Results"):
        raw_path = result['metadata'].get('source', '')
        if not raw_path:
            continue

        # Nettoyage du chemin
        cleaned_path = re.sub(r'^\.?[\\/]+', '', raw_path)
        cleaned_path = cleaned_path.replace('\\', '/')

        full_path = os.path.join(base_path, cleaned_path)

        # Lecture du fichier
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()
        except Exception as e:
            print(f"Erreur en lisant '{full_path}': {e}")
            continue

        if not doc_text.strip():
            continue

        # Encode la paire (query, doc_text)
        inputs = rerank_tokenizer.encode_plus(
            query,
            doc_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        with torch.no_grad():
            outputs = rerank_model(**inputs)
            # Le modèle cross-encoder/ms-marco-MiniLM-L-6-v2 renvoie un unique logit
            logits = outputs.logits  # shape: [1, 1]
            score = logits.squeeze().item()  # on récupère le scalaire

        rerank_scores.append((score, result))

    # Trier par score décroissant
    reranked_results = sorted(rerank_scores, key=lambda x: x[0], reverse=True)
    return [doc for (score, doc) in reranked_results]


def build_context(reranked_docs, base_path, max_files=3, max_chars=2000):
    """
    Construit un 'context' brut en lisant les fichiers (Markdown) indiqués dans reranked_docs.

    Paramètres:
    -----------
    reranked_docs : list
        Liste de documents (ex: renvoyés par Pinecone),
        chaque doc doit contenir doc['metadata']['source'] (un chemin .md).
    base_path : str
        Dossier racine où se trouvent vos fichiers (ex: "/content/drive/MyDrive/code_civil").
    max_files : int
        Nombre max de fichiers à concaténer dans le contexte.
    max_chars : int
        Nombre max de caractères lus par fichier (pour éviter un contexte trop lourd).

    Retour:
    -------
    context : str
        Un texte brut non nettoyé, simplement concaténé.
    """
    import os
    import re

    used_paths = set()
    context = ""
    files_used = 0

    for doc in reranked_docs:
        raw_path = doc["metadata"].get("source", "")
        if not raw_path:
            continue

        # Nettoyer le chemin relatif (retirer "./", etc.)
        # par ex: "./livre_ier\\titre_viii\\article_349.md" -> "livre_ier/titre_viii/article_349.md"
        clean_path = re.sub(r'^\.?[\\/]+', '', raw_path)
        clean_path = clean_path.replace('\\', '/')

        full_path = os.path.join(base_path, clean_path)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except Exception as e:
            print(f"Erreur en lisant '{full_path}': {e}")
            continue

        # Tronquer si trop volumineux
        raw_content = raw_content[:max_chars]

        # On ajoute un séparateur "=== Source: ... ===" pour repérer chaque fichier
        context += f"\n=== Source: {clean_path} ===\n{raw_content}\n"

        used_paths.add(clean_path)
        files_used += 1
        if files_used >= max_files:
            break

    return context

def clean_context(context: str) -> str:
    """
    Nettoie le contexte pour ne garder que le texte des articles de loi.

    - Supprime le front matter (--- ... ---).
    - Extrait uniquement les articles définis par <h1>Article ...</h1> et leur texte associé.
    - Ignore les balises HTML restantes et supprime les références inutiles.

    Paramètre:
    ----------
    context : str
        Le texte brut issu de `build_context`.

    Retour:
    -------
    cleaned_text : str
        Le texte contenant uniquement les articles de loi, sous la forme :
        "Article XXXX\nContenu de l'article..."
    """
    import re
    from bs4 import BeautifulSoup

    # 1) Supprimer le front matter (--- ... ---)
    context = re.sub(r'---.*?---', '', context, flags=re.DOTALL)

    # 2) Analyser le HTML avec BeautifulSoup
    soup = BeautifulSoup(context, "html.parser")

    # 3) Extraire les articles
    articles = []
    for h1 in soup.find_all("h1"):
        if "Article" in h1.get_text():  # Vérifie si c'est bien un titre d'article
            article_title = h1.get_text(strip=True)  # "Article 1302-1"

            # Trouver le texte immédiatement après <h1>
            next_element = h1.find_next_sibling(string=True)  # Utilise `string=True` pour éviter le warning

            if next_element:
                article_text = next_element.strip()
                articles.append(f"{article_title}\n{article_text}")

    # 4) Concaténer les articles extraits
    cleaned_text = "\n\n".join(articles)

    return cleaned_text

# Génération (Flan-T5-Large)

gen_model_name = "google/flan-t5-large"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

def generate_response(query: str, reranked_docs: list, max_new_tokens=300):
    """
    Construit le contexte en relisant les fichiers (si besoin),
    puis génère une réponse via Flan-T5-Large.
    """
    # 1) Reconstruire le "context" en lisant depuis Drive
    base_path = "./data"  # par exemple
    raw_context = build_context(reranked_docs, base_path, max_files=2)
    context = clean_context(raw_context)
    print("Contexte utilisé pour la génération :", context)
    
    # 2) Prompt
    prompt = (
        "Vous êtes un assistant juridique expérimenté en droit français.\n\n"
        "Vous avez accès à un ensemble de lois et régulations (appelé \"Contexte\").\n"
        "Veuillez répondre de manière précise, claire et naturelle à la question suivante,\n"
        "en vous basant sur les informations fournies dans le Contexte.\n\n"
        f"Question : {query}\n\n"
        f"Contexte:\n{context}\n\n"
        "Réponse :"
    )
    
    # 3) Génération
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Désactiver le sampling pour plus de cohérence
            temperature=0.5,  # Température plus basse
            top_k=50,
            top_p=0.95
        )
    
    answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Réponse :" in answer:
        answer = answer.split("Réponse :", 1)[-1].strip()
    
    return answer


def query_pinecone(query: str, top_k=5):
    """
    1) Embedding CamemBERT
    2) index.query(...) => QueryResponse
    3) On retourne results.matches => liste de ScoredVector
    """
    query_vec = get_embedding(query).tolist()
    query_resp = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )
    return query_resp.matches  # liste de ScoredVector (itérable)

# Pipeline Complète (RAG)

def full_rag_pipeline(query: str, top_k=5):
    """
    1) Récupérer (retrieval) via Pinecone => liste de ScoredVector
    2) Reranker via MS MARCO => liste triée
    3) Génération via Flan-T5 => string
    """
    # 1) Retrieval
    retrieved_results = query_pinecone(query, top_k=top_k)

    # 2) Reranking
    base_path = "./data"

    #reranked_docs = rerank_with_msmarco(query, retrieved_results, base_path)

    # 3) Génération
    final_answer = generate_response(query, retrieved_results, max_new_tokens=300)

    return final_answer

# user_question = " La négligence peut-elle être une cause de responsabilité "
# final_answer, final_docs = full_rag_pipeline(user_question, top_k=5)

# print("\n===== Réponse Générée =====")
# print(final_answer)

# print("\n===== Documents Rerankés (Top 5) =====")
# for i, doc in enumerate(final_docs, 1):
#     path = doc.metadata.get("source", "Inconnu")
#     score = doc.score
#     print(f"Document #{i} | Score = {score:.4f} | Path = {path}")
