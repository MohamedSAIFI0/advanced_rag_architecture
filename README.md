# Analyse des hallucinations dans un système RAG

## 1. Pourquoi le chatbot répond-il parfois alors que le PDF ne contient pas la réponse ?

Il y a trois causes principales :

- **Le LLM complète avec ses connaissances internes.**
  Le prompt lui demande de répondre uniquement avec le contexte,
  mais c'est une consigne, pas une garantie.

- **Le retrieval ramène des chunks peu pertinents.**
  La recherche vectorielle renvoie toujours les k chunks les plus proches,
  même si aucun ne contient la réponse.

- **La formulation de la question est mal alignée avec le texte du PDF.**
  Les embeddings peuvent rater le bon passage si la formulation
  de l'utilisateur diffère de celle du document.

## 2. Quelle amélioration a le plus réduit les hallucinations ?

En général, c'est le **Reranking**.

Il filtre les chunks non pertinents avant la génération.

La **Query Expansion** améliore le rappel, c'est-à-dire la capacité
à retrouver le bon passage.

**Les deux fonctionnent mieux ensemble.**

### Conseil pour le rapport

Posez les mêmes questions avec la case « RAG avancé »
cochée puis décochée, y compris des questions dont la réponse
n'est pas dans le PDF.

## 3. Pourquoi ne pas simplement changer de modèle LLM ?

- **Un meilleur LLM ne corrige pas un mauvais contexte.**
- **Le coût et les ressources :** un modèle plus gros demande
  davantage de mémoire, de GPU et de temps de calcul.
- **Aucune garantie contre les hallucinations :**
  les grands modèles hallucinent aussi.
- **Le RAG sert à ancrer les réponses dans les documents.**