
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Analyse des hallucinations dans un système RAG</title>
</head>
<body>

    <h1>Analyse des hallucinations dans un système RAG</h1>

    <h2>1. Pourquoi le chatbot répond-il parfois alors que le PDF ne contient pas la réponse ?</h2>

    <p>Il y a trois causes principales :</p>

    <ul>
        <li>
            <strong>Le LLM complète avec ses connaissances internes.</strong>
            Le prompt lui demande de répondre uniquement avec le contexte,
            mais c'est une consigne, pas une garantie. Un petit modèle local
            comme Llama 3.2 la respecte mal, surtout si la question porte
            sur un sujet qu'il connaît déjà.
        </li>

        <li>
            <strong>Le retrieval ramène des chunks peu pertinents.</strong>
            La recherche vectorielle renvoie toujours les k chunks les plus
            proches, même si aucun ne contient la réponse. Le modèle reçoit
            alors un contexte hors sujet et peut en tirer une réponse
            plausible mais fausse.
        </li>

        <li>
            <strong>La formulation de la question est mal alignée avec le texte du PDF.</strong>
            Si les mots de l'utilisateur diffèrent de ceux du document,
            les embeddings peuvent rater le bon passage. Le modèle se
            retrouve sans la bonne information et improvise.
        </li>
    </ul>

    <h2>2. Quelle amélioration a le plus réduit les hallucinations ?</h2>

    <p>
        En général, c'est le <strong>reranking</strong>. Il filtre les chunks
        non pertinents avant la génération, donc le LLM reçoit un contexte
        plus propre et a moins de matière pour inventer.
    </p>

    <p>
        La <strong>query expansion</strong> aide surtout à améliorer le rappel,
        c'est-à-dire à retrouver le bon passage. Elle réduit donc les
        hallucinations de façon indirecte, mais elle peut aussi ramener
        plus de bruit, que le reranking élimine ensuite.
    </p>

    <p><strong>Les deux fonctionnent mieux ensemble.</strong></p>

    <p>
        Pour votre rapport, appuyez cette réponse sur vos propres tests.
        Posez les mêmes questions avec la case « RAG avancé » cochée puis
        décochée, y compris des questions dont la réponse n'est pas dans
        le PDF, et notez ce que vous observez.
    </p>

    <h2>3. Pourquoi ne pas simplement changer de modèle LLM ?</h2>

    <ul>
        <li>
            <strong>Un meilleur LLM ne corrige pas un mauvais contexte.</strong>
            Si le chunk contenant la réponse n'est pas récupéré, aucun modèle
            ne peut la trouver. Le problème est en amont, dans le retrieval.
        </li>

        <li>
            <strong>Le coût et les ressources.</strong>
            Un modèle plus gros demande plus de mémoire, de GPU et de temps
            de calcul, ce qui est difficile en local. Query expansion et
            reranking sont plus légers et ciblent directement la cause.
        </li>

        <li>
            <strong>Aucune garantie contre les hallucinations.</strong>
            Les grands modèles hallucinent aussi, parfois avec plus
            d'assurance, ce qui les rend plus difficiles à repérer.
        </li>

        <li>
            <strong>Le RAG sert justement à ancrer les réponses dans les documents.</strong>
            Améliorer la qualité du contexte reste la voie la plus efficace,
            quel que soit le modèle.
        </li>
    </ul>

</body>
</html>
