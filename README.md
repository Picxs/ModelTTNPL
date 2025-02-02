# ModelTTNLP
Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de analisar um conjunto de dados contendo reclamações de veículos.

A base de dados utilizada contém informações detalhadas sobre reclamações de veículos registradas na NHTSA (National Highway Traffic Safety Administration). Cada registro inclui descrições de falhas, problemas relatados e outras características do incidente. A task principal é identificar, a partir desses dados, se uma reclamação específica está associada a um evento que demandou atendimento médico e com isso obter um modelo capaz de predizer se uma situação precisará do atendimento.

A ideia do projeto foi testar 3 modelos diferentes (Logistic Regression, Random Forest e XGBoost), compara-los com um Voting Classifier para ter indícios de qual melhor modelo. Entretanto os 3 modelos tiveram resultados muito semelhantes e optei por não usar Voting Classifier e fazer uma análise mais aprofundade sobre qual dos 3 modelos seria o mais apto para o contexto da task.

São duas formas de utilizar o modelo:

Acessando o terminal, onde o usuário pode rodar o script *python -m models.model* para iniciar o processo de aquisição de data, processamento dos dados e treinamento do modelo. Está maneira de rodar o modelo vai somente até a etapa onde os modelos são treinados e salvos, ele não contem a análise exploratória completa. O treinamento dessa maneira demora um pouco mais que o normal.

Pela pasta ./notebooks acessando o arquivo Relatório.ipynb, que é o relatório sobre todo o projeto, detalhando o código como um todo, e ao mesmo tempo é uma maneira mais visual de ver o funcionamento do modelo.

É possível utilizar os modelos treinados pelo notebook utilizando usando o FastAPI, use o script *python -m uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload* e acesse o link gerado. Na api será possível utilizar uma versão adaptada para os padrões da api da função test_all_models.

Um leve adendo, os diagramas mermaid não foram gerados corretamente no Relatório.ipynb, mas se o arquivo for rodado no VSCode tendo a extensão de Mermaid Preview, vai funcionar perfeitamente
