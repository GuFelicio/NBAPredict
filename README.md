# 🏀 NBA Game Prediction - Predict and Analyze Like a Pro! 🏆

Seja bem-vindo ao **NBA Game Prediction Project**! Este projeto usa aprendizado de máquina para prever o placar e a chance de vitória de jogos da NBA em tempo real, além de destacar os principais pontuadores do jogo! Vamos mergulhar nos detalhes para entender melhor como funciona! 😎

## 🚀 Funcionalidades

1. **Predição Inicial** 🔮:
   - Antes do início do jogo, o modelo gera:
     - O placar provável para cada time.
     - Os cinco jogadores que podem ser os maiores pontuadores. 🎯

2. **Atualizações em Tempo Real** ⏱️:
   - Durante o jogo, o modelo:
     - Monitora o placar em tempo real e compara com a predição inicial.
     - Atualiza a probabilidade de vitória para cada time, ajustando a cada atualização de pontuação! 💹

## 🔧 Tecnologias Utilizadas

- **Python** 🐍: A linguagem que organiza tudo.
- **Machine Learning com Scikit-Learn** 🤖: RandomForest e GradientBoosting foram treinados e validados para garantir previsões consistentes e uma boa acurácia!
- **NBA API** 🏀: Para capturar dados ao vivo diretamente dos jogos.
- **Pandas & Numpy** 📊: Manipulação e preparação de dados para alimentar o modelo.
- **TQDM** 🚀: Barra de progresso para manter o usuário informado durante as tarefas demoradas.

## 📈 Modelos Utilizados e Resultados

### Cross Validation Acurácia

- **RandomForest** 🌲: 93.97% ± 0.20%
- **GradientBoosting** 🔥: 76.83% ± 0.25%
- **Ensemble (RF + GB)** 🤝: 86.56% ± 0.21%

> **Conclusão**: O modelo Ensemble (RandomForest + GradientBoosting) mostrou uma boa capacidade preditiva para jogos completos e ao vivo!

## 🎮 Como Usar

1. **Pré-requisitos** 📦:
   - Instale as dependências com: `pip install -r requirements.txt`

2. **Execução Inicial** 🚀:
   - No terminal, rode `python src/ensemble_model.py` para treinar o modelo e gerar as predições iniciais.

3. **Predição em Tempo Real** ⏲️:
   - Use o comando `python src/live_prediction.py --game_id <GAME_ID>` substituindo `<GAME_ID>` pelo ID do jogo ao vivo para iniciar o acompanhamento!

## 📁 Estrutura do Projeto

- **src/**: Contém os scripts principais para treinamento, validação cruzada, agregação de dados e predições ao vivo.
- **data/**: Armazena os dados tratados e prontos para o modelo.
- **README.md**: Este documento explicativo do projeto. 