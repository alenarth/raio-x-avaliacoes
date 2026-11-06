# 📊 Raio-X de Avaliações

Dashboard que lê **todas as avaliações de um restaurante de delivery com IA** e responde em segundos o que o dono levaria horas para descobrir: **o que mais gera reclamação, o que está piorando e o que fazer primeiro.**

**Demo:** [SEU_LINK.streamlit.app](https://SEU_LINK.streamlit.app)

![tests](https://github.com/alenarth/raio-x-avaliacoes/actions/workflows/tests.yml/badge.svg)

## O problema

Um restaurante recebe dezenas de avaliações por semana. Na prática, ninguém lê todas: o dono olha a nota média e algumas reclamações soltas. Assim, um problema que está crescendo (ex.: atrasos na entrega desde que trocou de motoboy) passa semanas sem ser notado.

## O que a ferramenta faz

1. **Recebe um CSV** de avaliações de qualquer fonte (app de delivery, Google, planilha) e deixa você escolher qual coluna é o texto, a nota e a data.
2. **Classifica cada avaliação com um LLM:** os assuntos citados (comida, entrega, embalagem, preço, atendimento, pedido errado) com o **sentimento sobre cada assunto**, mais o tom geral e se é um caso urgente.
3. **Mostra o painel:**
   - indicadores (% negativas, nota média, casos urgentes, tempo economizado);
   - ranking de reclamações por assunto;
   - **tendência semanal**, para ver o que está piorando;
   - lista de casos urgentes para responder hoje;
   - tabela filtrável e download do CSV classificado.
4. **Gera um plano de ação** com 3 a 5 prioridades, cada uma com a evidência nos dados, uma ação para a semana e como medir se funcionou.

### Exemplo (dados fictícios incluídos)

O CSV de exemplo é de um restaurante **inventado**, com 152 avaliações em 8 semanas e uma história escondida: os atrasos na entrega pioram nas últimas 3 semanas. O painel mostra isso logo na aba **Tendência**, e o plano de ação coloca a entrega como prioridade.

## Decisões técnicas

- **Sentimento por assunto, não só por avaliação.** "Comida ótima, mas atrasou 1 hora" é elogio para a comida e reclamação da entrega. Classificar só o tom geral esconderia exatamente a reclamação que importa.
- **Saída estruturada com schema (Pydantic)** e envio em lotes de 25 avaliações: menos chamadas, menos custo e nenhuma resposta fora do formato.
- **A IA não inventa o que pulou:** se o modelo deixar de responder alguma avaliação, ela aparece como "não classificada" e o painel avisa, em vez de entrar com um chute.
- **O plano de ação recebe só números agregados e resumos**, nunca a base inteira: fica mais barato, mais rápido e o modelo é instruído a não inventar números.
- **Provedor de LLM atrás de uma interface:** os testes usam um LLM falso e rodam sem chave nem internet (GitHub Actions).

## Avaliação

O exemplo fictício tem gabarito com os assuntos e sentimentos corretos de cada avaliação (gerado com semente fixa por `scripts/generate_example.py`, igual em qualquer máquina). O botão **Rodar avaliação** na barra lateral da demo, ou o script `python -m scripts.evaluate`, mede o classificador:

| Métrica | Valor |
|---|---|
| Assuntos identificados (F1) | _preencher_ |
| Sentimento por assunto correto | _preencher_ |
| Sentimento geral correto | _preencher_ |
| Casos urgentes encontrados | _preencher_ |
| Tempo para 152 avaliações | _preencher_ |


## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt

# chave gratuita em https://aistudio.google.com/apikey
# Windows (PowerShell): $env:GEMINI_API_KEY="sua_chave"
# Linux/Mac:            export GEMINI_API_KEY="sua_chave"

streamlit run app.py          # abre o painel (gera o exemplo fictício na primeira vez)
python -m scripts.evaluate    # mede a qualidade do classificador
pytest -q                     # testes (não precisam de chave)
```

Formato mínimo do CSV: uma coluna com o texto da avaliação. Nota e data são opcionais (a data libera a aba de tendência).

## Estrutura

```
app.py                       painel Streamlit
reviews/
  schema.py                  formato da resposta do LLM (Pydantic)
  llm.py                     prompts + cliente Gemini
  analysis.py                classificação em lotes e agregações do painel
  evaluation.py              métricas contra gabarito
scripts/
  generate_example.py        gera o CSV fictício com gabarito
  evaluate.py                roda a avaliação
tests/                       testes com LLM falso
```

## Próximos passos

- Rodar com avaliações reais de um restaurante parceiro e medir a qualidade num gabarito revisado à mão.
- Alertas automáticos (ex.: e-mail quando a reclamação de um assunto sobe mais de X% na semana).
- Sugestão de resposta para cada caso urgente.

---

Feito por [Arthur Alen](https://linkedin.com/in/arthuralen) · Licença MIT
