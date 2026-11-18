# CLAUDE.md

## Sobre o projeto

**Financial Statements Extractor**: extração de demonstrações financeiras (Balanço Patrimonial, DRE, DFC) de PDFs usando IA generativa (LLMs com saída estruturada), com **avaliação quantitativa** contra os dados oficiais da CVM.

O diferencial do projeto não é só extrair, é **medir o quanto a extração acerta** e explicar onde e por que ela erra, com um olhar contábil (fechamento de totais, consistência entre demonstrações).

Projeto de portfólio pessoal, público no GitHub, voltado a posicionamento em **AI for Finance & Accounting**.

## Sobre mim (autora)

- Giovanna: analista de BI (Power BI, Power Query, DAX, SQL, Python) no time de Financial Modeling de uma consultoria, e estudante de Contabilidade na FEA-USP.
- Domino a parte contábil e de BI. Estou **aprendendo** IA generativa e engenharia de software em Python, então:
  - Explique decisões técnicas de forma breve quando introduzir algo novo (biblioteca, padrão, conceito).
  - Prefira soluções simples e legíveis a abstrações sofisticadas.
  - Trabalhe em passos pequenos e verificáveis; não gere o projeto inteiro de uma vez.

## Idiomas

- **Converse comigo em português.**
- **Tudo que vai para o repositório fica em inglês:** código, nomes de variáveis, comentários, docstrings, README, mensagens de commit.
- Termos contábeis brasileiros que vêm dos dados (ex.: nomes de contas da CVM) ficam como estão nos dados.

## Regras importantes

1. **Nunca usar dados de clientes reais** do meu trabalho. Só dados públicos (CVM, sites de RI das empresas) ou dados sintéticos.
2. **Nunca commitar segredos.** Chaves de API ficam em `.env` (que está no `.gitignore`); manter um `.env.example` sem valores.
3. **Não fazer commit nem push sem eu pedir.**
4. **Não baixar grandes volumes de dados** sem confirmar comigo antes.
5. PDFs e dados baixados ficam em `data/` e **não vão para o Git** (exceto pequenas amostras em `data/samples/`, se fizer sentido).

## Fontes de dados

- **Gabarito (ground truth):** Portal de Dados Abertos da CVM (dados.cvm.gov.br), conjuntos **DFP** (anual) e **ITR** (trimestral) de companhias abertas. São CSVs separados por demonstração (BPA, BPP, DRE, DFC etc.), consolidado (`con`) e individual (`ind`).
  - Atenção a colunas como `ESCALA_MOEDA` (UNIDADE vs. MIL), `ORDEM_EXERC` (último vs. penúltimo exercício) e `VERSAO` (reapresentações). Confirmar a estrutura real dos arquivos antes de escrever o loader.
- **Entrada da IA:** PDFs das demonstrações financeiras das mesmas empresas e períodos (disponíveis na CVM / sites de RI).
- Começar com **3 a 4 empresas conhecidas** e **um único exercício**.

## Stack

- Python 3.11+
- Gerenciamento de ambiente: `uv` (se não estiver disponível, `venv` + `requirements.txt`)
- Leitura de PDF: PyMuPDF (`pymupdf`); `pdfplumber` como alternativa para tabelas difíceis
- LLM: saída estruturada validada com **Pydantic**
  - O provedor deve ser **plugável** (configurado por variável de ambiente): Anthropic, OpenAI ou modelo local via Ollama. Facilita comparar modelos e controlar custo.
- Dados: pandas e/ou DuckDB
- Demo: Streamlit
- Testes: pytest

## Estrutura

```
.
├── CLAUDE.md
├── README.md
├── LICENSE                   # MIT
├── .env.example
├── pyproject.toml            # grupos: dev (pytest), report (matplotlib), app (streamlit)
├── data/                     # gitignored (raw/, processed/), exceto samples/
│   ├── raw/                  # PDFs <TICKER>_DFP_2025.pdf; hard_cases/; cvm/ (zip DFP)
│   ├── processed/runs/       # cada execução de experimento (JSON)
│   └── samples/              # amostra versionada usada pelo app online
├── src/
│   ├── extraction/           # não importa nada de evaluation/
│   │   ├── pdf_reader.py      # PDF -> texto/páginas; localiza as páginas da demonstração
│   │   ├── schemas.py         # modelos Pydantic (linha, demonstração, escala)
│   │   ├── llm_client.py      # interface única para os provedores de LLM
│   │   └── extractor.py       # versões do prompt; orquestra PDF -> LLM -> dados
│   ├── evaluation/
│   │   ├── cvm_loader.py      # carrega e normaliza os CSVs da CVM (gabarito)
│   │   ├── matching.py        # totais padronizados, rascunho de de-para, pareamento
│   │   ├── metrics.py         # cobertura, valor exato, dentro do arredondamento
│   │   ├── integrity.py       # checagens contábeis (sem gabarito)
│   │   ├── error_injection.py # experimento de erros plantados
│   │   └── evaluate.py        # avaliação via linha de comando
│   └── experiments/          # combina extração + avaliação
│       ├── run_experiment.py  # prompt x modelo x effort x empresa x repetição
│       ├── feedback.py        # retry com as checagens que falharam
│       ├── tables.py          # tabelas de resultados (README e app)
│       ├── report.py          # gera docs/results.md e docs/figures/
│       └── build_samples.py   # gera data/samples/ para o app
├── mappings/                 # de-para PDF -> CVM revisados pela autora
├── results/                  # um CSV por experimento (versionado)
├── docs/                     # results.md e figuras do README
├── app/
│   ├── streamlit_app.py
│   └── requirements.txt       # gerado com `uv export` (Streamlit Community Cloud)
└── tests/
```

Mantenha a separação entre **extração** e **avaliação**: o que precisa das duas fica em `src/experiments/`.

## Roadmap

1. **MVP de extração:** 1 PDF -> texto -> LLM -> tabela (conta, valor, período, empresa) validada por Pydantic.
2. **Avaliação:** carregar os dados da CVM, casar as contas e calcular métricas de acerto.
3. **Checagens contábeis:** Ativo = Passivo + PL; subtotais fecham; consistência de escala (mil vs. unidade); sinais (despesas negativas etc.).
4. **Experimentos:** comparar prompts e modelos (incluindo um local), tratar tabelas quebradas entre páginas, registrar custo e tempo por documento.
5. **Apresentação:** app Streamlit e README com resultados, gráficos e limitações.

Antes de começar cada fase, proponha um plano curto e espere meu ok.

## Convenções de código

- Funções pequenas, com type hints e docstrings curtas.
- Valores monetários: usar `Decimal` ou inteiros (centavos/unidades) na comparação, nunca comparar floats diretamente.
- Normalizar sempre a escala para a mesma unidade antes de comparar valores.
- Logs simples (`logging`) em vez de `print` no código de `src/`.
- Mensagens de commit no padrão Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

## README (quando chegar a hora)

O README é a vitrine do projeto. Deve conter: problema de negócio, abordagem, **resultados da avaliação em destaque** (tabela de métricas por modelo), exemplos de entrada/saída, limitações conhecidas e como rodar.
