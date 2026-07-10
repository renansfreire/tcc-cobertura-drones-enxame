# Otimização de Cobertura de Áreas por Drones: Uma Análise Comparativa de Algoritmos de Enxame

Trabalho de Conclusão de Curso — Engenharia da Computação, Universidade Federal do Maranhão (UFMA), 2026.

**Autor:** Renan Sousa Freire
**Orientador:** Prof. Dr. Pedro Baptista Fernandes

## Sobre o trabalho

Estudo comparativo de cinco algoritmos de inteligência de enxame — PSO, GWO, WOA, SSA e Cuckoo Search (CS) — aplicados ao posicionamento estático de VANTs (drones) para cobertura de área, avaliados por uma função custo multi-objetivo (cobertura, sobreposição e conectividade de rede). Os algoritmos foram testados em 9 cenários (3 escalas de área × 3 densidades de drones), sob 3 perfis de peso operacional, com 20 execuções independentes por condição (3.240 execuções no total), com validação estatística por teste de Wilcoxon com correção de Holm-Bonferroni.

Este repositório contém o código-fonte da simulação e os resultados brutos completos — incluindo os que não entraram no corpo do TCC, disponíveis aqui para quem quiser conferir ou reproduzir o experimento na íntegra.

## Estrutura

```
├── tcc_simulacao_v3.py   → código completo da simulação (algoritmos, função custo, análise estatística)
├── requirements.txt      → dependências Python
├── resultados/           → CSVs com todas as análises geradas (ranking, Wilcoxon, ganho vs. baseline etc.)
└── figuras/               → gráficos de convergência e mapas de cobertura gerados pela simulação
```

## Como rodar

```bash
pip install -r requirements.txt
python tcc_simulacao_v3.py
```

O script permite alternar entre `DEMO_MODE = True` (execução rápida, 1 cenário, poucos agentes/iterações — útil para testar em qualquer máquina) e `DEMO_MODE = False` (experimento oficial completo, 9 cenários × 3 perfis × 20 execuções, com paralelização via `joblib`).

## Contato

contato@engrenan.dev