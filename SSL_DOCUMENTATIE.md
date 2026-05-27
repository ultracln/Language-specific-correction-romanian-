# Documentație SSL – Corectare gramaticală specifică limbii române

**Proiect:** Language-specific correction for Romanian  
**Autor:** Alexandru Duca  
**Notebook principal:** `kaggle_ssl_comparison.ipynb`  
**Dataset:** `nlpssl-project` (Kaggle) → `full_dataset.csv` (400.000 propoziții)

---

## 1. Ce este SSL (Self-Supervised Learning)?

SSL (Învățare Auto-Supervizată) permite antrenarea unui model pe **date fără etichete**. Nu avem nevoie de perechi (greșit, corect) annotate manual — modelul generează singur semnalul de supervizare prin **coruperea și reconstruirea textului**.

### Paradigma Denoising Autoencoder (DAE)

```
Text curat  →  [Corupere artificială]  →  Text corupt
Text corupt →  [Model BERT]            →  Reconstruire text curat
```

Modelul primește textul corupt ca intrare și trebuie să prezică textul original curat token cu token. Astfel, el **învață singur** reprezentări robuste ale limbii române, utile pentru detectarea erorilor reale ulterior.

---

## 2. Arhitectura sistemului

### 2.1 Fișiere principale

| Fișier | Rol |
|---|---|
| `src/ssl_corruption.py` | Generarea corupțiilor artificiale pe text românesc |
| `src/ssl_trainer.py` | Antrenarea DAE (varianta script CLI) |
| `src/data_prep.py` | Preprocesarea dataset-ului CSV → fișiere JSONL pentru detector |
| `src/detector.py` | Detectorul de erori (clasificare token-level) |
| `src/prepare_unlabeled_corpus.py` | Extrage coloana `correct` din CSV → corpus SSL |
| `src/utils.py` | Funcții comune (tokenizare, seed, scriere JSONL) |
| `kaggle_ssl_comparison.ipynb` | Notebook Kaggle: rulează tot fluxul de la A la Z |

### 2.2 Fluxul complet

```
full_dataset.csv
      │
      ├─► [Celula 4] data_prep.py
      │         └─► data/prepared/detector_train.jsonl
      │             data/prepared/detector_val.jsonl
      │             data/prepared/detector_test.jsonl
      │
      ├─► [Celula 5] prepare_unlabeled_corpus.py
      │         └─► data/unlabeled_corpus_full.txt
      │
      ├─► [Celula 6] SSL Pre-training (DAE)
      │         input:  unlabeled_corpus_full.txt
      │         model:  readerbench/RoBERT-large
      │         output: results/ssl_dae/best/  ← model SSL antrenat
      │
      ├─► [Celula 7] Detector BASELINE
      │         input:  data/prepared/ + RoBERT-large (stock)
      │         output: results/detector_baseline/
      │
      └─► [Celula 8] Detector SSL
                input:  data/prepared/ + results/ssl_dae/best/
                output: results/detector_ssl/
```

---

## 3. Modulul de corupere (`ssl_corruption.py`)

### 3.1 Tipuri de corupere implementate

| Tip | Metodă | Exemplu |
|---|---|---|
| Diacritice | `corrupt_diacritics` | `"ână"` → `"ana"` |
| Transpunere caractere | `corrupt_character_swap` | `"carte"` → `"catre"` |
| Substituție tastatură | `corrupt_character_substitute` | `"bun"` → `"vun"` |
| Erori fonetice | `corrupt_phonetic` | `"împreună"` → `"înpreună"` |
| Majuscule greșite | `corrupt_case` | `"România"` → `"românia"` |
| Ștergere token | `corrupt_token_drop` | `"o zi bună"` → `"o bună"` |
| Spații extra | `corrupt_whitespace` | `"test"` → `"te st"` |
| Repetare vocale | `corrupt_letter_repetition` | `"drag"` → `"draaag"` (social media) |
| Abrevieri | `corrupt_abbreviation` | `"pentru"` → `"ptr"` |
| Spațiu lipsă | `corrupt_missing_space` | `"zi bună"` → `"zibună"` |

### 3.2 Tipuri de sesiuni de corupere (curriculum)

```python
'light'  → pool: ['diacritics', 'char_swap']          # 1 corupere/propoziție
'medium' → pool: ['diacritics', 'char_swap', 'char_substitute', 'phonetic']  # 2 coruperi
'heavy'  → pool: ['char_substitute', 'phonetic', 'case', 'token_drop', ...]  # 3 coruperi
'mixed'  → toate metodele disponibile                  # 2 coruperi
```

### 3.3 Curriculum learning

Antrenarea SSL urmează un curriculum în 3 epoci, de la coruperi ușoare la complexe:

| Epocă | Tip | Intensitate |
|---|---|---|
| 1 | `light` | 0.3 |
| 2 | `medium` | 0.55 |
| 3 | `mixed` | 0.8 |

Intensitatea este un **multiplicator de probabilitate** — cu intensitate 0.3, metoda `diacritics` (prob=0.4) devine efectiv 0.4×0.3=0.12.

---

## 4. Antrenarea SSL în Celula 6 (notebook)

### 4.1 Hiperparametri actuali

```python
MODEL_NAME   = 'readerbench/RoBERT-large'  # Model de bază
EPOCHS       = 3
BATCH_SIZE   = 8          # per GPU
MAX_LENGTH   = 96         # tokeni per propoziție
LR           = 2e-5
GRAD_ACCUM   = 8          # batch efectiv = 8×8 = 64
MAX_EXAMPLES = 20000      # propoziții din corpus SSL
```

### 4.2 Arhitectura modelului SSL

```
Encoder: RoBERT-large (BertModel)     ← frozen/fine-tuned
Head:    nn.Linear(hidden_size, vocab_size)  ← capul de reconstruire
Loss:    CrossEntropyLoss (masked — ignoră padding)
```

Modelul prezice fiecare token al textului curat, pornind din reprezentarea textului corupt.

### 4.3 Suport multi-GPU (T4 x2)

Notebook-ul detectează automat numărul de GPU-uri și activează `DataParallel`:

```python
n_gpus = torch.cuda.device_count()
if n_gpus > 1:
    encoder = nn.DataParallel(encoder)
    head    = nn.DataParallel(head)
```

La salvare se folosește `.module` pentru a extrage modelul din wrapper-ul DataParallel:
```python
_enc = encoder.module if n_gpus > 1 else encoder
_enc.save_pretrained(best_dir)
```

### 4.4 Optimizări de memorie

- **Gradient checkpointing**: `encoder.gradient_checkpointing_enable()` — economisește ~30% VRAM prin recalcularea activărilor la backward
- **`PYTORCH_ALLOC_CONF=expandable_segments:True`** — reduce fragmentarea memoriei CUDA
- `gc.collect()` + `torch.cuda.empty_cache()` la final

---

## 5. Ce s-a schimbat față de versiunea anterioară

### 5.1 Dataset

| | Versiune veche | Versiune nouă |
|---|---|---|
| Fișier | `synthetic.csv` | `full_dataset.csv` |
| Dimensiune | ~50.000 rânduri | **400.000 propoziții** |
| Surse | Sintetic | CC-100, Wikipedia-RO, EUR-Lex, RoTexts |
| Coloane noi | — | `difficulty`, `edit_distance`, `curriculum_phase` |

Dataset-ul nou are propoziții din 4 surse diferite, acoperind registre lingvistice diverse (informal, enciclopedic, juridic, variat).

### 5.2 Notebook (`kaggle_ssl_comparison.ipynb`)

| Celulă | Schimbare |
|---|---|
| **Celula 1** | Reinstalare automată `torch==2.3.1` cu CUDA corect (fix pentru eroarea `no kernel image`) |
| **Celula 3** | Caută `full_dataset.csv` în loc de `synthetic.csv`; cale dataset: `nlpssl-project` |
| **Celula 4** | `--csv data/full_dataset.csv` |
| **Celula 5** | `--input data/full_dataset.csv` |
| **Celula 6** | Suport multi-GPU (DataParallel), gradient checkpointing, `BATCH_SIZE=8`, `GRAD_ACCUM=8` |

### 5.3 Dataset Kaggle

- Nume vechi: `NLP_Project`
- Nume nou: `NLP/SSL_Project`
- Slug Kaggle: `nlpssl-project`
- Cale completă: `/kaggle/input/datasets/alexandruduca/nlpssl-project`

---

## 6. Rezultate SSL – Dataset original (synthetic.csv, referință)

| Epocă | Tip corupție | Intensitate | Loss mediu |
|---|---|---|---|
| 1 | light | 0.3 | 4.6422 |
| 2 | medium | 0.55 | 2.9669 |
| 3 | mixed | 0.8 | **2.5551** ← best |

Cu dataset-ul nou (mai mare și mai divers), loss-ul final este așteptat ≤ 2.55.

---

## 7. Cum rulezi notebook-ul (pas cu pas)

### Pregătire Kaggle

1. Mergi la [kaggle.com](https://kaggle.com) → **Your Work → Notebooks**
2. Deschide `kaggle_ssl_comparison.ipynb`
3. Settings (dreapta) → **Accelerator → GPU T4 x1** sau **T4 x2**
4. Settings → **Add Data** → caută `nlpssl-project` → Add

### Rulare celule

```
Celula 1  → Instalare pachete + fix CUDA (apoi Restart Session!)
Celula 2  → Creare directoare + verificare dataset disponibil
Celula 3  → Copiere fișiere src/ și full_dataset.csv în /kaggle/working/
Celula 4  → data_prep.py → generează detector_train/val/test.jsonl
Celula 5  → prepare_unlabeled_corpus.py → generează unlabeled_corpus_full.txt
Celula 6  → SSL Pre-training DAE (~20-25 min cu T4 x1, ~12 min cu T4 x2)
Celula 7  → Antrenare Detector BASELINE (RoBERT-large stock)
Celula 8  → Antrenare Detector SSL (pornind din results/ssl_dae/best/)
Celula 9  → Comparație finală Baseline vs SSL (F1, Accuracy, grafice)
```

> **Atenție:** Dacă laptopul intră în hibernare, sesiunea Kaggle se poate pierde.  
> Folosește **Save & Run All (Commit)** pentru rulare în background fără browser.

---

## 8. Structura fișierelor după rulare

```
/kaggle/working/
├── src/                          ← fișierele Python copiate din dataset
├── data/
│   ├── full_dataset.csv          ← dataset principal
│   ├── unlabeled_corpus_full.txt ← corpus SSL (coloana 'correct')
│   └── prepared/
│       ├── detector_train.jsonl
│       ├── detector_val.jsonl
│       └── detector_test.jsonl
└── results/
    ├── ssl_dae/
    │   ├── best/                 ← model SSL cel mai bun (după loss)
    │   ├── final/                ← model SSL de la ultima epocă
    │   └── training_info.json    ← {'best_loss': ..., 'epochs': 3}
    ├── detector_baseline/
    │   ├── model/                ← detector fără SSL
    │   └── history.json          ← metrici per epocă
    └── detector_ssl/
        ├── model/                ← detector cu SSL
        └── history.json
```

---

## 9. Dependențe

```
torch>=2.3.1
transformers>=4.40
scikit-learn
pandas
tqdm
matplotlib
```

Instalate automat în Celula 1 a notebook-ului.

---

## 10. Note tehnice pentru colegi

- **`data_prep.py`** citește coloanele: `correct`, `incorrect`, `error_type`, `has_error` — compatibil cu `full_dataset.csv`
- **Modelul SSL salvat** în `results/ssl_dae/best/` este un `BertModel` standard (fără head) — poate fi încărcat cu `AutoModel.from_pretrained('results/ssl_dae/best')`
- **`detector.py`** acceptă `--model_name` ca argument — pentru a folosi modelul SSL, pasează calea `results/ssl_dae/best/` în loc de `readerbench/RoBERT-large`
- Coloana `curriculum_phase` din dataset (`easy/medium/hard`) **nu este folosită** de `data_prep.py` curent — poate fi exploatată în viitor pentru curriculum learning și la nivelul detectorului
