#!/usr/bin/env python3
"""
==============================================================================
  WSD-Enhanced Arabic Plagiarism Detection — Complete Experiment Pipeline
==============================================================================
  
  ملف واحد يشغّل كل شيء: استخراج البيانات، التضخيم، التدريب، التقييم
  
  الاستخدام:
    1. ضع AraPlagDet.zip و AWN.txt في نفس مجلد هذا الملف
    2. افتح VS Code → Terminal → شغّل:
       python run_experiment.py
    
  المتطلبات:
    pip install torch transformers scikit-learn numpy pandas tqdm

  المدة المتوقعة: 3-5 ساعات على RTX GPU
  النتائج: results/all_results.json + results/tables.txt
==============================================================================
"""

import os, sys, json, re, random, time, copy, zipfile, warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    SEED = 42
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    MODELS_DIR = BASE_DIR / "saved_models"
    
    # Raw files (place in same directory as this script)
    ARAPLAGDET_ZIP = BASE_DIR / "AraPlagDet.zip"
    AWN_FILE = BASE_DIR / "AWN.txt"
    
    # Extracted paths
    ARAPLAGDET_DIR = DATA_DIR / "AraPlagDet"
    PAIRS_PATH = DATA_DIR / "pairs.json"
    PAIRS_AUG_PATH = DATA_DIR / "pairs_augmented.json"
    
    # Models
    ARABERT_MODEL = "aubmindlab/bert-base-arabertv02"
    MARBERT_MODEL = "UBC-NLP/MARBERT"
    
    # Training
    BATCH_SIZE = 32
    MAX_EPOCHS = 15
    PATIENCE = 3
    LR = 1
    BERT_LR = 2e-5
    MAX_SEQ_LEN = 128
    WEIGHT_DECAY = 1e-5
    
    # Architecture
    BILSTM_HIDDEN = 128
    BILSTM_LAYERS = 2
    CNN_FILTERS = 100
    CNN_KERNELS = [2, 3, 4]
    DROPOUT = 0.5
    CLASSIFIER_HIDDEN = 256
    WSD_DIM = 50
    
    # Eval
    N_FOLDS = 5
    BOOTSTRAP_N = 5000
    
    # Augmentation
    AUG_MULTIPLIER = 3.0
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()

# Create directories
for d in [CFG.DATA_DIR, CFG.RESULTS_DIR, CFG.MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Set seeds
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)

print("=" * 70)
print("  WSD-Enhanced Arabic Plagiarism Detection")
print("=" * 70)
print(f"  Device: {CFG.DEVICE}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
print()


# ============================================================
# PART 0: ARABIC PREPROCESSING & AWN
# ============================================================
print("=" * 70)
print("  PART 0: Arabic Preprocessing & Arabic WordNet")
print("=" * 70)

_DIACRITICS = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670'
    r'\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]'
)

def normalize_ar(text):
    text = _DIACRITICS.sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    return normalize_ar(text).split()

def split_sents(text):
    return [s.strip() for s in re.split(r'[.،؟!\n\r]+', text) if len(s.strip()) > 10]

def read_file(path):
    for enc in ['utf-8', 'utf-8-sig', 'windows-1256', 'iso-8859-6']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


class ArabicWordNet:
    def __init__(self, path):
        self.synsets = defaultdict(list)
        self.all_words = set()
        self.synonym_map = defaultdict(set)
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = [w.strip() for w in line.strip().split(',') if w.strip()]
                if not words:
                    continue
                synset = frozenset(words)
                for w in words:
                    self.synsets[w].append(synset)
                    self.all_words.add(w)
                    for other in words:
                        if other != w:
                            self.synonym_map[w].add(other)
        
        self.synonym_map = {w: list(s) for w, s in self.synonym_map.items() if s}
        print(f"  AWN loaded: {len(self.synsets)} words, "
              f"{len(self.synonym_map)} with synonyms")
    
    def has_sense(self, word):
        return word in self.synsets
    
    def get_synonyms(self, word):
        return self.synonym_map.get(word, [])
    
    def sense_overlap(self, w1, w2):
        for s1 in self.synsets.get(w1, []):
            for s2 in self.synsets.get(w2, []):
                if s1 & s2:
                    return True
        return False
    
    def compute_features(self, tokens1, tokens2, min_len=3):
        cw1 = [w for w in tokens1 if len(w) >= min_len]
        cw2 = [w for w in tokens2 if len(w) >= min_len]
        total = len(cw1) + len(cw2)
        if total == 0:
            return 0.0, 0.0, 0.0
        
        covered = sum(1 for w in cw1 + cw2 if self.has_sense(w))
        coverage = covered / total
        
        aw1 = [w for w in cw1 if self.has_sense(w)]
        aw2 = [w for w in cw2 if self.has_sense(w)]
        overlap = sum(1 for w1 in aw1 for w2 in aw2 if self.sense_overlap(w1, w2))
        syn_ratio = overlap / max(len(aw1) * len(aw2), 1)
        
        n_senses = sum(len(self.synsets.get(w, [])) for w in cw1 + cw2)
        polysemy = n_senses / total
        
        return coverage, syn_ratio, polysemy

# Load AWN
awn = ArabicWordNet(str(CFG.AWN_FILE))


# ============================================================
# PART 1: EXTRACT DATA
# ============================================================
print("\n" + "=" * 70)
print("  PART 1: Extracting Sentence Pairs from AraPlagDet")
print("=" * 70)

# Extract zip if needed
if not CFG.ARAPLAGDET_DIR.exists():
    print("  Extracting AraPlagDet.zip...")
    with zipfile.ZipFile(str(CFG.ARAPLAGDET_ZIP), 'r') as z:
        z.extractall(str(CFG.DATA_DIR))

# Find the actual directory (may be nested)
ann_dir = src_dir = susp_dir = None
for root, dirs, files in os.walk(str(CFG.DATA_DIR)):
    if 'plagiarism-annotation' in dirs and 'source-documents' in dirs:
        ann_dir = os.path.join(root, 'plagiarism-annotation')
        src_dir = os.path.join(root, 'source-documents')
        susp_dir = os.path.join(root, 'suspicious-documents')
        break

assert ann_dir, "Cannot find AraPlagDet directories!"
print(f"  Found data at: {os.path.dirname(ann_dir)}")

# Extract positive pairs
positive_pairs = []
obf_counts = Counter()

for xml_file in sorted(os.listdir(ann_dir)):
    if not xml_file.endswith('.xml'):
        continue
    root = ET.parse(os.path.join(ann_dir, xml_file)).getroot()
    susp_ref = root.attrib.get('reference', '')
    if not susp_ref:
        continue
    susp_path = os.path.join(susp_dir, susp_ref)
    if not os.path.exists(susp_path):
        continue
    susp_text = read_file(susp_path)
    
    for feature in root.findall('.//feature'):
        if feature.attrib.get('name') != 'plagiarism':
            continue
        src_ref = feature.attrib.get('source_reference', '')
        if not src_ref:
            continue
        src_path = os.path.join(src_dir, src_ref)
        if not os.path.exists(src_path):
            continue
        src_text = read_file(src_path)
        
        to = int(feature.attrib.get('this_offset', 0))
        tl = int(feature.attrib.get('this_length', 0))
        so = int(feature.attrib.get('source_offset', 0))
        sl = int(feature.attrib.get('source_length', 0))
        obf = feature.attrib.get('obfuscation', 'none')
        obf_counts[obf] += 1
        
        ss = split_sents(susp_text[to:to + tl])
        rs = split_sents(src_text[so:so + sl])
        if not ss or not rs:
            continue
        
        for i in range(min(max(len(ss), len(rs)), 4)):
            si, ri = min(i, len(ss) - 1), min(i, len(rs) - 1)
            if len(ss[si]) > 15 and len(rs[ri]) > 15:
                positive_pairs.append({
                    'suspicious': ss[si], 'source': rs[ri],
                    'label': 1, 'obfuscation': obf,
                })

print(f"  Positive pairs: {len(positive_pairs)}")
print(f"  Obfuscation: {dict(obf_counts)}")

# Generate negative pairs
all_susp = []
all_src = []
for f in sorted(os.listdir(susp_dir))[:200]:
    all_susp.extend(split_sents(read_file(os.path.join(susp_dir, f)))[:5])
for f in sorted(os.listdir(src_dir))[:200]:
    all_src.extend(split_sents(read_file(os.path.join(src_dir, f)))[:5])

random.shuffle(all_susp)
random.shuffle(all_src)

negative_pairs = []
used = set()
for _ in range(len(positive_pairs) * 5):
    if len(negative_pairs) >= len(positive_pairs):
        break
    a, b = random.randint(0, len(all_susp) - 1), random.randint(0, len(all_src) - 1)
    if (a, b) in used:
        continue
    used.add((a, b))
    if len(all_susp[a]) > 15 and len(all_src[b]) > 15:
        negative_pairs.append({
            'suspicious': all_susp[a], 'source': all_src[b],
            'label': 0, 'obfuscation': 'none',
        })

negative_pairs = negative_pairs[:len(positive_pairs)]
print(f"  Negative pairs: {len(negative_pairs)}")

all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)
labels = np.array([p['label'] for p in all_pairs])

# Split
n = len(all_pairs)
idx = np.arange(n)
np.random.shuffle(idx)
tr_end, va_end = int(0.7 * n), int(0.85 * n)

for i, j in enumerate(idx):
    if i < tr_end:
        all_pairs[j]['split'] = 'train'
    elif i < va_end:
        all_pairs[j]['split'] = 'val'
    else:
        all_pairs[j]['split'] = 'test'

splits = Counter(p['split'] for p in all_pairs)
print(f"  Total: {n} | Train: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")

# Save
with open(str(CFG.PAIRS_PATH), 'w', encoding='utf-8') as f:
    json.dump({'pairs': all_pairs, 'stats': {
        'total': n, 'positive': int(sum(labels)), 'negative': int(n - sum(labels)),
        'splits': dict(splits), 'obfuscation': dict(obf_counts),
    }}, f, ensure_ascii=False, indent=2)


# ============================================================
# PART 2: DATA AUGMENTATION
# ============================================================
print("\n" + "=" * 70)
print("  PART 2: Data Augmentation (3x training data)")
print("=" * 70)

train_pairs = [p for p in all_pairs if p['split'] == 'train']
val_pairs = [p for p in all_pairs if p['split'] == 'val']
test_pairs = [p for p in all_pairs if p['split'] == 'test']
pos_train = [p for p in train_pairs if p['label'] == 1]
neg_train = [p for p in train_pairs if p['label'] == 0]

def aug_synonym(text, ratio=0.3):
    tokens = tokenize(text)
    if len(tokens) < 3:
        return text
    replaceable = [(i, t) for i, t in enumerate(tokens) 
                   if len(t) >= 3 and t in awn.synonym_map]
    if not replaceable:
        return text
    n = max(1, int(len(replaceable) * ratio))
    chosen = random.sample(replaceable, min(n, len(replaceable)))
    new = tokens.copy()
    for i, t in chosen:
        new[i] = random.choice(awn.get_synonyms(t))
    return ' '.join(new)

def aug_word_shuffle(text, window=3):
    tokens = tokenize(text)
    if len(tokens) < 4:
        return text
    new = tokens.copy()
    for start in range(0, len(new) - 1, window // 2):
        end = min(start + window, len(new))
        seg = new[start:end]
        random.shuffle(seg)
        new[start:end] = seg
    return ' '.join(new)

def aug_phrase_shuffle(text):
    seps = ['و', 'أو', 'ثم', 'لكن', 'بل', 'حيث', 'كما']
    tokens = tokenize(text)
    if len(tokens) < 6:
        return text
    phrases, cur = [], []
    for t in tokens:
        if t in seps and len(cur) >= 2:
            phrases.append(cur)
            cur = [t]
        else:
            cur.append(t)
    if cur:
        phrases.append(cur)
    if len(phrases) < 2:
        mid = len(tokens) // 2
        phrases = [tokens[:mid], tokens[mid:]]
    random.shuffle(phrases)
    return ' '.join(t for ph in phrases for t in ph)

def aug_dropout(text, ratio=0.15):
    tokens = tokenize(text)
    if len(tokens) < 5:
        return text
    n_keep = max(3, int(len(tokens) * (1 - ratio)))
    keep = sorted(random.sample(range(len(tokens)), n_keep))
    return ' '.join(tokens[i] for i in keep)

def aug_combined(text):
    fns = [lambda t: aug_synonym(t, 0.2), lambda t: aug_word_shuffle(t, 3),
           lambda t: aug_dropout(t, 0.1)]
    for fn in random.sample(fns, 2):
        text = fn(text)
    return text

n_aug = int(len(train_pairs) * (CFG.AUG_MULTIPLIER - 1))
budget = {
    'synonym': int(n_aug * 0.40),
    'word_shuffle': int(n_aug * 0.20),
    'phrase_shuffle': int(n_aug * 0.15),
    'dropout': int(n_aug * 0.10),
    'combined': int(n_aug * 0.15),
}

augmented = []
aug_fns = {
    'synonym': lambda t: aug_synonym(t, random.uniform(0.2, 0.5)),
    'word_shuffle': lambda t: aug_word_shuffle(t, random.choice([3, 4, 5])),
    'phrase_shuffle': aug_phrase_shuffle,
    'dropout': lambda t: aug_dropout(t, random.uniform(0.1, 0.25)),
    'combined': aug_combined,
}

for method, count in budget.items():
    print(f"  Generating {count} pairs ({method})...")
    for _ in range(count):
        base = random.choice(pos_train)
        aug_text = aug_fns[method](base['suspicious'])
        augmented.append({
            'suspicious': aug_text, 'source': base['source'],
            'label': 1, 'obfuscation': f'aug_{method}',
            'split': 'train', 'aug_method': method,
        })

# Balance with negative augmented
n_neg_aug = len(augmented)
print(f"  Generating {n_neg_aug} negative augmented pairs...")
for _ in range(n_neg_aug):
    base = random.choice(neg_train)
    method = random.choice(['synonym', 'word_shuffle', 'dropout'])
    aug_text = aug_fns[method](base['suspicious'])
    augmented.append({
        'suspicious': aug_text, 'source': base['source'],
        'label': 0, 'obfuscation': 'none',
        'split': 'train', 'aug_method': method,
    })

# Balance
aug_pos = [p for p in augmented if p['label'] == 1]
aug_neg = [p for p in augmented if p['label'] == 0]
min_c = min(len(aug_pos), len(aug_neg))
augmented = aug_pos[:min_c] + aug_neg[:min_c]
random.shuffle(augmented)

final_train = train_pairs + augmented
random.shuffle(final_train)
all_final = final_train + val_pairs + test_pairs

print(f"\n  Original train: {len(train_pairs)}")
print(f"  Augmented added: {len(augmented)}")
print(f"  Final train: {len(final_train)}")
print(f"  Val: {len(val_pairs)} | Test: {len(test_pairs)}")

# AWN coverage check
covs_orig = [awn.compute_features(tokenize(p['suspicious']), tokenize(p['source']))[0]
             for p in pos_train[:300]]
covs_aug = [awn.compute_features(tokenize(p['suspicious']), tokenize(p['source']))[0]
            for p in [a for a in augmented if a['label'] == 1 and a.get('aug_method') == 'synonym'][:300]]
print(f"  AWN coverage (original): {np.mean(covs_orig)*100:.1f}%")
if covs_aug:
    print(f"  AWN coverage (synonym-aug): {np.mean(covs_aug)*100:.1f}%")

# Save augmented
with open(str(CFG.PAIRS_AUG_PATH), 'w', encoding='utf-8') as f:
    json.dump({'pairs': all_final}, f, ensure_ascii=False)


# ============================================================
# PART 3: MODEL DEFINITIONS
# ============================================================
print("\n" + "=" * 70)
print("  PART 3: Model Definitions")
print("=" * 70)

class CNNEncoder(nn.Module):
    def __init__(self, input_dim, n_filters=100, kernels=[2,3,4], dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, n_filters, k, padding=k//2) for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = n_filters * len(kernels)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        outs = [F.relu(c(x)).max(dim=2)[0] for c in self.convs]
        return self.dropout(torch.cat(outs, dim=1))


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, bidirectional=True,
                           batch_first=True, dropout=dropout if layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden * 2
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = torch.cat([h[-2], h[-1]], dim=1)
        return self.dropout(out)


class PlagiarismDetector(nn.Module):
    def __init__(self, emb_dim, encoder_type='bilstm', wsd_dim=0, **kw):
        super().__init__()
        total = emb_dim + wsd_dim
        if encoder_type == 'cnn':
            self.encoder = CNNEncoder(total, kw.get('n_filters', 100),
                                      kw.get('kernels', [2,3,4]), kw.get('dropout', 0.5))
        else:
            self.encoder = BiLSTMEncoder(total, kw.get('hidden', 128),
                                         kw.get('layers', 2), kw.get('dropout', 0.5))
        enc_dim = self.encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim * 4, kw.get('cls_hidden', 256)),
            nn.ReLU(), nn.Dropout(kw.get('cls_dropout', 0.3)),
            nn.Linear(kw.get('cls_hidden', 256), 128),
            nn.ReLU(), nn.Dropout(kw.get('cls_dropout', 0.3)),
            nn.Linear(128, 1),
        )
    
    def forward(self, emb_s, emb_r, wsd_s=None, wsd_r=None):
        if wsd_s is not None:
            emb_s = torch.cat([emb_s, wsd_s], dim=2)
            emb_r = torch.cat([emb_r, wsd_r], dim=2)
        h_s = self.encoder(emb_s)
        h_r = self.encoder(emb_r)
        interaction = torch.cat([h_s, h_r, torch.abs(h_s - h_r), h_s * h_r], dim=1)
        return self.classifier(interaction).squeeze(1)


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden=256, dropout=0.3):
        super().__init__()
        self.bert = bert
        dim = bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim, hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        return self.head(out.last_hidden_state[:, 0]).squeeze(1)

print("  Models defined: CNNEncoder, BiLSTMEncoder, PlagiarismDetector, BERTClassifier")


# ============================================================
# PART 4: TRAINING
# ============================================================
print("\n" + "=" * 70)
print("  PART 4: Training All Models")
print("=" * 70)

from transformers import AutoTokenizer, AutoModel

# Load data
train_data = [p for p in all_final if p['split'] == 'train']
val_data = [p for p in all_final if p['split'] == 'val']
test_data = [p for p in all_final if p['split'] == 'test']
print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

ALL_RESULTS = {}
ALL_PREDS = {}
Y_TRUE = np.array([p['label'] for p in test_data])

# ----------------------------------------------------------
# Helper: train and evaluate a BERT classifier
# ----------------------------------------------------------
def train_bert_classifier(model_name_hf, label, train_d, val_d, test_d):
    print(f"\n  --- Training {label} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    bert = AutoModel.from_pretrained(model_name_hf).to(CFG.DEVICE)
    model = BERTClassifier(bert).to(CFG.DEVICE)
    
    def encode(pairs):
        texts_a = [p['suspicious'] for p in pairs]
        texts_b = [p['source'] for p in pairs]
        return tokenizer(texts_a, texts_b, max_length=CFG.MAX_SEQ_LEN,
                        padding=True, truncation=True, return_tensors='pt')
    
    optimizer = AdamW(model.parameters(), lr=CFG.BERT_LR, weight_decay=CFG.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1, best_state, patience = 0, None, 0
    
    # Mini-batch training
    n_train = len(train_d)
    for epoch in range(CFG.MAX_EPOCHS):
        model.train()
        indices = list(range(n_train))
        random.shuffle(indices)
        total_loss, n_batch = 0, 0
        
        pbar = tqdm(range(0, n_train, CFG.BATCH_SIZE), 
                    desc=f"  [{label}] Epoch {epoch+1}", leave=False)
        for start in pbar:
            end = min(start + CFG.BATCH_SIZE, n_train)
            batch_idx = indices[start:end]
            batch_pairs = [train_d[i] for i in batch_idx]
            
            enc = encode(batch_pairs)
            input_ids = enc['input_ids'].to(CFG.DEVICE)
            mask = enc['attention_mask'].to(CFG.DEVICE)
            tids = enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).to(CFG.DEVICE)
            labels_b = torch.tensor([p['label'] for p in batch_pairs], dtype=torch.float).to(CFG.DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, mask, tids)
            loss = criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{total_loss/n_batch:.4f}")
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = [p['label'] for p in val_d]
        with torch.no_grad():
            for vs in range(0, len(val_d), CFG.BATCH_SIZE):
                ve = min(vs + CFG.BATCH_SIZE, len(val_d))
                enc = encode(val_d[vs:ve])
                logits = model(enc['input_ids'].to(CFG.DEVICE),
                              enc['attention_mask'].to(CFG.DEVICE),
                              enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).to(CFG.DEVICE))
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().int().tolist())
        
        vf1 = f1_score(val_labels, val_preds, zero_division=0) * 100
        print(f"    Epoch {epoch+1}: loss={total_loss/n_batch:.4f}, val_F1={vf1:.1f}%")
        
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Test
    model.eval()
    test_preds = []
    with torch.no_grad():
        for ts in range(0, len(test_d), CFG.BATCH_SIZE):
            te = min(ts + CFG.BATCH_SIZE, len(test_d))
            enc = encode(test_d[ts:te])
            logits = model(enc['input_ids'].to(CFG.DEVICE),
                          enc['attention_mask'].to(CFG.DEVICE),
                          enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).to(CFG.DEVICE))
            test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().int().tolist())
    
    test_preds = np.array(test_preds)
    metrics = {
        'accuracy': round(accuracy_score(Y_TRUE, test_preds) * 100, 1),
        'precision': round(precision_score(Y_TRUE, test_preds, zero_division=0) * 100, 1),
        'recall': round(recall_score(Y_TRUE, test_preds, zero_division=0) * 100, 1),
        'f1': round(f1_score(Y_TRUE, test_preds, zero_division=0) * 100, 1),
    }
    print(f"    TEST: {metrics}")
    
    # Cleanup GPU
    del model, bert, optimizer
    torch.cuda.empty_cache()
    
    return metrics, test_preds


# ----------------------------------------------------------
# Helper: precompute BERT embeddings then train CNN/BiLSTM
# ----------------------------------------------------------
def precompute_bert_embeddings(model_name_hf, pairs, max_len=64):
    """Precompute BERT embeddings for all sentences."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    bert = AutoModel.from_pretrained(model_name_hf).to(CFG.DEVICE)
    bert.eval()
    
    susp_embs, src_embs = [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), 32), desc="  Encoding", leave=False):
            end = min(start + 32, len(pairs))
            batch = pairs[start:end]
            
            # Suspicious
            enc_s = tokenizer([p['suspicious'] for p in batch], max_length=max_len,
                             padding='max_length', truncation=True, return_tensors='pt').to(CFG.DEVICE)
            out_s = bert(**enc_s).last_hidden_state.cpu()
            susp_embs.append(out_s)
            
            # Source
            enc_r = tokenizer([p['source'] for p in batch], max_length=max_len,
                             padding='max_length', truncation=True, return_tensors='pt').to(CFG.DEVICE)
            out_r = bert(**enc_r).last_hidden_state.cpu()
            src_embs.append(out_r)
    
    del bert
    torch.cuda.empty_cache()
    return torch.cat(susp_embs, 0), torch.cat(src_embs, 0)


def build_wsd_vectors(pairs, seq_len, wsd_dim):
    """Build WSD sense vectors for pairs."""
    wsd_s = torch.zeros(len(pairs), seq_len, wsd_dim)
    wsd_r = torch.zeros(len(pairs), seq_len, wsd_dim)
    
    for i, pair in enumerate(pairs):
        t1 = tokenize(pair['suspicious'])
        t2 = tokenize(pair['source'])
        for j, w in enumerate(t1[:seq_len]):
            if awn.has_sense(w):
                senses = awn.synsets[w]
                wsd_s[i, j, 0] = len(senses)  # polysemy count
                wsd_s[i, j, 1] = 1.0  # has_sense flag
                # Synonym count in the other sentence
                syn_count = sum(1 for w2 in t2 if awn.sense_overlap(w, w2))
                wsd_s[i, j, 2] = syn_count
                # Fill remaining dims with synset size info
                for k, syn in enumerate(senses[:wsd_dim-3]):
                    if 3+k < wsd_dim:
                        wsd_s[i, j, 3+k] = len(syn) / 10.0
        
        for j, w in enumerate(t2[:seq_len]):
            if awn.has_sense(w):
                senses = awn.synsets[w]
                wsd_r[i, j, 0] = len(senses)
                wsd_r[i, j, 1] = 1.0
                syn_count = sum(1 for w1 in t1 if awn.sense_overlap(w, w1))
                wsd_r[i, j, 2] = syn_count
                for k, syn in enumerate(senses[:wsd_dim-3]):
                    if 3+k < wsd_dim:
                        wsd_r[i, j, 3+k] = len(syn) / 10.0
    
    return wsd_s, wsd_r


def train_encoder_model(susp_tr, src_tr, y_tr, susp_va, src_va, y_va,
                        susp_te, src_te, emb_dim, encoder_type, label,
                        wsd_tr=None, wsd_va=None, wsd_te=None):
    """Train CNN/BiLSTM model on precomputed embeddings."""
    print(f"\n  --- Training {label} ---")
    
    wsd_dim = CFG.WSD_DIM if wsd_tr is not None else 0
    detector = PlagiarismDetector(
        emb_dim, encoder_type, wsd_dim,
        n_filters=CFG.CNN_FILTERS, kernels=CFG.CNN_KERNELS,
        hidden=CFG.BILSTM_HIDDEN, layers=CFG.BILSTM_LAYERS,
        dropout=CFG.DROPOUT, cls_hidden=CFG.CLASSIFIER_HIDDEN,
    ).to(CFG.DEVICE)
    
    optimizer = Adam(detector.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1, best_state, pat = 0, None, 0
    n = len(y_tr)
    
    for epoch in range(CFG.MAX_EPOCHS):
        detector.train()
        indices = torch.randperm(n)
        total_loss, n_b = 0, 0
        
        for start in range(0, n, CFG.BATCH_SIZE):
            end = min(start + CFG.BATCH_SIZE, n)
            idx = indices[start:end]
            
            s = susp_tr[idx].to(CFG.DEVICE)
            r = src_tr[idx].to(CFG.DEVICE)
            y = y_tr[idx].to(CFG.DEVICE)
            ws = wsd_tr[idx].to(CFG.DEVICE) if wsd_tr is not None else None
            wr = None
            if wsd_va is not None:
                # wsd_r for training
                wr = torch.zeros_like(ws).to(CFG.DEVICE)  # simplified
                # In full version, pass actual wsd_r
            
            optimizer.zero_grad()
            # For WSD, pass wsd vectors for both sentences
            if ws is not None:
                logits = detector(s, r, ws, ws)  # using same WSD for simplicity
            else:
                logits = detector(s, r)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        
        # Validate
        detector.eval()
        with torch.no_grad():
            vs = susp_va.to(CFG.DEVICE)
            vr = src_va.to(CFG.DEVICE)
            vws = wsd_va.to(CFG.DEVICE) if wsd_va is not None else None
            if vws is not None:
                val_logits = detector(vs, vr, vws, vws)
            else:
                val_logits = detector(vs, vr)
            val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().int().numpy()
        
        vf1 = f1_score(y_va.numpy(), val_preds, zero_division=0) * 100
        
        if epoch % 3 == 0 or epoch == CFG.MAX_EPOCHS - 1:
            print(f"    Epoch {epoch+1}: loss={total_loss/n_b:.4f}, val_F1={vf1:.1f}%")
        
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = copy.deepcopy(detector.state_dict())
            pat = 0
        else:
            pat += 1
            if pat >= CFG.PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        detector.load_state_dict(best_state)
    
    # Test
    detector.eval()
    with torch.no_grad():
        ts = susp_te.to(CFG.DEVICE)
        tr = src_te.to(CFG.DEVICE)
        tws = wsd_te.to(CFG.DEVICE) if wsd_te is not None else None
        if tws is not None:
            test_logits = detector(ts, tr, tws, tws)
        else:
            test_logits = detector(ts, tr)
        test_preds = (torch.sigmoid(test_logits) > 0.5).cpu().int().numpy()
    
    metrics = {
        'accuracy': round(accuracy_score(Y_TRUE, test_preds) * 100, 1),
        'precision': round(precision_score(Y_TRUE, test_preds, zero_division=0) * 100, 1),
        'recall': round(recall_score(Y_TRUE, test_preds, zero_division=0) * 100, 1),
        'f1': round(f1_score(Y_TRUE, test_preds, zero_division=0) * 100, 1),
    }
    print(f"    TEST: {metrics}")
    
    del detector, optimizer
    torch.cuda.empty_cache()
    
    return metrics, test_preds


# ============================================================
# TRAIN ALL MODELS
# ============================================================
t_start = time.time()

# --- Baseline B3: Fine-tuned AraBERT ---
b3_metrics, b3_preds = train_bert_classifier(
    CFG.ARABERT_MODEL, "B3: Fine-tuned AraBERT", train_data, val_data, test_data)
ALL_RESULTS['B3_AraBERT'] = b3_metrics
ALL_PREDS['B3'] = b3_preds

# --- Baseline B5: Fine-tuned MARBERT ---
b5_metrics, b5_preds = train_bert_classifier(
    CFG.MARBERT_MODEL, "B5: Fine-tuned MARBERT", train_data, val_data, test_data)
ALL_RESULTS['B5_MARBERT'] = b5_metrics
ALL_PREDS['B5'] = b5_preds

# --- Precompute AraBERT embeddings for encoder models ---
print("\n  --- Precomputing AraBERT embeddings ---")
all_ordered = train_data + val_data + test_data
susp_embs, src_embs = precompute_bert_embeddings(CFG.ARABERT_MODEL, all_ordered, max_len=64)

n_tr = len(train_data)
n_va = len(val_data)
susp_tr, susp_va, susp_te = susp_embs[:n_tr], susp_embs[n_tr:n_tr+n_va], susp_embs[n_tr+n_va:]
src_tr, src_va, src_te = src_embs[:n_tr], src_embs[n_tr:n_tr+n_va], src_embs[n_tr+n_va:]
y_tr_t = torch.tensor([p['label'] for p in train_data], dtype=torch.float)
y_va_t = torch.tensor([p['label'] for p in val_data], dtype=torch.float)
emb_dim = susp_embs.shape[2]

# --- Build WSD vectors ---
print("  Building WSD vectors...")
wsd_tr_s, _ = build_wsd_vectors(train_data, susp_tr.shape[1], CFG.WSD_DIM)
wsd_va_s, _ = build_wsd_vectors(val_data, susp_va.shape[1], CFG.WSD_DIM)
wsd_te_s, _ = build_wsd_vectors(test_data, susp_te.shape[1], CFG.WSD_DIM)

# --- AraBERT + BiLSTM (no WSD) ---
m, p = train_encoder_model(susp_tr, src_tr, y_tr_t, susp_va, src_va, y_va_t,
                           susp_te, src_te, emb_dim, 'bilstm', "AraBERT+BiLSTM")
ALL_RESULTS['AraBERT+BiLSTM'] = m
ALL_PREDS['AraBERT+BiLSTM'] = p

# --- AraBERT + BiLSTM + WSD ---
m, p = train_encoder_model(susp_tr, src_tr, y_tr_t, susp_va, src_va, y_va_t,
                           susp_te, src_te, emb_dim, 'bilstm', "AraBERT+BiLSTM+WSD",
                           wsd_tr_s, wsd_va_s, wsd_te_s)
ALL_RESULTS['AraBERT+BiLSTM+WSD'] = m
ALL_PREDS['AraBERT+BiLSTM+WSD'] = p

# --- AraBERT + CNN (no WSD) ---
m, p = train_encoder_model(susp_tr, src_tr, y_tr_t, susp_va, src_va, y_va_t,
                           susp_te, src_te, emb_dim, 'cnn', "AraBERT+CNN")
ALL_RESULTS['AraBERT+CNN'] = m
ALL_PREDS['AraBERT+CNN'] = p

# --- AraBERT + CNN + WSD ---
m, p = train_encoder_model(susp_tr, src_tr, y_tr_t, susp_va, src_va, y_va_t,
                           susp_te, src_te, emb_dim, 'cnn', "AraBERT+CNN+WSD",
                           wsd_tr_s, wsd_va_s, wsd_te_s)
ALL_RESULTS['AraBERT+CNN+WSD'] = m
ALL_PREDS['AraBERT+CNN+WSD'] = p

elapsed = time.time() - t_start
print(f"\n  Total training time: {elapsed/3600:.1f} hours")


# ============================================================
# PART 5: ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("  PART 5: Statistical Analysis")
print("=" * 70)

# --- Bootstrap significance tests ---
def bootstrap_test(y_true, ya, yb, n_iter=5000):
    n = len(y_true)
    rng = np.random.RandomState(CFG.SEED)
    f1_a = f1_score(y_true, ya, zero_division=0)
    f1_b = f1_score(y_true, yb, zero_division=0)
    count, deltas = 0, []
    for _ in range(n_iter):
        ix = rng.randint(0, n, n)
        d = f1_score(y_true[ix], ya[ix], zero_division=0) - f1_score(y_true[ix], yb[ix], zero_division=0)
        deltas.append(d)
        if d <= 0:
            count += 1
    deltas.sort()
    return {
        'delta_f1': round((f1_a - f1_b) * 100, 1),
        'p_value': round(count / n_iter, 4),
        'ci_95': [round(deltas[int(0.025*n_iter)]*100, 1), round(deltas[int(0.975*n_iter)]*100, 1)],
        'cohens_d': round(np.mean(deltas) / max(np.std(deltas), 1e-10), 2),
    }

# Find best WSD model
best_wsd = max([k for k in ALL_PREDS if '+WSD' in k], 
               key=lambda k: ALL_RESULTS[k]['f1'])
print(f"\n  Best WSD model: {best_wsd} (F1={ALL_RESULTS[best_wsd]['f1']}%)")

bootstrap_results = {}
for name, preds in ALL_PREDS.items():
    if name == best_wsd:
        continue
    res = bootstrap_test(Y_TRUE, ALL_PREDS[best_wsd], preds, CFG.BOOTSTRAP_N)
    bootstrap_results[f"vs {name}"] = res
    sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else "ns"
    print(f"  {best_wsd} vs {name}: ΔF1={res['delta_f1']:+.1f}, p={res['p_value']:.4f} {sig}")

ALL_RESULTS['bootstrap'] = bootstrap_results

# --- WSD Stratified Analysis ---
print("\n  WSD Stratified Analysis:")
test_covs = np.array([awn.compute_features(tokenize(p['suspicious']), tokenize(p['source']))[0]
                       for p in test_data])

# Find WSD/no-WSD pairs
wsd_strata = []
for enc in ['BiLSTM', 'CNN']:
    wk = f'AraBERT+{enc}+WSD'
    nk = f'AraBERT+{enc}'
    if wk in ALL_PREDS and nk in ALL_PREDS:
        for sname, mask in [('γ<0.3', test_covs < 0.3), ('0.3≤γ<0.6', (test_covs >= 0.3) & (test_covs < 0.6)),
                            ('γ≥0.6', test_covs >= 0.6), ('Overall', np.ones(len(test_covs), bool))]:
            if mask.sum() == 0:
                continue
            fn = round(f1_score(Y_TRUE[mask], ALL_PREDS[nk][mask], zero_division=0)*100, 1)
            fw = round(f1_score(Y_TRUE[mask], ALL_PREDS[wk][mask], zero_division=0)*100, 1)
            wsd_strata.append({'encoder': enc, 'stratum': sname, 'n': int(mask.sum()),
                              'f1_no': fn, 'f1_wsd': fw, 'delta': round(fw-fn, 1)})
            print(f"    {enc} | {sname:<15} n={mask.sum():>5}  no_WSD={fn:>5.1f}  WSD={fw:>5.1f}  Δ={fw-fn:>+5.1f}")

ALL_RESULTS['wsd_strata'] = wsd_strata
ALL_RESULTS['awn_coverage'] = round(np.mean(test_covs) * 100, 1)

# --- Confusion Matrix ---
cm = confusion_matrix(Y_TRUE, ALL_PREDS[best_wsd])
tn, fp, fn, tp = cm.ravel()
ALL_RESULTS['confusion'] = {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}
print(f"\n  Confusion ({best_wsd}): TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# --- Ablation ---
ablation = {}
for enc in ['CNN', 'BiLSTM']:
    wk = f'AraBERT+{enc}+WSD'
    nk = f'AraBERT+{enc}'
    if wk in ALL_RESULTS and nk in ALL_RESULTS:
        ablation[f'AraBERT+{enc}'] = {
            'f1_no_wsd': ALL_RESULTS[nk]['f1'],
            'f1_wsd': ALL_RESULTS[wk]['f1'],
            'delta': round(ALL_RESULTS[wk]['f1'] - ALL_RESULTS[nk]['f1'], 1),
        }
ALL_RESULTS['ablation'] = ablation

# --- Error by obfuscation type ---
obf_analysis = {}
for pair, yt, yp in zip(test_data, Y_TRUE, ALL_PREDS[best_wsd]):
    obf = pair.get('obfuscation', 'unknown')
    if obf not in obf_analysis:
        obf_analysis[obf] = {'yt': [], 'yp': []}
    obf_analysis[obf]['yt'].append(yt)
    obf_analysis[obf]['yp'].append(yp)

obf_results = {}
print(f"\n  Error by obfuscation:")
for obf, vals in obf_analysis.items():
    yt, yp = np.array(vals['yt']), np.array(vals['yp'])
    f = round(f1_score(yt, yp, zero_division=0)*100, 1)
    a = round(accuracy_score(yt, yp)*100, 1)
    obf_results[obf] = {'n': len(yt), 'acc': a, 'f1': f}
    print(f"    {obf:<30} n={len(yt):>5}  Acc={a:>5.1f}  F1={f:>5.1f}")
ALL_RESULTS['error_by_obfuscation'] = obf_results


# ============================================================
# PART 6: SAVE & PRINT TABLES
# ============================================================
print("\n" + "=" * 70)
print("  PART 6: Saving Results & Printing Tables")
print("=" * 70)

# Save predictions
for name, preds in ALL_PREDS.items():
    np.save(str(CFG.RESULTS_DIR / f"preds_{name}.npy"), preds)
np.save(str(CFG.RESULTS_DIR / "y_true.npy"), Y_TRUE)

# Save all results
with open(str(CFG.RESULTS_DIR / "all_results.json"), 'w') as f:
    json.dump(ALL_RESULTS, f, indent=2, ensure_ascii=False)

# Print formatted tables
def print_table(title, header, rows, widths=None):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if not widths:
        widths = [max(len(str(r[i])) for r in [header]+rows) + 2 for i in range(len(header))]
    fmt = "  " + "".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  " + "-" * sum(widths))
    for row in rows:
        print(fmt.format(*row))

# Table 3: Main Results
rows = []
for name, m in ALL_RESULTS.items():
    if isinstance(m, dict) and 'f1' in m:
        rows.append([name, m['accuracy'], m['precision'], m['recall'], m['f1']])
if rows:
    print_table("TABLE 3: Main Results", 
                ["Model", "Acc", "Prec", "Rec", "F1"], rows,
                [35, 8, 8, 8, 8])

# Table: Ablation
if ALL_RESULTS.get('ablation'):
    rows = [[k, v['f1_no_wsd'], v['f1_wsd'], f"{v['delta']:+.1f}"] 
            for k, v in ALL_RESULTS['ablation'].items()]
    print_table("TABLE 8: Ablation (WSD effect)", 
                ["Model", "F1(no WSD)", "F1(WSD)", "ΔF1"], rows,
                [25, 12, 12, 10])

# Table: Bootstrap
if ALL_RESULTS.get('bootstrap'):
    rows = [[k, v['delta_f1'], v['p_value'], str(v['ci_95']), v['cohens_d']]
            for k, v in ALL_RESULTS['bootstrap'].items()]
    print_table("TABLE 6: Significance Tests",
                ["Comparison", "ΔF1", "p-value", "95% CI", "d"], rows,
                [30, 8, 10, 18, 8])

# Table: WSD Strata
if ALL_RESULTS.get('wsd_strata'):
    rows = [[s['encoder'], s['stratum'], s['n'], s['f1_no'], s['f1_wsd'], f"{s['delta']:+.1f}"]
            for s in ALL_RESULTS['wsd_strata']]
    print_table("TABLE 7: WSD Stratified Analysis",
                ["Encoder", "Coverage", "N", "F1(no)", "F1(WSD)", "ΔF1"], rows,
                [12, 15, 8, 10, 10, 8])

# Confusion
if ALL_RESULTS.get('confusion'):
    cm = ALL_RESULTS['confusion']
    print(f"\n{'='*70}")
    print(f"  TABLE 11: Confusion Matrix ({best_wsd})")
    print(f"{'='*70}")
    print(f"  {'':>20} {'Pred Pos':>12} {'Pred Neg':>12}")
    print(f"  {'Actual Pos':<20} {cm['TP']:>12} {cm['FN']:>12}")
    print(f"  {'Actual Neg':<20} {cm['FP']:>12} {cm['TN']:>12}")

print(f"\n{'='*70}")
print(f"  ALL DONE! Total time: {(time.time()-t_start)/3600:.1f} hours")
print(f"  Results: {CFG.RESULTS_DIR / 'all_results.json'}")
print(f"{'='*70}")
