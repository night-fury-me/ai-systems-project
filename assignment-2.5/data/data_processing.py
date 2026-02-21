import json
import gzip
import re
import numpy as np # type: ignore
from collections import Counter, defaultdict

from tqdm import tqdm # type: ignore
from tokenizers import Tokenizer, trainers, models # type: ignore
from tokenizers import pre_tokenizers, processors, normalizers # type: ignore

import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore

def remove_xml_attributes(formula):
    # Pattern to match XML tags with attributes
    pattern = re.compile(r"<([^\s>]+)(\s[^>]*)?>")
    
    # Replace tags with attributes with clean tags
    cleaned = pattern.sub(r"<\1>", formula)
    
    # Handle self-closing tags
    cleaned = re.sub(r"<\/?([^>]+)\s*\/>", r"<\1>", cleaned)
    cleaned = re.sub(r'-?\d*\.?\d+([eE][+-]?\d+)?', lambda m: process_number(m[0]), cleaned)

    return cleaned

def process_number(num_str):
    components = []
    
    # Handle sign
    if num_str.startswith('-'):
        components.append("[NEG]")
        num_str = num_str[1:]
        
    # Split into parts
    if 'e' in num_str or 'E' in num_str:
        base, exp = num_str.lower().split('e')
        components.extend(["[SCI]", split_digits(base), "[EXP]", exp])
    elif '.' in num_str:
        int_part, frac_part = num_str.split('.')
        components.extend(["[FLOAT]", int_part, "[DECIMAL]", frac_part])
    else:
        components.extend(["[INT]", num_str])
        
    return ' '.join(components)

def split_digits(num_part):
    return ' '.join(num_part)

class MathDatasetCleaner:
    def __init__(self, file_path=None, label_file_path=None):
        self.file_path = file_path
        self.label_file_path  = label_file_path
        self.cleaned_formulas = []
        self.cleaned_dataset  = []
        self.labels           = []

    def parse_server_data(self, batch_data):
        for data in tqdm(batch_data, desc="Parsing file", ascii="->"):
            cleaned_forms = [remove_xml_attributes(f) for f in data]
            self.cleaned_dataset.extend([{
                "doc_id"  : "paper_1234",
                "formulas": cleaned_forms,
                "label"   : "astro-ph"
            }])

            self.cleaned_formulas.extend(cleaned_forms)

        # print(f"Cleaned {len(self.cleaned_formulas)} formulas")
        return self.cleaned_dataset, self.cleaned_formulas
        
    def parse(self):
        with gzip.open(self.file_path, 'rt') as f:
            for line in tqdm(f, desc="Parsing file"):
                data = json.loads(line)
                cleaned_forms = [remove_xml_attributes(f) for f in data["formulas"]]
                if self.label_file_path is None:
                    paper_id = re.search(r'(\d+)', data["paper"])
                    if paper_id:
                        paper_id = paper_id.group(1)
                        self.cleaned_dataset.extend([{
                            "doc_id"  : paper_id,
                            "formulas": cleaned_forms,
                            "label"   : data["classification"]
                        }])
                        self.labels.extend([data["classification"]])
                    else:
                        print("PAPER ID NOT FOUND!!!!!")
                else:
                    with open(self.label_file_path, 'rt') as f:
                        labels = json.load(f)
                        self.cleaned_dataset.extend([{
                            "doc_id"  : data["id"],
                            "formulas": cleaned_forms,
                            "label"   : labels[data["id"]]
                        }])
                        self.labels.extend([labels[data["id"]]])

                self.cleaned_formulas.extend(cleaned_forms)
        
        self.labels = list(set(self.labels))

        print(f"Cleaned {len(self.cleaned_formulas)} formulas")
        return self.cleaned_dataset, self.cleaned_formulas, self.labels
    
    def _get_xml_tags(self, formula):
        # Pattern to match all XML tags
        pattern = re.compile(r"<[^>]+>")
        
        # Find all matches
        tags = pattern.findall(formula)
        return tags
    
    def get_xml_tags(self):
        all_xml_tags = []
        with gzip.open(self.file_path, 'rt') as f:
            for line in tqdm(f, desc="Parsing file"):
                data = json.loads(line)
                cleaned_forms = [remove_xml_attributes(f) for f in data["formulas"]]
                for formula in cleaned_forms:
                    xml_tags = self._get_xml_tags(formula)
                    all_xml_tags.extend(xml_tags)
        all_xml_tags = list(set(all_xml_tags))
        return all_xml_tags
    
    @staticmethod
    def get_xml_tags(formulas):
        all_xml_tags = []
        pattern = re.compile(r"<[^>]+>")
        for formula in formulas:
            xml_tags = tags = pattern.findall(formula)
            all_xml_tags.extend(xml_tags)

        all_xml_tags = list(set(all_xml_tags))
        return all_xml_tags

class MathXMLTokenizer:
    def __init__(self, formulas, vocab_size=30000, additional_sym_path=None, min_frequency=10):
        self.formulas = formulas
        self.tokenizer = Tokenizer(models.BPE())
        self.vocab_size = vocab_size
        self.additional_sym_paths = additional_sym_path
        self.min_frequency = min_frequency

        self._add_special_tokens()

        # Configure XML-aware pre-tokenizer
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([            
            pre_tokenizers.Split(r"(<\/?[\w]+>)", behavior="isolated"),  # Split XML tags
            pre_tokenizers.ByteLevel(add_prefix_space=False)
        ])
    
    def _add_special_tokens(self):
        self.special_tokens = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]",
            "[NUM]", "[INT]", "[FLOAT]", "[SCI]",
            "[NEG]", "[DECIMAL]", "[EXP]"
        ]

        for path in self.additional_sym_paths:
            with open(path, "r") as f:
                symbols = json.load(f)
                if "xml_tags" in symbols.keys():
                    self.special_tokens.extend(symbols["xml_tags"])
                else:
                    for symbol, cnt in symbols.items():
                        if cnt >= self.min_frequency:
                            self.special_tokens.extend(symbol)

        self.special_tokens = list(set(self.special_tokens))

    def train(self):
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC(),  # Unicode normalization
        ])

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        self.tokenizer.train_from_iterator(tqdm(desc="BPE Training: ", iterable=self.formulas), trainer=trainer)

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self.tokenizer.token_to_id("[BOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ]
        )
        return self
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self):
        return len(self.get_vocab())
    
    def save(self, path):
        self.tokenizer.save(path)

# 4. PyTorch Dataset

class MathDataset(Dataset):
    def __init__(self, documents, tokenizer, max_formulas=10, max_length=150):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_formulas = max_formulas
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.unk_token_id = tokenizer.token_to_id("[UNK]")
        self.max_length = max_length
        self.vocab_size = len(tokenizer.get_vocab())

        arxiv_classes = [
            'astro-ph', 'cond-mat', 'cs', 'eess', 'gr-qc', 'hep-ex',
            'hep-lat', 'hep-ph', 'hep-th', 'math', 'nlin', 'nucl-ex',
            'nucl-th', 'physics', 'q-bio', 'q-fin', 'quant-ph', 'stat'
        ]

        self.n_classes      = len(arxiv_classes)
        self.label_to_id    = {label: idx for idx, label in enumerate(arxiv_classes)}
        self.id_to_label    = {idx: label for idx, label in enumerate(arxiv_classes)}

    def __len__(self):
        return len(self.documents)

    def get_label(self, idx):
        return self.id_to_label[idx]

    def __getitem__(self, idx):
        doc_formulas = self.documents[idx]["formulas"]
        label = self.label_to_id[self.documents[idx]["label"]]
        
        processed_formulas = torch.full(
            (self.max_formulas, self.max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long
        )

        # print(f"{self.pad_token_id=}")
        
        # Fill with actual formulas (up to max_formulas)
        for i, formula in enumerate(doc_formulas[:self.max_formulas]):
            encoded = self.tokenizer.encode(formula)
            # if len(encoded.ids) <= self.max_length:
            truncated = encoded.ids[:self.max_length]
            # print(f"{encoded.tokens=}")
            # print(f"{encoded.tokens[:self.max_length]=}")
            # max_id = max(truncated)
            # print(f"Max id={max_id}")
            # Create padded formula
            padded = truncated + [self.pad_token_id] * (self.max_length - len(truncated))
            
            # Convert to tensor and insert into processed_formulas
            processed_formulas[i] = torch.tensor(padded, dtype=torch.long)
            # break
        
        max_id = torch.max(processed_formulas)
        # print(f"Max id={max_id}")
        assert max_id < 7500, "Exists an id larger then 7500"

        doc_id = re.findall(r'\d+', self.documents[idx]["doc_id"])
        doc_id = int(''.join(doc_id))
        return {
            "input": processed_formulas,  # Shape [max_formulas, max_length]
            "label": torch.tensor([label], dtype=torch.long),  # Shape [1],
            "doc_id":  torch.tensor([doc_id], dtype=torch.long)
        }

# Usage Example
if __name__ == "__main__":
    print("Data processing script")
    # data/test-data.jsonl.gz
    # train_data, train_formulas, labels  = MathDatasetCleaner("data/training-data.jsonl.gz").parse()
    # print(f"labels cnt = {len(labels)}")
    # print(f"{labels=}")

    # xml_tags  = MathDatasetCleaner("data/training-data.jsonl.gz").get_xml_tags()
    # print(f"Found {len(xml_tags)=} xml tags!")

    # with open("data/xml_tags.json", "w+", encoding='utf-8') as f:
    #     json.dump({ "xml_tags" : xml_tags}, f, indent=4)

    # train_data, train_formulas, _  = MathDatasetCleaner("data/training-data.jsonl.gz").parse()
    # test_data, test_formulas, _    = MathDatasetCleaner("data/example-test-data.jsonl.gz", label_file_path="data/example-test-results.json").parse()

    # combined_formulas = train_formulas + test_formulas

    # xml_tags = MathDatasetCleaner.get_xml_tags(combined_formulas)
    # print(f"Found {len(xml_tags)} xml tags!")
    # with open("data/xml_tags.json", "w+", encoding='utf-8') as f:
    #     json.dump({ "xml_tags" : xml_tags}, f, indent=4)


    # # Train tokenizer
    # tokenizer = MathXMLTokenizer(
    #     combined_formulas, 
    #     vocab_size=7500, 
    #     additional_sym_path=[
    #         "data/special_symbols.json",
    #         "data/xml_tags.json"
    #     ]).train()
    

    # print(f"Vocab size = {tokenizer.get_vocab_size()}")

    # tokenizer.save("data/mathml_tokenizer.json")

    # # # max_n_formulas  = max([len(data["formulas"]) for data in train_data])
    # # max_b_formulas = 0
    # # for data in train_data:
    # #     max_b_formulas = max(max_b_formulas, len(data["formulas"]))
    # # print(f"Max number of formulas: {max_b_formulas}")

    # print("-----------------------------------")
    # max_form_length = 0
    # freq_counter = Counter()
    # for form in tqdm(desc="Formula length caulculation: ", iterable=combined_formulas):
    #     form_length = len(tokenizer.tokenizer.encode(form).tokens)
    #     freq_counter.update([form_length])
    
    # sorted_freq = dict(sorted(freq_counter.items(), key=lambda item: (-item[1], -item[0])))
    # with open("data/form_length_freq.json", "w") as f:
    #     json.dump(sorted_freq, f, indent=4)
    # print("-----------------------------------")

    # with open("data/form_length_freq.json", "r") as f:
    #     sorted_freq = json.load(f)
    #     taken = sum([val for key, val in sorted_freq.items() if int(key) <= 400])
    #     total = sum(sorted_freq.values())
    #     print(f"Taken formula: {taken}")
    #     print(f"In percentage: {taken * 100 / total}%")


    # taken_formulas = sum([val for key, val in sorted_freq.items() if key <= 200])
    # print(f"Taken formula: {taken_formulas}")
    # print(f"In percentage: {taken_formulas * 100 / len(combined_formulas)}%")
    # # print(f"Max number of formulas: {max_n_formulas}")

    # # Test transformation
    # test_input = '''<math><mrow id="S1.p1.2.m2.1.1" xref="S1.p1.2.m2.1.1.cmml"><msub id="S1.p1.2.m2.1.1.2" xref="S1.p1.2.m2.1.1.2.cmml"> -e^1231.234'''
    # cleaned = remove_xml_attributes(test_input)

    # print("Cleaned XML:", cleaned)
    
    # # Test tokenization
    # encoding = tokenizer.tokenizer.encode(cleaned)
    # print("Tokens:", encoding.tokens)
    
    # print("Getting vocab...")
    # vocab = tokenizer.get_vocab()
    # print("Vocab size:", len(vocab))
    # # print("Voab sample:", list(vocab.items()))

    # print("-----------------------------------")
    # words = set()
    # for word, _ in vocab.items():
    #     # print(word)
    #     words.add(word)
    # print("-----------------------------------")
    # print("Total words:", len(words))
    # print("-----------------------------------")
    
    # Mathematical special tokens
    # math_symbols = [
    #     "∑", "∫", "∏", "∮", "≠", "≈", "≡", "≤", "≥", "±", 
    #     "∇", "∂", "∞", "∈", "∉", "⊂", "⊃", "∪", "∩", "∅",
    #     "→", "↔", "∀", "∃", "∴", "∵", "∧", "∨", "¬", "⊕",
    #     "⊗", "⊥", "∠", "∥", "≅", "∼", "≜", "⨯", "√", "∛"
    # ]

    # print(f"\nSymbol test:")
    # for s in math_symbols:
    #     encoded = tokenizer.tokenizer.encode(s)
    #     print(f"{s} → {encoded.tokens} (Length: {len(encoded.tokens)})")