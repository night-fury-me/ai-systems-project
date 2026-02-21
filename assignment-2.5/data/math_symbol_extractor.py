import re
import json
import gzip
import html
import unicodedata
from collections import Counter
from tqdm import tqdm  # For progress tracking

def is_math_symbol(char):
    try:
        category = unicodedata.category(char)
        return category in ('Sm', 'So', 'Sk', 'Lu', 'Ll', 'Lm') and not char.isalnum()
    except TypeError:
        return False

def extract_math_symbols(formula):
    normalized = unicodedata.normalize('NFKC', html.unescape(formula))
    
    symbols = []
    for char in normalized:
        if is_math_symbol(char):
            symbols.append(char)
    return symbols

class MathSymbolExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.symbol_counter = Counter()
        
    def process_batch(self, formulas):
        for formula in formulas:
            symbols = extract_math_symbols(formula)
            self.symbol_counter.update(symbols)
    
    def extract(self):
        with gzip.open(self.file_path, 'rt') as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)  # Reset file pointer
            
            batch = []
            with tqdm(total=total_lines, desc="Processing formulas", ascii="->") as pbar:
                for line in f:
                    data = json.loads(line)
                    batch.extend(data["formulas"])
                    if len(batch) >= 1000:
                        self.process_batch(batch)
                        batch = []
                    pbar.update(1)
                if batch:
                    self.process_batch(batch)
        return self
    
    def get_symbols(self, min_frequency=10):
        return {s: c for s, c in self.symbol_counter.items() if c >= min_frequency}

if __name__ == "__main__":
    extractor = MathSymbolExtractor("data/training-data.jsonl.gz").extract()
    
    common_symbols = extractor.get_symbols(min_frequency=2)
    print(f"Found {len(common_symbols)} unique math symbols")
    
    # with open("data/math_symbols.json", "w") as f:
    #     json.dump(common_symbols, f, ensure_ascii=False, indent=2)
    
    # print("\nTop 50 symbols:")
    # for symbol, count in sorted(common_symbols.items(), key=lambda x: -x[1])[:50]:
    #     print(f"{symbol}: {count} (U+{ord(symbol):04X})")


    # Add mathematical special tokens
    math_symbols_1 = [
        "∑", "∫", "∏", "∮", "≠", "≈", "≡", "≤", "≥", "±", 
        "∇", "∂", "∞", "∈", "∉", "⊂", "⊃", "∪", "∩", "∅",
        "→", "↔", "∀", "∃", "∴", "∵", "∧", "∨", "¬", "⊕",
        "⊗", "⊥", "∠", "∥", "≅", "∼", "≜", "⨯", "√", "∛"
    ]

    math_symbols_2 = [
        "!", "\"", "#", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@",
        "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "±", "°", "¼", "½", "¾", "÷", "×", "∅", "∈", "∉", "∋",
        "∝", "∞", "√", "∑", "∏", "∫", "∂", "∇", "∧", "∨", "¬", "⊕", "⊗", "⊥", "⊢", "⊨", "Γ", "Δ", "Θ", "Λ", "Ξ",
        "Π", "Σ", "Υ", "Φ", "Ψ", "Ω", "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "π",
        "ρ", "ς", "σ", "τ", "υ", "φ", "χ", "ψ", "ω", "ϑ", "ϕ", "ϖ", "ϰ", "ϱ", "ϵ", "Å", "ß", "æ", "ø", "Œ", "ð",
        "þ", "ı", "Ł", "ł", "ȷ", "ˇ", "˘", "˙", "̃", "̊", "̸", "¡", "£", "¥", "¦", "§", "¨", "¯", "´", "µ", "¶", "·",
        "¸", "¿", "Â", "Ê", "â", "è", "é", "ï", "ö", "Ĝ", "Ṁ", "–", "’", "“", "”", "†", "‡", "•", "…", "‰", "′", "″",
        "\u2061", "\u2062", "\u2063", "\u2064", "⃡", "ℂ", "ℋ", "ℌ", "ℍ", "ℎ", "ℏ", "ℐ", "ℑ", "ℒ", "ℓ", "ℕ", "℘",
        "ℙ", "ℚ", "ℛ", "ℜ", "ℝ", "ℤ", "Ω", "℧", "ℬ", "ℭ", "ℰ", "ℱ", "ℳ", "ℵ", "ℶ", "ⅆ", "ⅇ", "ⅈ", "←", "↑", "→",
        "↓", "↔", "↘", "↝", "↦", "↪", "↫", "↬", "↶", "↺", "↽", "↾", "⇀", "⇄", "⇆", "⇈", "⇉", "⇌", "⇢", "⇐",
        "⇑", "⇒", "⇓", "⇔", "∀", "∃", "∄", "∐", "−", "∓", "∔", "∖", "∗", "∘", "∙", "∠", "∡", "∢", "∣", "∤",
        "∥", "∩", "∪", "∬", "∭", "∮", "∼", "∽", "≀", "≂", "≃", "≅", "≈", "≊", "≌", "≍", "≐", "≑", "≔", "≕", "≜",
        "≠", "≡", "≢", "≤", "≥", "≦", "≧", "≪", "≫", "≲", "≳", "≶", "≺", "≻", "≽", "⊀", "⊂", "⊃", "⊄", "⊆", "⊇",
        "⊈", "⊊", "⊎", "⊑", "⊓", "⊔", "⊖", "⊘", "⊙", "⊛", "⊞", "⊟", "⊠", "⊤", "⊧", "⊳", "⊵", "⊺", "⋀", "⋁", "⋂",
        "⋃", "⋄", "⋅", "⋆", "⋉", "⋊", "⋋", "⋐", "⋘", "⋮", "⋯", "⋱", "⌈", "⌉", "⌊", "⌋", "⌞", "⌟", "⌢", "⌣",
        "⏞", "⏟", "□", "△", "▷", "▽", "◁", "◆", "○", "☉", "♠", "♡", "♢", "♣", "♮", "♯", "➀", "➁", "⟂", "⟦",
        "⟧", "⟨", "⟩", "⟶", "⟷", "⟸", "⟹", "⟺", "⟼", "⩽", "⩾", "⪅", "⪆", "⪯", "⪰", "⫋", "⫽", "ﬀ", "ﬁ", "／"
    ]
    
    additional_symbol = set(math_symbols_1) | set(math_symbols_2) 
    print("Total additional symbols:", len(additional_symbol))
    additional_symbol = additional_symbol - set(common_symbols.keys())
    print("Total additional symbols after removing common symbols:", len(additional_symbol))

    additional_symbol = {(sym, 10) for sym in additional_symbol}

    symbols = common_symbols | dict(additional_symbol)

    print("Total symbols:", len(symbols))
    print("-----------------------------------")

    with open("data/special_symbols.json", "w") as f:
        json.dump(symbols, f, ensure_ascii=False, indent=2)

    print("-----------------------------------")
