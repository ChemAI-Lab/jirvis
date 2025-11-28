import re
import json
from typing import List, Dict, Tuple, Optional
import re
import json
from typing import List, Dict, Tuple, Optional


class NMRTokenizer:

    def __init__(
        self,
        max_length: int = 227,
        vocab_path: Optional[str] = None,
        split_decimals: bool = False,
    ):
        self.max_length = max_length
        self.split_decimals = split_decimals
        self.vocab = {"[PAD]": 0, "[scalar]": 1}  # Add [scalar] token
        self.id_to_token = {0: "[PAD]", 1: "[scalar]"}

        # Special token patterns
        self.special_patterns = [
            (r"\d+H NMR:", "NMR_TYPE"),
            (r"\d+C NMR:", "NMR_TYPE"),
            (r"J\s*=", "J_COUPLING"),
        ]

        # Load existing vocab if provided
        if vocab_path:
            self.load_vocab(vocab_path)

    def _find_special_tokens(self, text: str) -> List[Tuple[int, int, str, str]]:
        special_matches = []
        for pattern, token_type in self.special_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                token = match.group()
                special_matches.append((start, end, token, token_type))
        return sorted(special_matches, key=lambda x: x[0])

    def _find_numeric_tokens(
        self, text: str, special_matches: List[Tuple[int, int, str, str]]
    ) -> List[Tuple[int, int, str, float]]:
        if self.split_decimals:
            # Pattern that captures decimal numbers but we'll split them manually
            number_pattern = r"\d+(?:\.\d+)?"
        else:
            # Original pattern - keep decimals together
            number_pattern = r"\d+(\.\d+)?"

        number_matches = []

        for match in re.finditer(number_pattern, text):
            start, end = match.span()
            number = match.group()

            is_inside_special = any(
                start >= special_start and end <= special_end
                for special_start, special_end, _, _ in special_matches
            )

            if not is_inside_special:
                if self.split_decimals and "." in number:
                    # Split decimal number into parts
                    parts = number.split(".")
                    current_pos = start

                    # Add integer part
                    integer_part = parts[0]
                    number_matches.append(
                        (
                            current_pos,
                            current_pos + len(integer_part),
                            "[scalar]",
                            float(integer_part),
                        )
                    )
                    current_pos += len(integer_part)

                    # Add decimal point - treat as special token, not scalar
                    number_matches.append((current_pos, current_pos + 1, ".", 0.0))
                    current_pos += 1

                    # Add fractional part
                    fractional_part = parts[1]
                    number_matches.append(
                        (
                            current_pos,
                            current_pos + len(fractional_part),
                            "[scalar]",
                            float(fractional_part),
                        )
                    )
                else:
                    # Keep number as is (either no decimal or split_decimals=False)
                    number_matches.append((start, end, "[scalar]", float(number)))

        return number_matches

    def _build_tokens(self, text: str) -> Tuple[List[str], List[float]]:
        special_matches = self._find_special_tokens(text)
        number_matches = self._find_numeric_tokens(text, special_matches)

        all_matches = []
        for start, end, token, _ in special_matches:
            all_matches.append((start, end, token, "special", 0.0))
        for start, end, token, value in number_matches:
            all_matches.append((start, end, token, "number", value))

        all_matches.sort(key=lambda x: x[0])

        tokens = []
        scalar_values = []
        last_end = 0

        for start, end, token, match_type, value in all_matches:
            text_before = text[last_end:start]
            if text_before.strip():
                text_tokens = [t for t in text_before.split() if t]
                tokens.extend(text_tokens)
                scalar_values.extend([0.0] * len(text_tokens))

            tokens.append(token)
            scalar_values.append(value)
            last_end = end

        remaining_text = text[last_end:]
        if remaining_text.strip():
            text_tokens = [t for t in remaining_text.split() if t]
            tokens.extend(text_tokens)
            scalar_values.extend([0.0] * len(text_tokens))

        return tokens, scalar_values

    def _update_vocab(self, tokens: List[str]) -> None:
        for token in tokens:
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.id_to_token[token_id] = token

    def tokenize(
        self, h_nmr_text: str, c_nmr_text: str, update_vocab: bool = True
    ) -> Dict[str, List]:
        full_text = h_nmr_text + " [SEP] " + c_nmr_text
        tokens, scalar_values = self._build_tokens(full_text)

        if update_vocab:
            self._update_vocab(tokens)

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            # Truncate if too long
            tokens = tokens[: self.max_length]
            scalar_values = scalar_values[: self.max_length]
            padded_tokens = tokens
            padded_scalar_values = scalar_values
        else:
            # Pad if too short
            padding_length = self.max_length - len(tokens)
            padded_tokens = tokens + ["[PAD]"] * padding_length
            padded_scalar_values = scalar_values + [0.0] * padding_length

        input_ids = [
            self.vocab.get(token, self.vocab["[PAD]"]) for token in padded_tokens
        ]
        attention_mask = [1 if token != "[PAD]" else 0 for token in padded_tokens]
        scalar_mask = [1 if token == "[scalar]" else 0 for token in padded_tokens]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scalar_values": padded_scalar_values,
            "scalar_mask": scalar_mask,
            "tokens": tokens,  # Return original tokens for debugging
        }

    def batch_tokenize(
        self,
        h_nmr_texts: List[str],
        c_nmr_texts: List[str],
        update_vocab: bool = True,
        max_samples: Optional[int] = None,
    ) -> Dict[str, List[List]]:
        if len(h_nmr_texts) != len(c_nmr_texts):
            raise ValueError("h_nmr_texts and c_nmr_texts must have the same length")

        # Limit samples if max_samples is specified
        if max_samples is not None:
            h_nmr_texts = h_nmr_texts[:max_samples]
            c_nmr_texts = c_nmr_texts[:max_samples]

        batch_results = {
            "input_ids": [],
            "attention_mask": [],
            "scalar_values": [],
            "scalar_mask": [],
            "tokens": [],
        }

        for h_nmr, c_nmr in zip(h_nmr_texts, c_nmr_texts):
            result = self.tokenize(h_nmr, c_nmr, update_vocab=update_vocab)
            for key in batch_results:
                batch_results[key].append(result[key])

        return batch_results

    def save_vocab(self, vocab_file: str) -> None:
        vocab_data = {
            "vocab": self.vocab,
            "id_to_token": self.id_to_token,
            "max_length": self.max_length,
            "split_decimals": self.split_decimals,  # Save the setting
            "vocab_size": len(self.vocab),
        }

        with open(vocab_file, "w") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    def load_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, "r") as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data["vocab"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}

        # Ensure [scalar] token exists
        if "[scalar]" not in self.vocab:
            self.vocab["[scalar]"] = len(self.vocab)
            self.id_to_token[len(self.id_to_token)] = "[scalar]"

        if "max_length" in vocab_data:
            self.max_length = vocab_data["max_length"]

        # Load split_decimals setting if saved, otherwise keep current setting
        if "split_decimals" in vocab_data:
            self.split_decimals = vocab_data["split_decimals"]
            print(f"Loaded tokenizer with split_decimals={self.split_decimals}")
        else:
            print(
                f"No split_decimals setting found in vocab file, using current setting: {self.split_decimals}"
            )

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_tokenization_mode(self) -> str:
        """Return a string describing the current tokenization mode"""
        return "decimal_split" if self.split_decimals else "decimal_together"
