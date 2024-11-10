import torch
from typing import List, Tuple


class TextTransform:
    """Maps characters to integers and vice versa with optimized processing"""

    def __init__(self):
        base_map = """' 0\n<SPACE> 1\na 2\nb 3\nc 4\nd 5\ne 6\nf 7\ng 8\nh 9\ni 10\nj 11\nk 12\nl 13\nm 14\nn 15\no 16\np 17\nq 18\nr 19\ns 20\nt 21\nu 22\nv 23\nw 24\nx 25\ny 26\nz 27"""

        # Pre-compute char_map and index_map using dictionary comprehensions
        self.char_map = {
            line.split()[0]: int(line.split()[1]) for line in base_map.split("\n")
        }
        self.index_map = {v: k for k, v in self.char_map.items()}
        self.index_map[1] = " "  # Special case for space

        # Pre-compute diacritic mapping
        self.diacritic_map = {
            char: replacement
            for chars, replacement in [
                ("áàäâãåāăąǎ", "a"),
                ("éèëêēěęė", "e"),
                ("íìïîīįǐ", "i"),
                ("óòöôõōøőǒ", "o"),
                ("úùüûūůűǔ", "u"),
                ("çčć", "c"),
                ("ñń", "n"),
                ("žźż", "z"),
                ("ğģǵ", "g"),
                ("łľļ", "l"),
                ("šśşș", "s"),
                ("ťț", "t"),
                ("ýÿ", "y"),
            ]
            for char in chars
        }

        # Add special character mappings
        special_chars = {
            "æ": "ae",
            "œ": "oe",
            "å": "a",
            "ø": "o",
            "đ": "d",
            "ʃ": "sh",
            "ʒ": "zh",
            "ð": "d",
            "þ": "th",
            "ŋ": "n",
            "ɖ": "d",
            "ă": "a",
            "ą": "a",
            "ơ": "o",
            "ư": "u",
            "ắ": "a",
            "à": "a",
        }
        self.diacritic_map.update(special_chars)

        # Pre-compute the default space value
        self.space_value = self.char_map["<SPACE>"]

        # Cache for processed characters
        self._char_cache = {}

    def _process_char(self, c: str) -> int:
        """Process a single character with caching"""
        if c in self._char_cache:
            return self._char_cache[c]

        # Convert diacritics first, then look up in char_map
        normalized_char = self.diacritic_map.get(c, c)
        result = self.char_map.get(normalized_char, self.space_value)

        self._char_cache[c] = result
        return result

    def text_to_int(self, text: str) -> List[int]:
        """Optimized conversion of text to integer sequence"""
        return [self._process_char(c) for c in text.lower()]

    def int_to_text(self, labels: List[int]) -> str:
        """Optimized conversion of integer labels to text"""
        return "".join(self.index_map[i] for i in labels).replace("<SPACE>", " ")


def GreedyDecoder(
    output: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    blank_label: int = 28,
    collapse_repeated: bool = True,
) -> Tuple[List[str], List[str]]:
    """Optimized greedy decoder"""
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []

    # Process targets first
    for i, length in enumerate(label_lengths):
        targets.append(text_transform.int_to_text(labels[i][:length].tolist()))

    # Process decodes
    for args in arg_maxes:
        if collapse_repeated:
            # Use tensor operations for faster processing
            mask = torch.ones_like(args, dtype=torch.bool)
            mask[1:] = args[1:] != args[:-1]
            args = args[mask]

        # Filter out blank labels
        decode = [idx.item() for idx in args if idx != blank_label]
        decodes.append(text_transform.int_to_text(decode))

    return decodes, targets


# Initialize TextProcess for text processing
text_transform = TextTransform()
