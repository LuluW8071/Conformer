import torch

class TextTransform:
    """ Maps characters to integers and vice versa """
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '


    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        special_chars = {' ', '–', '-', '"', '`', '(', ')', ',', ':', ';', '?', '!', '’', '‘', '“', '”', '…',
                         '«', '»', '[', ']', '{', '}', '&', '*', '#', '@', '%', '$', '^', '=', '|', '_', '+', '<',
                         '>', '~', '.', 'ł', '\t', '�', 'ß'}
        diacritic_map = {
                'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a',
                'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
                'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
                'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o', 'ō': 'o',
                'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
                'ç': 'c',
                'ñ': 'n',
                'ž': 'z', 'ź': 'z',
                'ğ': 'g', 'ģ': 'g', 'ğ': 'g' 
            }
        for c in text:
            if c in special_chars:
                ch = self.char_map['<SPACE>']
            # elif c in diacritic_map:
            #     ch = self.char_map[diacritic_map[c]]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence


    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to a text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i]) 
        return ''.join(string).replace('<SPACE>', ' ')

# import re

# class TextTransform:
#     """ Maps characters to integers and vice versa """
#     def __init__(self):
#         char_map_str = """
#         ' 0
#         <SPACE> 1
#         a 2
#         b 3
#         c 4
#         d 5
#         e 6
#         f 7
#         g 8
#         h 9
#         i 10
#         j 11
#         k 12
#         l 13
#         m 14
#         n 15
#         o 16
#         p 17
#         q 18
#         r 19
#         s 20
#         t 21
#         u 22
#         v 23
#         w 24
#         x 25
#         y 26
#         z 27
#         """
#         self.char_map = {}
#         self.index_map = {}
#         for line in char_map_str.strip().split('\n'):
#             ch, index = line.split()
#             self.char_map[ch] = int(index)
#             self.index_map[int(index)] = ch
#         self.index_map[1] = ' '  # Map <SPACE> to an actual space character

#         # Predefined sets and mappings for text cleaning
#         self.special_chars = {' ', '–', '-', '"', '`', '(', ')', ',', ':', ';', '?', '!', '’', '‘', '“', '”', '…',
#                               '«', '»', '[', ']', '{', '}', '&', '*', '#', '@', '%', '$', '^', '=', '|', '_', '+', '<',
#                               '>', '~', '.', 'ł', '\t', '�', 'ß'}
#         self.diacritic_map = {
#             'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a',
#             'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
#             'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
#             'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o', 'ō': 'o',
#             'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
#             'ç': 'c',
#             'ñ': 'n',
#             'ž': 'z', 'ź': 'z',
#             'ğ': 'g', 'ģ': 'g'
#         }
#         # Compile regex to substitute diacritics in one pass
#         self.diacritic_re = re.compile('|'.join(re.escape(k) for k in self.diacritic_map.keys()))

#     def clean_diacritics(self, text):
#         # Substitute diacritics based on the mapping
#         return self.diacritic_re.sub(lambda x: self.diacritic_map[x.group()], text)

#     def text_to_int(self, text):
#         """ Convert text to an integer sequence with optimized character mapping. """
#         cleaned_text = self.clean_diacritics(text)

#         # Convert characters to integers, using <SPACE> for any special character
#         int_sequence = [
#             self.char_map.get(c, self.char_map.get('<SPACE>')) if c in self.special_chars else self.char_map.get(c, None)
#             for c in cleaned_text if self.char_map.get(c, None) is not None
#         ]
#         return int_sequence

#     def int_to_text(self, labels):
#         """ Convert integer labels to a text sequence using character map. """
#         return ''.join(self.index_map.get(i, '') for i in labels).replace('<SPACE>', ' ')


# Initialize TextProcess for text processing
text_transform = TextTransform()

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j-1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets
