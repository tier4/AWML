# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Optional, Tuple, Union

import nltk
from mmdet.models.detectors.glip import GLIP
from mmdet.registry import MODELS
from mmdet.utils import ConfigType


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a text.
    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    punctuation = [
        "|",
        ":",
        ";",
        "@",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "^",
        "'",
        '"',
        "’",
        "`",
        "?",
        "$",
        "%",
        "#",
        "!",
        "&",
        "*",
        "+",
        ",",
        ".",
    ]
    for p in punctuation:
        text = text.replace(p, "")
    return text


def clean_label_name(name: str) -> str:
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.
    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption) # ['cat', 'a remote', 'the picture']
    """

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == "NP":
            noun_phrases.append(" ".join(t[0] for t in subtree.leaves()))

    return noun_phrases


def run_ner(caption: str) -> Tuple[list, list]:
    """Run NER on a caption and return the tokens and noun phrases.
    Args:
        caption (str): The input caption.

    Returns:
        Tuple[List, List]: A tuple containing the tokens and noun phrases.
            - tokens_positive (List): A list of token positions.
            - noun_phrases (List): A list of noun phrases.
    """
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != ""]
    print("noun_phrases:", noun_phrases)
    relevant_phrases = noun_phrases
    labels = noun_phrases

    tokens_positive = []
    for entity, label in zip(relevant_phrases, labels):
        try:
            # search all occurrences and mark them as different entities
            # TODO: Not Robust
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except Exception:
            print("noun entities:", noun_phrases)
            print("entity:", entity)
            print("caption:", caption.lower())
    return tokens_positive, noun_phrases


@MODELS.register_module()
class GLIP_FIXED(GLIP):
    def __init__(self, *args, **kwargs):
        nltk.download("punkt", download_dir="/usr/share/nltk_data")
        nltk.download("averaged_perceptron_tagger", download_dir="/usr/share/nltk_data")
        super().__init__(*args, **kwargs)

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None,
    ) -> Tuple[dict, str, list, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(original_caption)

            tokenized = self.language_model.tokenizer([caption_string], return_tensors="pt")
            entities = original_caption
        else:
            original_caption = original_caption.strip(self._special_tokens)
            tokenized = self.language_model.tokenizer([original_caption], return_tensors="pt")
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities
