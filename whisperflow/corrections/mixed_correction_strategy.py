import os
import re
from hunspell import Hunspell
from rapidfuzz import process, fuzz

from .correction_strategy import CorrectionStrategy

language_dic = {
    "portuguese": "pt_BR"
}

class MixedCorrectionStrategy(CorrectionStrategy):
    """
    Estratégia mista: Hunspell para palavras comuns e RapidFuzz para termos médicos.
    """

    DICTIONARY_DIR = './whisperflow/corrections/spell_dicts'

    # Caches estáticos (por classe) para evitar recarregar os dicionários mais de uma vez.
    _hunspell_cache = {}
    _clinical_terms_cache = {}

    def __init__(self, language: str = 'portuguese', similarity_threshold: int = 85):
        super().__init__(language)
        self.similarity_threshold = similarity_threshold
        self.hunspell = None
        self.clinical_terms = []
        self.is_active = False

        language_selected = (
            None
            if language is None
            else language_dic.get(language.lower())
        )
        if language_selected is not None:
            self.is_active = True
            self.load_resources(language_selected)

    def load_resources(self, language: str):
        """Carrega recursos de acordo com o idioma definido usando cache para evitar recarga."""
        
        # Verifica se o Hunspell já foi carregado para este idioma
        if language in self._hunspell_cache:
            self.hunspell = self._hunspell_cache[language]
        else:
            self.hunspell = Hunspell(language, hunspell_data_dir=self.DICTIONARY_DIR)
            self._hunspell_cache[language] = self.hunspell

        # Verifica se os termos médicos já foram carregados para este idioma
        if language in self._clinical_terms_cache:
            self.clinical_terms = self._clinical_terms_cache[language]
        else:
            self.clinical_terms = self._load_terms_from_dic(language)
            self._clinical_terms_cache[language] = self.clinical_terms

        print(f"Recursos carregados (ou obtidos do cache) para idioma: {self.language}")

    def _load_terms_from_dic(self, language: str) -> list:
        """Carrega termos médicos de um arquivo .dic, caso não estejam em cache."""
        clinical_terms = []
        file_path = f"{self.DICTIONARY_DIR}/clinical_terms_{language}.dic"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    term = line.strip().split('/')[0]
                    clinical_terms.append(term.lower())
            print(f"Carregados {len(clinical_terms)} termos médicos do arquivo {file_path}.")
        else:
            print(f"Arquivo {file_path} não encontrado. Nenhum termo médico carregado.")

        return clinical_terms

    def correct_word(self, word: str, similarity_threshold: int = 80) -> str:
        """
        Corrige uma palavra usando:
        1. Verificação no dicionário comum Hunspell.
        2. Se não é válida no dicionário comum, busca similaridade em clinical_terms.
        3. Caso não ache similaridade suficiente, retorna a primeira sugestão do Hunspell.
        """
        # Se a palavra já é válida no dicionário comum, não altera
        if self.hunspell.spell(word):
            return word
        
        # Tenta encontrar correspondência nos termos clínicos
        match = process.extractOne(word, self.clinical_terms, scorer=fuzz.ratio)
        if match and match[1] >= similarity_threshold:
            return match[0]

        # Caso contrário, pega a primeira sugestão do hunspell se existir
        suggestions = self.hunspell.suggest(word)
        if suggestions:
            return suggestions[0]
        return word  # fallback caso não haja sugestões

    def split_text_preserving_format(self, text: str) -> list:
        """
        Separa palavras, espaços e pontuação de um texto.
        Mantém a ordem original e preserva os espaços.
        """
        tokens = re.findall(r'(\s+|\w+|[^\w\s])', text, re.UNICODE)
        return tokens

    def correct_text(self, text: str) -> str:
        """Corrige um texto completo usando a estratégia definida."""
        # se não está ativo por alguma razão, devolve o mesmo texto.
        if not self.is_active:
            return text
        
        tokens = self.split_text_preserving_format(text)
        corrected_tokens = [
            self.correct_word(t) if re.match(r'\w+', t) else t
            for t in tokens
        ]
        return ''.join(corrected_tokens)
