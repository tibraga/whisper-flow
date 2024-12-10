from abc import ABC, abstractmethod

class CorrectionStrategy(ABC):
    """
    Interface para estratégias de correção.
    Todas as estratégias devem implementar `correct_text` e `load_resources`.
    """

    def __init__(self, language: str):
        self.language = language

    @abstractmethod
    def correct_text(self, text: str, is_medical_context: bool = True) -> str:
        """Corrige um texto usando uma abordagem específica."""
        pass
