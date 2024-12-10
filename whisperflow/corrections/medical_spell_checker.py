from .correction_strategy import CorrectionStrategy
class MedicalSpellChecker:
    """
    Controlador principal que delega a execução de correção para a estratégia configurada.
    """

    def __init__(self, strategy: CorrectionStrategy):
        """
        Inicializa o corretor com a estratégia fornecida.
        
        :param strategy: Instância de uma classe que implementa CorrectionStrategy.
        """
        self.strategy = strategy

    def correct_text(self, text: str) -> str:
        """Corrige o texto usando a estratégia configurada."""
        return self.strategy.correct_text(text)
