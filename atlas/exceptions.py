
class ExceptionAsContinue(Exception):
    pass


class StepTermination(Exception):
    def __init__(self, choices, probs):
        self.choices = choices
        self.probs = probs


