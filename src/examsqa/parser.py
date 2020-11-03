import abc
from typing import Any, Dict

from examsqa.tools import Question, TestInfo


class BasicParser:
    @abc.abstractmethod
    def match_questions(
        self, test_info: TestInfo, text: str, answers_dict: Dict, verbose: bool = False
    ) -> Question:
        """Match the questions from the raw text"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_basic_info(self, test_name: str) -> TestInfo:
        """Extract the test's basic info its name"""
        raise NotImplementedError

    @abc.abstractmethod
    def parse_answers(self, test_name: str) -> Dict[Any, str]:
        """Extract dict with question number and correct answer"""
        raise NotImplementedError
