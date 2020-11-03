import json
import math
import os
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path

import docx
import regex as re
import textract
from bs4 import BeautifulSoup, Tag


class Question:
    def __init__(self, question, qid, answers, correct, source, subject, grade, test_info):
        self.question = question
        self.qid = qid
        self.answers = answers
        self.correct = correct
        self.source = source
        self.idx = str(uuid.uuid1())
        self.subject = subject
        self.grade = grade
        self.test_info = test_info

    def _clean_text(self, text):
        if text is None:
            return None

        return re.sub(r"\s+", r" ", text)

    def show(self):
        print("Question #", self.qid, self.source, self.subject, self.grade)
        print("Q:", self.question)
        print("A:", self.answers)
        print("Correct:", self.correct)

    def to_dict(self):
        return {
            "id": self.idx,
            "qid": self.qid,
            "question": self._clean_text(self.question),
            "answers": [self._clean_text(v) for k, v in self.answers.items()],
            "correct": self._clean_text(self.correct),
            "source": self.source,
            "subject": self.subject,
            "grade": self.grade,
            "metadata": self.test_info.to_dict(),
        }

    def __repr__(self):
        return (
            f"{self.qid} - {self.question}, {self.answers}, {self.correct}, "
            f"{self.source}, {self.subject} ({self.grade} grade)"
        )


class TestInfo:
    def __init__(self, month, year, comment, file_name):
        self.month = month
        self.year = year
        self.comment = comment
        self.file_name = file_name

    def __hash__(self):
        return hash(self.month) ^ hash(self.year) ^ hash(self.comment)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.month == other.month
            and self.year == other.year
            and self.comment == other.comment
        )

    def __repr__(self):
        return f"{self.comment} - {self.month}, {self.year}, {self.file_name}"

    def to_dict(self):
        return {"year": self.year, "month": self.month}


def read_text(root_dir, test_name, encoding="utf-8"):
    with open(os.path.join(root_dir, f"{test_name}"), "r", encoding=encoding) as o:
        txt = "\n".join(o.readlines())
        return txt


def get_text_from_doc(filename, encoding="utf-8"):
    try:
        doc = docx.Document(filename)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except:
        return textract.process(filename).decode(encoding)


def get_text_with_br(tag, result=""):
    for x in tag.contents:
        if isinstance(x, Tag):  # check if content is a tag
            if x.name == "br":  # if tag is <br> append it as string
                result += "\n"
            else:  # for any other tag, recurse
                result = get_text_with_br(x, result)
        else:  # if content is NavigableString (string), append
            result += x

    return result


def parse_paragraph(p, with_br=False):
    top, left = re.search(r"top:(\-?\d+)px;left:(\-?\d+)px", p["style"]).groups()

    return int(top), int(left), p.text if not with_br else get_text_with_br(p)


def get_text_from_html(
    filename,
    join_char=" ",
    margin=5,
    start_page=1,
    end_page=math.inf,
    min_left=0,
    max_left=math.inf,
    min_top=0,
    max_top=math.inf,
    with_br=False,
):
    with open(filename, "r") as o:
        soup = BeautifulSoup("".join(o.readlines()), features="lxml")

    rows = defaultdict(list)
    i = start_page
    while i <= end_page:
        ps = soup.select(f"#page{i}-div > p")
        if not ps:
            if not soup.select(f"#page{i+1}-div > p"):
                break
            else:
                i += 1
                continue
        ps = map(lambda x: parse_paragraph(x, with_br), ps)

        prev_top = 0
        for top, left, text in sorted(ps, key=lambda x: x[0]):
            if top >= max_top or min_top > top:
                continue

            top = int(1e6) * i + top
            if top - prev_top > margin:
                prev_top = top
            rows[prev_top].append((left, text))
        i += 1

    rows = OrderedDict(rows)
    ltr_sorting = map(lambda x: sorted(x, key=lambda r: r[0]), rows.values())
    ltr_filter = [y for x in ltr_sorting for y in x if y[0] >= min_left and y[0] <= max_left]
    text = join_char.join([x[1] for x in ltr_filter]).replace("\xa0", " ").replace("\uf0b7", "")

    return text


def read_answers_file(test_name, root_dir="ocrs"):
    path = Path(test_name)
    txt = read_text(Path(root_dir) / path.stem, "answers.txt")

    return txt


def export_questions(data, path):
    with open(path, "w") as o:
        json.dump(data, o, ensure_ascii=False, indent=4)
