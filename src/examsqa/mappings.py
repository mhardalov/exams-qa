subject_combine_map = {
    "Economics Basics (Business)": "Business & Economics",
    "Economics Basics (Theoretical)": "Business & Economics",
    "Economics & Marketing": "Business & Economics",
    "Economics": "Business & Economics",
    "Business": "Business & Economics",
    "Agriculture (Mechanical knowledge)": "Agriculture",
}

subj_groups = {
    "Natural Science": [
        "Geology",
        "Chemistry",
        "Physics",
        "Biology",
        "Natural Science",
        "Science",
    ],
    "Social Science": [
        "Politics",
        "Ethics",
        "Citizenship",
        "Geography",
        "History",
        "Philosophy",
        "Psychology",
        "Social",
        "Sociology",
        "Business & Economics",
    ],
    "Other": [
        "Professional",
        "Forestry",
        "Landscaping",
        "Agriculture",
        "Tourism",
        "Fine Arts",
        "Informatics",
        "Islamic Studies",
        "Religion",
    ],
}

lang_codes = {
    "Albanian": "sq",
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "French": "fr",
    "German": "de",
    "Hungarian": "hu",
    "Italian": "it",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Polish": "pl",
    "Portuguese": "pt",
    "Serbian": "sr",
    "Spanish": "es",
    "Turkish": "tr",
    "Vietnamese": "vi",
}

lang_families = {
    "Albanian": "Albanian",
    "Arabic": "Semitic",
    "German": "Germanic",
    "Turkish": " Turkic",
    "Vietnamese": "Austroasiatic",
    "Hungarian": "Uralic",
    "Bulgarian": "Balto-Slavic",
    "Croatian": "Balto-Slavic",
    "Lithuanian": " Balto-Slavic",
    "Macedonian": "Balto-Slavic",
    "Polish": "Balto-Slavic",
    "Serbian": "Balto-Slavic",
    "French": "Romance",
    "Portuguese": "Romance",
    "Italian": "Romance",
    "Spanish": "Romance",
}

index_mapping = {
    "albanian": "sqwiki",
    "arabic": "arwiki",
    "bulgarian": "bgwiki",
    "croatian": "hrwiki",
    "french": "frwiki",
    "german": "dewiki",
    "hungarian": "huwiki",
    "italian": "itwiki",
    "lithuanian": "ltwiki",
    "macedonian": "mkwiki",
    "polish": "plwiki",
    "portuguese": "ptwiki",
    "serbian": "srwiki",
    "spanish": "eswiki",
    "turkish": "trwiki",
    "vietnamese": "viwiki",
}

subj_groups_mapper = {}
for cat, subjects in subj_groups.items():
    subj_groups_mapper.update({subj: cat for subj in subjects})


def get_subject_name(subject: str) -> str:
    subject = subject_combine_map.get(subject, subject)
    return subject


def get_subject_group(subject: str) -> str:
    subject = get_subject_name(subject)
    return subj_groups_mapper[subject]


def get_lang_family(language: str) -> str:
    return lang_families.get(language, " UNK")


def get_lang_code(language: str) -> str:
    return lang_codes.get(language, language)


def get_grade_type(grade) -> str:
    grade = int(grade) if grade.isdigit() else 12
    return "Middle" if grade < 8 else "High"


def add_info_fields(df):
    df = df.copy()

    df["subject"] = df.subject.apply(get_subject_name)
    df["subject_group"] = df.subject.apply(get_subject_group)
    df["lang_family"] = df.language.apply(get_lang_family)
    df["lang_code"] = df.language.apply(get_lang_code)
    df["grade_group"] = df.grade.apply(get_grade_type)

    return df
