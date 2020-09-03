from enum import Enum
from itertools import tee

import csv
import requests
import sys

WORD_NOVELTY_API = "http://api.corpora.uni-leipzig.de/ws/words/deu_news_2012_1M/word/"


def CONTEXT_NOVELTY_API(
    q): return "https://api.scaleserp.com/search?api_key=9E89BE384E38406C8FCCD63E5C792EA0&q=%s&google_domain=google.de&location=Germany&gl=de&hl=de&page=1&output=json" % (q)


class Scores(Enum):
    WORD_NOVELTY = 1
    CONTEXT_NOVELTY = 2
    PARTICIPANT_SIMILARITY = 3
    SENTENCE_SIMILARITY = 4
    TOTAL_SCORE = 5


def pairwise(iterable):
    f, s = tee(iterable)
    next(s, None)

    return zip(f, s)


def novelty_class(num_results: int, num_classes: int, upper_bound: int):
    while (num_results > upper_bound and num_classes > 1):
        upper_bound *= 2
        num_classes -= 1

    return num_classes


if __name__ == "__main__":
    file_name = "data"
    if len(sys.argv) > 1:
        file_name = str(sys.argv[1])
    lines = csv.reader(open(file_name + ".csv", "r"), delimiter=";")
    next(lines)

    samples = {}
    scores = {}

    for line in lines:
        if line[0] not in samples:
            samples[line[0]] = {}
            scores[line[0]] = {}

        samples[line[0]][line[1]] = line[2]
        scores[line[0]][line[1]] = {}

    subject_vocabulary = {}
    variable_vocabulary = {}

    for subject, pairs in samples.items():
        if subject not in subject_vocabulary:
            subject_vocabulary[subject] = {}

        for variable, sentence in pairs.items():
            if variable not in variable_vocabulary:
                variable_vocabulary[variable] = {}

            for word in sentence.strip(" ,;.:!?").lower().split():
                if word not in subject_vocabulary[subject]:
                    subject_vocabulary[subject][word] = 1
                else:
                    subject_vocabulary[subject][word] += 1

                if word not in variable_vocabulary[variable]:
                    variable_vocabulary[variable][word] = 1
                else:
                    variable_vocabulary[variable][word] += 1

    for subject, pairs in subject_vocabulary.items():
        for word, count in pairs.items():
            subject_vocabulary[subject][word] /= len(
                subject_vocabulary[subject])

    for variable, pairs in variable_vocabulary.items():
        for word, count in pairs.items():
            variable_vocabulary[variable][word] /= len(
                variable_vocabulary[variable])

    max_subject_occurence = 0.0
    min_subject_occurence = 1.0
    for subject in subject_vocabulary.values():

        if max(subject.values()) > max_subject_occurence:
            max_subject_occurence = max(subject.values())

        if min(subject.values()) < min_subject_occurence:
            min_subject_occurence = min(subject.values())

    max_variable_occurence = 0.0
    min_variable_occurence = 1.0
    for variable in variable_vocabulary.values():

        if max(variable.values()) > max_variable_occurence:
            max_variable_occurence = max(variable.values())

        if min(variable.values()) < min_variable_occurence:
            min_variable_occurence = min(variable.values())

    for subject, pairs in samples.items():
        for variable, sentence in pairs.items():
            stripped_sentence = sentence.strip(" ,;.:!?").lower().split()

            # Determine WORD_NOVELTY
            score = 0
            for word in stripped_sentence:
                lower_word = word.lower()
                lower_score = sys.maxsize
                lower_resp = requests.get(WORD_NOVELTY_API + lower_word)

                if lower_resp.status_code == 200:
                    lower_score = min(20, lower_resp.json()[
                                      "frequencyClass"] + 1)

                upper_word = lower_word.capitalize()
                upper_score = sys.maxsize
                upper_resp = requests.get(WORD_NOVELTY_API + upper_word)

                if upper_resp.status_code == 200:
                    upper_score = min(20, upper_resp.json()[
                                      "frequencyClass"] + 1)

                word_score = min(lower_score, upper_score)

                if word_score != sys.maxsize:
                    score += word_score

            scores[subject][variable][Scores.WORD_NOVELTY] = int(
                round(score / 4))

            # Determine CONTEXT_NOVELTY
            score = 0
            for first, second in pairwise(stripped_sentence):
                try:
                    resp = requests.get(CONTEXT_NOVELTY_API(
                        "\"%s + %s\"" % (first, second)))

                    if resp.status_code == 200:
                        response_json = resp.json()

                        if response_json["request_info"]["success"] == True:
                            if "total_results" in response_json["search_information"]:
                                score += novelty_class(
                                    int(response_json["search_information"]["total_results"]), 20, 512)
                            else:
                                score += 20
                except Exception:
                    continue

            for indices in [[0, 2], [0, 3], [1, 3]]:
                try:
                    resp = requests.get(CONTEXT_NOVELTY_API(
                        "\"%s * %s\"" % (stripped_sentence[indices[0]], stripped_sentence[indices[1]])))

                    if resp.status_code == 200:
                        response_json = resp.json()

                        if response_json["request_info"]["success"] == True:
                            if "total_results" in response_json["search_information"]:
                                score += novelty_class(
                                    int(response_json["search_information"]["total_results"]), 20, 512)
                            else:
                                score += 20
                except Exception:
                    continue

            scores[subject][variable][Scores.CONTEXT_NOVELTY] = int(
                round(score / 6))

            # Determine PARTICIPANT_SIMILARITY
            score = 0
            for word in stripped_sentence:
                score += subject_vocabulary[subject][word]

            score = 1 - ((score/4) - min_subject_occurence) / \
                (max_subject_occurence - min_subject_occurence)
            scores[subject][variable][Scores.PARTICIPANT_SIMILARITY] = int(
                round(score * 20))

            # Determine SENTENCE_SIMILARITY
            score = 0
            for word in stripped_sentence:
                score += variable_vocabulary[variable][word]

            score = 1 - ((score/4) - min_variable_occurence) / \
                (max_variable_occurence - min_variable_occurence)
            scores[subject][variable][Scores.SENTENCE_SIMILARITY] = int(
                round(score * 20))

            # Determine TOTAL_SCORE
            scores[subject][variable][Scores.TOTAL_SCORE] = int(round(0.4 * scores[subject][variable][Scores.WORD_NOVELTY] + 0.4 * scores[subject][variable]
                                                                      [Scores.CONTEXT_NOVELTY] + 0.1 * scores[subject][variable][Scores.PARTICIPANT_SIMILARITY] + 0.1 * scores[subject][variable][Scores.SENTENCE_SIMILARITY]))

            print("\"%s\",%s,%s,%s,%s,%s" % (
                sentence,
                scores[subject][variable][Scores.WORD_NOVELTY],
                scores[subject][variable][Scores.CONTEXT_NOVELTY],
                scores[subject][variable][Scores.PARTICIPANT_SIMILARITY],
                scores[subject][variable][Scores.SENTENCE_SIMILARITY],
                scores[subject][variable][Scores.TOTAL_SCORE]))
