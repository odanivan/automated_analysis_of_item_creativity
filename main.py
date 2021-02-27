from enum import Enum
from itertools import tee

import csv
import requests
import sys
import cologne_phonetics
import pylcs
import termcolor

# Define interfaces
WORD_NOVELTY_API = "http://api.corpora.uni-leipzig.de/ws/words/deu_news_2012_1M/word/"


def CONTEXT_NOVELTY_API(q):
    return "https://api.scaleserp.com/search?api_key=827FF4DDC28347C1A13FA45DA7289CE9&q=%s&google_domain=google.de&location=Germany&gl=de&hl=de&page=1&output=json" % (
        q)


# Define scores
class Scores(Enum):
    WORD_NOVELTY = 1
    CONTEXT_NOVELTY = 2
    PARTICIPANT_SIMILARITY = 3
    SENTENCE_SIMILARITY = 4
    RHYTHMIC_SCORE = 5
    PHONETIC_SCORE = 6
    TOTAL_SCORE = 7


# Define helpers


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
    # Read survey results
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

    # Build survey vocabularies
    subject_vocabulary = {}
    variable_vocabulary = {}

    for subject, pairs in samples.items():
        if subject not in subject_vocabulary:
            subject_vocabulary[subject] = {}

        # Process subject
        for variable, sentence in pairs.items():

            # Process variable
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

    # Calculate relative occurrences
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

    # Determine survey scores
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
                    lower_score = min(20,
                                      lower_resp.json()["frequencyClass"] + 1)

                upper_word = lower_word.capitalize()
                upper_score = sys.maxsize
                upper_resp = requests.get(WORD_NOVELTY_API + upper_word)

                if upper_resp.status_code == 200:
                    upper_score = min(20,
                                      upper_resp.json()["frequencyClass"] + 1)

                word_score = min(lower_score, upper_score)

                if word_score != sys.maxsize:
                    score += word_score

            scores[subject][variable][Scores.WORD_NOVELTY] = int(
                round(score / 4))

            # Determine CONTEXT_NOVELTY
            score = 0
            for first, second in pairwise(stripped_sentence):
                try:
                    resp = requests.get(
                        CONTEXT_NOVELTY_API("\"%s + %s\"" % (first, second)))

                    if resp.status_code == 200:
                        response_json = resp.json()

                        if response_json["request_info"]["success"] == True:
                            if "total_results" in response_json[
                                    "search_information"]:
                                score += novelty_class(
                                    int(response_json["search_information"]
                                        ["total_results"]), 20, 512)
                            else:
                                score += 20
                except Exception:
                    continue

            for indices in [[0, 2], [0, 3], [1, 3]]:
                try:
                    resp = requests.get(
                        CONTEXT_NOVELTY_API(
                            "\"%s * %s\"" % (stripped_sentence[indices[0]],
                                             stripped_sentence[indices[1]])))

                    if resp.status_code == 200:
                        response_json = resp.json()

                        if response_json["request_info"]["success"] == True:
                            if "total_results" in response_json[
                                    "search_information"]:
                                score += novelty_class(
                                    int(response_json["search_information"]
                                        ["total_results"]), 20, 512)
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

            score= 1 - ((score/4) - min_subject_occurence) / \
                (max_subject_occurence - min_subject_occurence)
            scores[subject][variable][Scores.PARTICIPANT_SIMILARITY] = int(
                round(score * 20))

            # Determine SENTENCE_SIMILARITY
            score = 0
            for word in stripped_sentence:
                score += variable_vocabulary[variable][word]

            score = 1 - ((score / 4) - min_variable_occurence) / (
                max_variable_occurence - min_variable_occurence)
            scores[subject][variable][Scores.SENTENCE_SIMILARITY] = int(
                round(score * 20))

            # Determine RHYTHMIC_SCORE
            score = 0

            phonetic_result = cologne_phonetics.encode(
                sentence.strip(" ,;.:!?").lower())

            sounds_to_word_groups = {}

            for i in range(len(phonetic_result)):
                for j in range(i + 1, len(phonetic_result)):
                    (word1, sound1) = phonetic_result[i]
                    (word2, sound2) = phonetic_result[j]

                    for x in range(1, 1 + min(len(sound1), len(sound2))):

                        if ((sound1[-x:] == sound2[-x:])
                                and not (word1 == word2)):

                            adj_word1 = word1
                            adj_word2 = word2

                            if x > 1:
                                if (adj_word1[-1:] == 'd'):
                                    adj_word1 = adj_word1[:-1] + 't'
                                if (adj_word1[-1:] == 's'):
                                    adj_word1 = adj_word1[:-1] + 'z'

                                if (adj_word2[-1:] == 'd'):
                                    adj_word2 = adj_word2[:-1] + 't'
                                if (adj_word2[-1:] == 's'):
                                    adj_word2 = adj_word2[:-1] + 'z'

                            if not adj_word1[-(x + 1):] == adj_word2[-(
                                    x + 1):]:
                                continue

                            if sound1[-x:] not in sounds_to_word_groups:
                                sounds_to_word_groups[sound1[-x:]] = set()

                            sounds_to_word_groups[sound1[-x:]].add(word1)
                            sounds_to_word_groups[sound1[-x:]].add(word2)
                        else:
                            break

            word_groups_to_sounds = {}

            for sound, word_group in sounds_to_word_groups.items():
                sounds_to_word_groups[sound] = frozenset(word_group)

            for sound, word_group in sounds_to_word_groups.items():
                if (not word_group in word_groups_to_sounds
                    ) or len(sound) > len(word_groups_to_sounds[word_group]):
                    word_groups_to_sounds[word_group] = sound

            num_different_rhymes = len(word_groups_to_sounds)

            for word_group, sound in word_groups_to_sounds.items():
                rhyme_length = len(sound)
                score += min(20, (rhyme_length * 5) - 5)
                num_words_in_rhyme = len(word_group)
                score += min(20, (num_words_in_rhyme - 1) * 5)

            score = min(20, score)

            scores[subject][variable][Scores.RHYTHMIC_SCORE] = int(
                round(score))

            # Determine PHONETIC_SCORE
            score = 0

            total_combinations = 0
            levenstein_score = 0
            substring_score = 0

            for i in range(len(phonetic_result)):
                for j in range(i + 1, len(phonetic_result)):
                    total_combinations += 1

                    (word1, sound1) = phonetic_result[i]
                    (word2, sound2) = phonetic_result[j]

                    levenstein_distance = pylcs.levenshtein_distance(
                        sound1, sound2)
                    longest_substr_len = pylcs.lcs2(sound1, sound2)

                    levenstein_score += levenstein_distance / max(
                        len(sound1), len(sound2))

                    substring_score += longest_substr_len / min(
                        len(sound1), len(sound2))

            score += 0.5 * ((1 - (levenstein_score / total_combinations)) * 20)
            score += 0.5 * ((substring_score / total_combinations) * 20)

            scores[subject][variable][Scores.PHONETIC_SCORE] = int(
                round(score))

            # Determine TOTAL_SCORE
            scores[subject][variable][Scores.TOTAL_SCORE] = int(
                round(0.4 * scores[subject][variable][Scores.WORD_NOVELTY] +
                      0.4 * scores[subject][variable][Scores.CONTEXT_NOVELTY] +
                      0.1 * scores[subject][variable][
                          Scores.PARTICIPANT_SIMILARITY] + 0.1 *
                      scores[subject][variable][Scores.SENTENCE_SIMILARITY]))

            scores[subject][variable][Scores.TOTAL_SCORE] += int(
                round(0.1 * scores[subject][variable][Scores.RHYTHMIC_SCORE] +
                      0.1 * scores[subject][variable][Scores.PHONETIC_SCORE]))
            scores[subject][variable][Scores.TOTAL_SCORE] = min(
                20, scores[subject][variable][Scores.TOTAL_SCORE])

            print("\"%s\",\"%s\",%s,%s,%s,%s,%s,%s,%s \n" % (
                subject, sentence,
                termcolor.colored(
                    scores[subject][variable][Scores.WORD_NOVELTY], "blue"),
                termcolor.colored(
                    scores[subject][variable][Scores.CONTEXT_NOVELTY], "blue"),
                termcolor.colored(
                    scores[subject][variable][Scores.PARTICIPANT_SIMILARITY],
                    "blue"),
                termcolor.colored(
                    scores[subject][variable][Scores.SENTENCE_SIMILARITY],
                    "blue"),
                termcolor.colored(
                    scores[subject][variable][Scores.RHYTHMIC_SCORE], "cyan"),
                termcolor.colored(
                    scores[subject][variable][Scores.PHONETIC_SCORE], "cyan"),
                termcolor.colored(
                    scores[subject][variable][Scores.TOTAL_SCORE], "red")))
