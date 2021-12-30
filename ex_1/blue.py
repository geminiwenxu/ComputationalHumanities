from nltk import ngrams
import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import csv


def convert(lst):
    return ' '.join(lst).split()


def modified_uni_precision(candidate, reference):
    # print("total number of candidate sentence: ", len(candidate))
    total_candidate = len(candidate)
    count = 0
    for q in candidate:
        if q in reference:
            count += 1
    # print("total counts of each candidate word: ", count)

    ls_clip_count = []

    for i in candidate:
        print(i)
        max_count = reference.count(i)
        print(max_count)
        for j in range(max_count):
            reference.remove(i)
        print(reference)
        clip_count = min(max_count, count)
        ls_clip_count.append(clip_count)
    print(ls_clip_count)
    print(sum(ls_clip_count))
    print(total_candidate)


def modified_n_precision(candidate, reference, n):
    total_candidate = 0
    count = 0
    for grams in ngrams(candidate, n):
        # print(grams)
        total_candidate += 1

        if grams in ngrams(reference, n):
            count += 1
    # print(count)
    total_candidate = total_candidate
    # print(total_candidate)

    ls_clip_count = []
    for i in ngrams(candidate, n):  # candidate: each sentence in corpus
        max_count = 0
        for j in ngrams(reference, n):  # reference: corresponding to each candidate in corpus
            if i == j:
                # print(i, j)
                max_count = 1
                reference.remove(j[0])
                # print(reference)
        # print(max_count)
        clip_count = min(max_count, count)
        ls_clip_count.append(clip_count)
    # print(ls_clip_count)
    return total_candidate, sum(ls_clip_count)


def generate_df(can_path, ref_path):
    can_f = open(can_path)
    data = json.load(can_f)
    can_df = pd.json_normalize(data['sentences'])
    temp = can_df.drop(['id', 'lang'], axis=1)
    temp.columns = ['ilang', "candidate"]  # from translated corpus, take only ilang(L2) and candidate sentence
    # print(temp)

    ref_f = open(ref_path)
    data = json.load(ref_f)
    ref_df = pd.json_normalize(data['sentences'])
    ref_df.columns = ['id', 'lang',
                      'reference']  # from the original source corpus, take id, lang(L1) and reference sentence
    # print(ref_df)

    result = pd.concat([ref_df, temp], axis=1)
    # print(result)  # id, lang(L1), reference, ilang(L2), candidate
    return result


def precision(result):
    ls_count_clip = []
    ls_count = []
    lang = result.iloc[2]['lang']
    for i in range(4):  # calculate from 1-gram to 4-gram
        sum_count = 0
        sum_clip_count = 0
        for index, row in result.iterrows():  # iterate over each candidate and reference sentence
            if lang in ['de', 'en']:
                candidate = (row['candidate'].split())
                reference = (row['reference'].split())
                # print(candidate, reference)
            elif lang in ['zh', 'ja']:
                candidate = convert(list(row['candidate']))  # each sentence from candidate corpus
                reference = convert(list(row['reference']))  # each corresponding reference sentence
                # print(candidate, reference)
            count, clip_count = modified_n_precision(candidate, reference,
                                                     i + 1)  # return the total_candidate counts (by i+1-gram) and clip count of each sentence
            print(count, clip_count)
            sum_count += count  # accumulate the sum of count of each sentence -> corpus count
            sum_clip_count += clip_count  # accumulate the sum of count_clip of each sentence -> corpus count
        print(i + 1, "gram", sum_count, sum_clip_count)
        ls_count_clip.append(sum_clip_count)  # append sum_count and sum_clip_count for all 4 grams
        ls_count.append(sum_count)
    print("len should be 4 each", ls_count, ls_count_clip)
    p_n = []
    for i in range(4):
        p = ls_count_clip[i] / ls_count[i]
        p_n.append(p)
    return p_n


def brevity_penalty(result, p_n):  # p_n is one value in the dictionary: "ru": [0.37, 0.62, 0.6482, 0.639]
    w_n = 1 / len(p_n)  # since we choose N=4, len(p_n)=4
    sum_c = 0
    sum_r = 0
    lang = result.iloc[2]['lang']
    # print(lang)
    # print(type(lang))
    for index, row in result.iterrows():
        if lang in ['de', 'en']:
            candidate = (row['candidate'].split())
            reference = (row['reference'].split())
            # print(candidate, reference)
        elif lang in ['zh', 'ja']:
            candidate = convert(list(row['candidate']))
            reference = convert(list(row['reference']))
            # print(candidate, reference)
        r = len(reference)
        c = len(candidate)
        # print(r, c)
        sum_c += c
        sum_r += r
    print("sum c and r", sum_c, sum_r)
    if sum_c > sum_r:
        BP = 1
    else:
        BP = math.exp(1 - sum_r / sum_c)
    temp = 0
    for i in p_n:
        # print(i)
        # print(w_n)
        if i > 0:
            temp += w_n * math.log(i)
            # print("temp", temp)
        else:
            temp += w_n * math.log(0.00001)
    BLEU = BP * math.exp(temp)
    # print(BLEU)
    return BLEU


def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width / 2. + i / float(n) * width, vals[i],
                width=width / float(n), align="edge")
    plt.xticks(_X, X)


if __name__ == '__main__':
    # # test if function: modified_n_precision can calculate clip_count and count for each sentence correct
    # candidate_1_1 = convert(
    #     ['It is a guide to action which ensures that the military always obeys the commands of the party'.lower()])
    # candidate_1_2 = convert(
    #     ['It is to insure the troops forever hearing the activity guidebook that party direct'.lower()])
    # candidate_2 = convert(
    #     ['the_1 the_2 the the the the the'.lower()])
    # reference_1 = convert(
    #     ['It is a guide to action that ensures that the military will forever heed Party commands '.lower()])
    # reference_2 = convert(
    #     ['The_1 cat is on the_2 mat'.lower()])
    # candidate_3 = convert(['I always invariably perpetually do'])
    # reference_3 = convert(['I always do'])
    # print(candidate_3, reference_3)
    # total_candidate, count_clip = modified_n_precision(candidate_3, reference_3, 3)
    # print(total_candidate, count_clip)

    # ex 2 and 3
    zh_ls_lang = ['ar', 'en', 'ja', 'ko', 'de']  # each source language and its target languages list
    ja_ls_lang = ['zh', 'en', 'fr', 'ko', 'hi']
    de_ls_lang = ['en', 'it', 'fr', 'zh', 'nl']
    en_ls_lang = ['zh', 'de', 'hi', 'ru', 'ja']
    dict_p_n = {}
    dict_BLEU = {}
    for lang in zh_ls_lang:
        can_path = '/Users/wenxu/PycharmProjects/ComputationalHumanities/translated/zh_' + lang + '.json'
        print(can_path)
        ref_path = '/unzipped/dataset-zh.json'  # "zh-en"
        result = generate_df(can_path, ref_path)  # generate a result df: id, lang, reference, ilang, candidate
        p_n = precision(result)  # calculate the p_n: a list for 1-4 grams for one language pair
        print("per language pair p_n of 1-4 gram", p_n)
        BLEU = brevity_penalty(result, p_n)
        dict_p_n[lang] = p_n
        dict_BLEU[lang] = BLEU
    print(dict_p_n)
    print(dict_BLEU)
    with open('dict_BLEU.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_BLEU.items():
            writer.writerow([key, value])

    # plot for ex 2: N=4
    plt.style.use('ggplot')
    labels = ['1', '2', '3', '4']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    for key, value in dict_p_n.items():
        i = list(dict_p_n).index(key)
        rects = ax.bar(x - - width / 2. + i / float(5) * width, value, width=width / float(5), label=key)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('modified precision')
    ax.set_title('Distingushing among languages')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()
