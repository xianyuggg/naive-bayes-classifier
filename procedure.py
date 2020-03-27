import function
from config import TrainingFeature, FeatureConst


def getStopWords(training_feature: TrainingFeature):
    # pre-read stopwords
    stop_words = set()

    with open('./resource/stopwords.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            stop_words.add(line)
        if training_feature.FEATURE_ADD_STOP_WORDS:
            default_in_mails = ['oct', 'color', 'sep', 'sans', 'serif', 'aspx', 'valign', 'nowrap', 'rowspan', 'title',
                                'email', 'format', 'DEFANGED_SPAN', 'mon', 'sun', 'tue', 'apr', 'nil', 'fri', 'jan',
                                'mar',
                                'TABLE', 'thu', 'META', 'bgcolor', 'feb', 'EST', 'SPAN', 'RPCSS', 'arial', 'align',
                                'center', 'gif', 'div', 'float', 'CERT', 'mail', 'mailer', 'mime', 'SMTP', 'plain',
                                'reply',
                                'html', 'STRONG', 'FLOAT', 'font', 'info', 'subject', 'sender', 'DMDX', 'ascii', '3D2',
                                'href', 'body', 'ESMTP', 'media', 'strong', 'style', 'nbsp', 'left', 'img', 'height',
                                'src',
                                'cert', 'FONT', 'size', 'table', 'message', 'encoding', 'border', 'width', 'aug',
                                'span',
                                'transfer', 'net', 'subject', 'charset', 'received', 'content', 'version', 'text',
                                'DIV',
                                'class', 'color', 'HTML', 'http', 'type', 'MIME']
            for word in default_in_mails:
                stop_words.add(word)
        file.close()
    return stop_words


def getTrainTestSet(dirs, labels, training_feature: TrainingFeature):
    # shuffle the data
    assert (len(dirs) == len(labels))
    import random
    rand = random.randint(0, 100)
    random.seed(rand)
    random.shuffle(labels)
    random.seed(rand)
    random.shuffle(dirs)
    import math
    pivot = math.floor(len(dirs) / 5 * 4)
    import copy
    test_y = labels[pivot:]
    train_y = copy.deepcopy(labels[:pivot])
    test_dir = dirs[pivot:]
    train_dir = copy.deepcopy(dirs[:pivot])
    train_size = len(train_dir)
    test_size = len(test_dir)

    train_dir = train_dir[0:int(train_size * training_feature.train_proportion)]
    train_y = train_y[0:int(train_size * training_feature.train_proportion)]
    test_dir = test_dir[0:int(test_size * training_feature.test_proportion)]
    test_y = test_y[0:int(test_size * training_feature.test_proportion)]

    return train_dir, train_y, test_dir, test_y

def getVocabularyWithCount(train_dir, stop_words, training_feature:TrainingFeature):
    import chardet
    vocabulary = {}
    train_dir_is_ascii = []
    for dir in train_dir:
        with open(dir, 'rb') as file:
            data = file.read()
            if chardet.detect(data).get('encoding') != 'ascii':
                file.close()
                train_dir_is_ascii.append(0)
                continue
            else:
                train_dir_is_ascii.append(1)
            file.close()
        with open(dir, 'r') as file:
            data = file.read()
            words = function.textParse(data, training_feature)
            for word in words:
                if word not in stop_words:
                    if word not in vocabulary:
                        vocabulary.setdefault(word, 1)
                    else:
                        vocabulary[word] += 1
            file.close()
    return vocabulary, train_dir_is_ascii

def getVocabularyDict(vocabulary: dict, training_feature: TrainingFeature):
    """
    perform something like filtering
    :param vocabulary: a dictionary with all the word count in the training set
    :return: a dictionary with word->index in the feature array
    """
    vocab = {}
    index = 0
    if training_feature.FEATURE_DROP_FREQUENT_WORDS:
        print("Select vocabdict with drop_frequent")
        array = sorted([(k, v) for (k, v) in vocabulary.items()], key= lambda x: x[1])
        length = len(array)
        array = array[int(length * 0.85): int(length * 1.0)][0:training_feature.VOCAB_SIZE]
        for (k , _) in array:
            vocab.setdefault(k, index)
            index += 1
    else:
        print("Select vocabdict with non_drop_frequent")
        array = sorted([(k, v) for (k, v) in vocabulary.items()], key=lambda x: x[1])
        length = len(array)
        array = array[int(length * 0.85): int(length * 1.0)][-training_feature.VOCAB_SIZE:]
        for (k, _) in array:
            vocab.setdefault(k, index)
            index += 1
        # for (k, v) in vocabulary.items():
        #     if v > 50:
        #         vocab.setdefault(k, index)
        #         index += 1
    print("VocabDict length: ", len(vocab))
    # print(vocab)
    return vocab

def training(vocab: dict, train_dir: list, train_y: list, train_dir_is_ascii: list, const: FeatureConst, training_feature: TrainingFeature):
    assert (len(train_dir) == len(train_dir_is_ascii))
    hham_matrix = []
    hham_sum = [0 for _ in range(0, const.FEATURE_LENGTH)]
    spam_matrix = []
    spam_sum = [0 for _ in range(0, const.FEATURE_LENGTH)]

    for i in range(0, len(train_dir[0:])):
        if train_dir_is_ascii[i] == 1:
            # feature = [0 for _ in range(0, FEATURE_LENGTH)]
            with open(train_dir[i], 'r') as file:
                data = file.read()
                file.close()
                import function
                words = function.textParse(data, training_feature)
                feature = function.genFeatureArray(vocab, words, const)
            if train_y[i] == 1:
                spam_matrix.append(feature)
                spam_sum = [spam_sum[i] + feature[i] for i in range(0, const.FEATURE_LENGTH)]
                spam_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL] += 1
            else:
                hham_matrix.append(feature)
                hham_sum = [hham_sum[i] + feature[i] for i in range(0, const.FEATURE_LENGTH)]
                hham_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL] += 1

    LAPLACE = 1
    import math
    for i in range(0, const.VOCAB_LENGTH + 1):
        hham_sum[i] = math.log((hham_sum[i] + LAPLACE) / hham_sum[const.VOCAB_LENGTH + const.OFFSET_WORDS_TOTAL])
        spam_sum[i] = math.log((spam_sum[i] + LAPLACE) / spam_sum[const.VOCAB_LENGTH + const.OFFSET_WORDS_TOTAL])

    hham_sum[const.VOCAB_LENGTH + const.OFFSET_WORDS_TOTAL] /= hham_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL]
    spam_sum[const.VOCAB_LENGTH + const.OFFSET_WORDS_TOTAL] /= spam_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL]

    p_spam = spam_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL] / (
                hham_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL] + spam_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL])
    p_hham = hham_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL] / (
                hham_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL] + spam_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL])
    # print(spam_sum)
    # print(hham_sum)
    print("Average length, ham: ", hham_sum[const.VOCAB_LENGTH + const.OFFSET_WORDS_TOTAL], ",spam: ",
          spam_sum[const.VOCAB_LENGTH + const.OFFSET_WORDS_TOTAL], '\n')
    print("Total sum: ", (hham_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL] + spam_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL]))
    print("Spam sum: ", (spam_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL]))
    print("Ham sum: ", (hham_sum[const.VOCAB_LENGTH + const.OFFSET_MAILS_TOTAL]))
    print("P-spam: ", p_spam, '\n')

    return spam_sum, hham_sum, p_spam

def validating(vocab: dict, test_dir: list, test_y: list, spam_sum: list, hham_sum: list, p_spam: list, const: FeatureConst, training_feature: TrainingFeature):
    # test on data
    assert(len(test_dir) == len(test_y))
    test_dir_is_ascii = []
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(0, len(test_dir[0:])):
        with open(test_dir[i], 'rb') as file:
            data = file.read()
            import chardet
            if chardet.detect(data).get('encoding') != 'ascii':
                file.close()
                test_dir_is_ascii.append(0)
                continue
            else:
                test_dir_is_ascii.append(1)
            file.close()
        with open(test_dir[i], 'r') as file:
            data = file.read()
            file.close()
            words = function.textParse(data, training_feature)

            feature = function.genFeatureArray(vocab, words, const)
            (is_spam, is_ham) = function.calculateProb(feature, spam_sum, hham_sum, p_spam, const)
            tag = test_y[i]
            if tag == 1 and is_spam > is_ham:
                tp += 1
            if tag == 1 and is_ham > is_spam:
                fn += 1
            if tag == 0 and is_spam > is_ham:
                fp += 1  # false positive, judge is spam and real is ham
            if tag == 0 and is_ham > is_spam:
                tn += 1

    print("tp: %d, fp: %d, fn: %d, tn: %d" %(tp, fp, fn, tn))
    # print(is_spam, is_ham)

    return tp, fp, fn, tn

def measuring(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = (tp) / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    # print("Accuracy: ", accuracy, '\n')
    # print("Precision: ", precision, '\n')
    # print("Recall: ", recall, '\n')
    # print("F1-Measure: ", f1, '\n')

    return accuracy, precision, recall, f1