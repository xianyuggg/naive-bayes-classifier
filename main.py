import function
import procedure
import config

if __name__ == "__main__":
    training_feature = config.TrainingFeature(add_stop_words=True,
                                                 categorize_words=True,
                                                 drop_frequent_words=False,
                                                 vocab_size = 5000,
                                                 use_existence=False,
                                                 train_proportion=0.2,
                                                 test_proportion=1
                                                 )
    labels = []
    dirs = []
    # pre-read the data dir and labels
    with open('./trec06p/label/index', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            if line[0] == 'ham':
                labels.append(0)
            else:
                labels.append(1)
            dirs.append('./trec06p/' + line[1][3:])
        file.close()

    train_dir = []
    train_y = []
    test_dir = []
    test_y = []

    acc = 0
    prec = 0
    recall = 0
    f1 = 0

    def processing(training_feature: config.TrainingFeature, if_shuffle = True, iter = 5):
        global train_dir, train_y, test_y, test_dir
        global acc, prec, recall, f1
        for i in range(0, iter):
            stop_words = procedure.getStopWords(training_feature)
            if if_shuffle:
                train_dir, train_y, test_dir, test_y = procedure.getTrainTestSet(dirs, labels, training_feature)
            vocabulary, train_dir_is_ascii = procedure.getVocabularyWithCount(train_dir, stop_words, training_feature)
            vocab = procedure.getVocabularyDict(vocabulary, training_feature)

            feature_const = function.FeatureConst(vocab)

            spam_sum, hham_sum, p_spam = procedure.training(vocab, train_dir, train_y, train_dir_is_ascii, feature_const, training_feature)

            tp, fp, fn, tn = procedure.validating(vocab, test_dir, test_y, spam_sum, hham_sum, p_spam, feature_const, training_feature)

            acct, prect, recallt, f1t = procedure.measuring(tp, fp, fn, tn)
            acc += acct
            prect += prect
            recall += recallt
            f1 += f1t
        print("Result: Acc: %f, Prec: %f, Recall: %f, f1: %f \n" % (acc / iter, prec / iter, recall / iter, f1 / iter))
        acc = 0
        prec = 0
        f1 = 0
        recall = 0




    print('Round 1 VocabSize 5000=================================\n')
    processing(training_feature, if_shuffle=True, iter=5)

    print('Round 2 VocabSize 7500=================================\n')
    training_feature.VOCAB_SIZE = 7500
    processing(training_feature, if_shuffle=True, iter=5)

    print('Round 2 VocabSize 10000=================================\n')
    training_feature.VOCAB_SIZE = 10000
    processing(training_feature, if_shuffle=True, iter=5)

    print('Round 2 VocabSize 20000=================================\n')
    training_feature.VOCAB_SIZE = 20000
    processing(training_feature, if_shuffle=True, iter=5)