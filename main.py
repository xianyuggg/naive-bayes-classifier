import function
import procedure
import config

if __name__ == "__main__":
    # set training feature
    training_feature = config.TrainingFeature(add_stop_words=True,
                                                 categorize_words=True,
                                                 drop_frequent_words=False,
                                                 vocab_size = 5000,
                                                 use_existence=False,
                                                 train_proportion=0.2,
                                                 test_proportion=1,
                                                increase_precision=False,
                                                moderate_laplace=False
                                                 )

    # pre-read the data dir and labels
    labels = []
    dirs = []
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


    def processing(dirs, labels, training_feature: config.TrainingFeature, if_shuffle = True, iter = 5):
        global train_dir, train_y, test_y, test_dir
        global acc, prec, recall, f1
        for i in range(0, iter):
            stop_words = procedure.getStopWords(training_feature)
            if if_shuffle:
                train_dir, train_y, test_dir, test_y = procedure.getTrainTestSet(dirs, labels, training_feature)
            vocabulary, train_dir_is_ascii = procedure.getVocabularyWithCount(train_dir, stop_words, training_feature)

            # get a vocab -> index dict
            vocab = procedure.getVocabularyDict(vocabulary, training_feature)

            # get basic const for training (etc. vocab size)
            feature_const = function.FeatureConst(vocab)

            spam_sum, hham_sum, p_spam = procedure.training(vocab, train_dir, train_y, train_dir_is_ascii, feature_const, training_feature)

            tp, fp, fn, tn = procedure.validating(vocab, test_dir, test_y, spam_sum, hham_sum, p_spam, feature_const, training_feature)

            acct, prect, recallt, f1t = procedure.measuring(tp, fp, fn, tn) # tmp value
            acc += acct
            prec += prect
            recall += recallt
            f1 += f1t
            print('----------------------------------------------\n')
        print("Result: Acc: %f, Prec: %f, Recall: %f, f1: %f \n" % (acc / iter, prec / iter, recall / iter, f1 / iter))
        acc = 0
        prec = 0
        f1 = 0
        recall = 0



    # MainTrainingProcedure


    print('=================================Round 1 VocabSize 1000=================================\n')
    training_feature.VOCAB_SIZE = 1000
    training_feature.FEATURE_ADD_STOP_WORDS = False
    training_feature.CATEGORIZE_WORDS = False
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 2 VocabSize 1000=================================\n')
    training_feature.VOCAB_SIZE = 1000
    training_feature.FEATURE_ADD_STOP_WORDS = True
    training_feature.CATEGORIZE_WORDS = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    quit()

    print('+++++++++++++++++++++++++++++++++VocabSize Test:+++++++++++++++++++++++++++++++++\n')
    print('=================================Round 1 VocabSize 500=================================\n')
    training_feature.VOCAB_SIZE = 500
    training_feature.FEATURE_ADD_STOP_WORDS = True
    training_feature.CATEGORIZE_WORDS = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 2 VocabSize 1000=================================\n')
    training_feature.VOCAB_SIZE = 1000
    training_feature.FEATURE_ADD_STOP_WORDS = True
    training_feature.CATEGORIZE_WORDS = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 3 VocabSize 2000=================================\n')
    training_feature.VOCAB_SIZE = 2000
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 4 VocabSize 5000=================================\n')
    training_feature.VOCAB_SIZE = 5000
    training_feature.FEATURE_ADD_STOP_WORDS = True
    training_feature.CATEGORIZE_WORDS = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 5 VocabSize 7500=================================\n')
    training_feature.VOCAB_SIZE = 7500
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 6 VocabSize 15000， add stop words, dont categorize=================================\n')
    training_feature.VOCAB_SIZE = 15000
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 7 VocabSize 20000， without stop words, dont categorize=================================\n')
    training_feature.VOCAB_SIZE = 20000
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)



    print("==================================\n\n")

    print("+++++++++++++++++++++++++++++++++Feature Test+++++++++++++++++++++++++++++++++\n")

    print('=================================Round 1 VocabSize 10000， dont add stop words, dont categorize=================================\n')
    training_feature.VOCAB_SIZE = 10000
    training_feature.FEATURE_ADD_STOP_WORDS = False
    training_feature.CATEGORIZE_WORDS = False
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 2 VocabSize 10000, add stop words, dont categorize=================================\n')
    training_feature.FEATURE_ADD_STOP_WORDS = True
    training_feature.CATEGORIZE_WORDS = False
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 3 VocabSize 10000，dont add stop words, categorize=================================\n')
    training_feature.VOCAB_SIZE = 10000
    training_feature.FEATURE_ADD_STOP_WORDS = False
    training_feature.CATEGORIZE_WORDS = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 4 VocabSize 10000，stop words, categorize=================================\n')
    training_feature.VOCAB_SIZE = 10000
    training_feature.FEATURE_ADD_STOP_WORDS = True
    training_feature.CATEGORIZE_WORDS = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round 5 VocabSize 10000，stop words, categorize drop frequent word=================================\n')
    training_feature.VOCAB_SIZE = 10000
    training_feature.FEATURE_ADD_STOP_WORDS = True
    training_feature.CATEGORIZE_WORDS = True
    training_feature.FEATURE_DROP_FREQUENT_WORDS = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)
    training_feature.FEATURE_DROP_FREQUENT_WORDS = False

    print("==================================\n\n")
    print("+++++++++++++++++++++++++++++++++Precision and Moderate laplace test+++++++++++++++++++++++++++++++++\n")
    print('=================================Round 1 VocabSize 10000, increase_precision=================================')
    training_feature.INCREASE_PRECISION = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)
    training_feature.INCREASE_PRECISION = False

    print('=================================Round2 VocabSize 10000, moderate smoothing=================================')
    training_feature.MODERATE_LAPLACE = True
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)
    training_feature.MODERATE_LAPLACE = False

    print("==================================\n\n")
    print("+++++++++++++++++++++++++++++++++Training Size Test+++++++++++++++++++++++++++++++++")
    print('=================================Round1 VocabSize 10000, 0.05=================================')
    training_feature = config.TrainingFeature(add_stop_words=True,
                                              categorize_words=True,
                                              drop_frequent_words=False,
                                              vocab_size=10000,
                                              use_existence=False,
                                              train_proportion=0.05,
                                              test_proportion=1,
                                              increase_precision=False,
                                              moderate_laplace=False
                                              )
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round2 VocabSize 10000, 0.2=================================')
    training_feature.train_proportion = 0.2
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round3 VocabSize 10000, 0.5=================================')
    training_feature.train_proportion = 0.5
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)

    print('=================================Round4 VocabSize 10000, 1.0=================================')
    training_feature.train_proportion = 1.0
    processing(dirs, labels, training_feature, if_shuffle=True, iter=5)