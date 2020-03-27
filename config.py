class TrainingFeature:
    def __init__(self, add_stop_words: bool, drop_frequent_words: bool, vocab_size: int, use_existence: bool, categorize_words: bool, train_proportion: float, test_proportion: float ):
        self.FEATURE_ADD_STOP_WORDS = add_stop_words
        self.FEATURE_DROP_FREQUENT_WORDS = drop_frequent_words
        self.VOCAB_SIZE = vocab_size
        self.FEATURE_USE_EXISTENCE = use_existence
        self.CATEGORIZE_WORDS =  categorize_words
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion

class FeatureConst:
    def __init__(self, vocab):
        self.FEATURE_LENGTH =  len(vocab) + 10
        self.VOCAB_LENGTH = len(vocab)
        self.OFFSET_MAILS_TOTAL = 1
        self.OFFSET_NONE_VOCAB = 0
        self.OFFSET_WORDS_TOTAL = 2