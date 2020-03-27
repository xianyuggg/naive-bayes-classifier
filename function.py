from config import TrainingFeature, FeatureConst

def textParse(bigString, training_feature: TrainingFeature):
    '''
    Desc:
        接收一个大字符串并将其解析为字符串列表
    Args:
        bigString -- 大字符串
    Returns:
        去掉少于 2 个字符的字符串，并将所有字符串转换为小写，返回字符串列表
    '''
    import re
    # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    try:
        listOfTokens = re.split(r'\W*', bigString)
        res = []
        for tok in listOfTokens:
            if training_feature.CATEGORIZE_WORDS:
                if len(tok) > 20:
                    res.append('___OVERSIZE_WORD')
                    continue
                if len(tok) < 3:
                    continue
                if tok.isupper():
                    res.append('___PURE_UPPER') # regard it as a feature too
                    res.append(tok) # add all the upper case into vocab
                    continue
                if tok.isdigit():
                    res.append('___PURE_NUMBER')
                    continue
                if len(re.findall(r'.*\d+.*', tok)) > 0:
                    res.append('___CONTAIN_NUMBER')
                    continue
            res.append(tok.lower())
        return res
    except:
        return []

def genFeatureArray(vocab, words, const: FeatureConst):
    # get the training matrix
    feature = [0 for _ in range(0, const.FEATURE_LENGTH)]
    for word in words:
        val = vocab.get(word)
        if val != None:
            feature[val] += 1
        else:
            feature[const.VOCAB_LENGTH + const.OFFSET_NONE_VOCAB] += 1
        feature[const.VOCAB_LENGTH + const.OFFSET_WORDS_TOTAL] += 1
    return feature

# return (spam, hham)
def calculateProb(feature, spam, hham, p_spam, const: FeatureConst):

     p_predict_spam = p_spam
     p_predict_hham = 1 - p_spam
     import math
     p_predict_hham = math.log(p_predict_hham)
     p_predict_spam = math.log(p_predict_spam)
     for i in range(0, const.VOCAB_LENGTH):
         if feature[i] > 0:
             p_predict_spam += spam[i]
             p_predict_hham += hham[i]

     return (p_predict_spam, p_predict_hham)
