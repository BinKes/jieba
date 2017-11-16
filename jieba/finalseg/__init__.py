from __future__ import absolute_import, unicode_literals
import re
import os
import sys
import pickle
from .._compat import *

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"


# PrevStatus[y]是当前时刻的状态所对应上一时刻可能的状态。提前建立一个当前时刻的状态到上一时刻的状态的映射表，记录每个状态在前一时刻的可能状态，降低计算量
PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}


def load_model():
    start_p = pickle.load(get_module_res("finalseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("finalseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("finalseg", PROB_EMIT_P))
    return start_p, trans_p, emit_p

if sys.platform.startswith("java"):
    start_P, trans_P, emit_P = load_model()
else:
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P


def viterbi(obs, states, start_p, trans_p, emit_p):
    '''
    返回生成obs的最大概率及其对应的最佳路径（隐藏状态序列）
    -----------------------------------------------
    1. V
        V[{'B': p('B')*p(obs[y]|'B'), ...}]
        V[0]: 每个状态出现的概率*该状态发射obs第一个字的概率
    2. (V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p) ==> 用s表示状态，则 s(t-1)-->s(t)-->obs(t)的概率。
    3. 寻找最佳路径，记录中间数据(V[t][y])：寻找前一个状态st-1到当前状态st并生成obs_t的最佳路径使得概率最大，记录此概率和路径V[t][y] = prob， newpath[y] = path[st-1] + [st]
    '''
    V = [{}]  # tabular
    path = {}
    for y in states:  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in xrange(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            V[t][y] = prob
            # 上一时刻最优的状态路径 + 这一时刻的状态
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES') # 得到最大概率和最后一个字的隐藏状态

    return (prob, path[state])


def __cut(sentence):
    global emit_P
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]

re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile("(\d+\.\d+|[a-zA-Z0-9]+)")


def cut(sentence):
    '''
    若是汉字串，则用 HMM 分词；
    否则返回其他字符
    '''
    sentence = strdecode(sentence)
    blocks = re_han.split(sentence) # 按汉字串、非汉字串拆分
    for blk in blocks:
        if re_han.match(blk):
            for word in __cut(blk):
                yield word
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
