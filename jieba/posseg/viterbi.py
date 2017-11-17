import sys
import operator
MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")

if sys.version_info[0] > 2:
    xrange = range


def get_top_states(t_state_v, K=4):
    return sorted(t_state_v, key=t_state_v.__getitem__, reverse=True)[:K]


def viterbi(obs, states, start_p, trans_p, emit_p):
    '''
    mem_path = [
        {(state t, tag t): (state t-1, tag t-1), ......},
        {(观察值可能的状态(s), 词性tag): },
        ......
    ]
    每一个{}元素表示一个obs(观察值)的当前状态、标签对应的前一个观察值的状态、标签

    V[{(B,tag): p('B,tag')*p(obs[y]|'B,tag'), ...}]
        V[0]: 每个状态出现的概率*该状态发射obs第一个字的概率
    '''
    V = [{}]  # tabular
    mem_path = [{}]
    all_states = trans_p.keys()
    for y in states.get(obs[0], all_states):  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        mem_path[0][y] = ''
    for t in xrange(1, len(obs)):
        V.append({})
        mem_path.append({})
        #prev_states = get_top_states(V[t-1])
        # 所有可能的t-1时刻状态集合 {(state, tag)}
        prev_states = [
            x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]
        
        # 所有可能的t-1的下一个时刻状态集合 {(state, tag)}
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))
        # & 取交集
        obs_states = set(
            states.get(obs[t], all_states)) & prev_states_expect_next

        if not obs_states:
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states

        for y in obs_states:
            prob, state = max((V[t - 1][y0] + trans_p[y0].get(y, MIN_INF) +
                               emit_p[y].get(obs[t], MIN_FLOAT), y0) for y0 in prev_states)
            V[t][y] = prob
            mem_path[t][y] = state

    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    # if len(last)==0:
    #     print obs
    prob, state = max(last)

    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        state = mem_path[i][state]
        i -= 1
    return (prob, route)
