from config import config
from config_tac import config_tac
import json
from tqdm import tqdm
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from utils import *
from torch.utils.data import Dataset, DataLoader
import torch
import random
import pickle

seed = 999
random.seed(seed)
np.random.seed(seed)

tokenizer = BertTokenizer.from_pretrained(config.pre_train_path, do_lower_case=True, cache_dir=None)

def preprocess_data(task, subtask):
    task_data = []
    # for task in ['train', 'dev', 'test']:
    with open(config.previous_path + '/data/' + task + '.json', 'rb') as f:
        task_data.extend(json.load(f))

    # if task == 'train':
    #     with open(config.previous_path + '/data/' + 'dev' + '.json', 'rb') as f:
    #         task_data.extend(json.load(f))
    # if task == 'test':
    #     with open(config.previous_path + '/data/dev.json', 'rb') as f:
    #         task_data = json.load(f)

    if subtask == 'other':
        sent_id_list, sent_tag = [[] for _ in range(2)]
        for line in task_data:
            sent = line['sentence'].replace('\\', '').replace('``', '"').replace("''", '"').lower()
            if len(sent) != 0:
                sent_token = tokenizer.tokenize(sent)
                sent_token.append('[SEP]')
                sent_token.insert(0, '[CLS]')
                sent_id = tokenizer.convert_tokens_to_ids(sent_token)
                sent_id_list.append(sent_id)
                if line['golden-event-mentions']:
                    sent_tag.append(1)
                else:
                    sent_tag.append(0)
        return sent_id_list, sent_tag

    # control data length
    task_data = [line for line in task_data if line['golden-event-mentions']]
    for line in task_data:
        t_l = [l['trigger']['text'] for l in line['golden-event-mentions']]
    #     if len(set([len(l.split(' ')) for l in t_l])) > 1:
    #         print(line['sentence'])
    #         print([l['trigger']['text'] for l in line['golden-event-mentions']])
    #         print()
    # assert 1==-1

    # if task == 'train':
    #     # with open(config.previous_path + '/data/' + task + '_span.pkl', 'rb') as f:
    #     with open(config.previous_path + '/data/' + task + '_aug_de.pkl', 'rb') as f:
    #         task_data.extend(pickle.load(f))

    sent_id_add, trigger_idx_add = [[] for _ in range(2)]
    task_data_new = []
    # if task == 'train':
    if task != 'test':
        '''construct sequential sentences'''
        # with open(config.previous_path + '/data/' + task + '.json', 'rb') as f:
        #     task_data = json.load(f)
        # with open(config.previous_path + '/data/' + task + '_tag.json', 'rb') as f:
        #     task_tag = json.load(f)
        #
        # punct = ['?', '!', '.', '"', "'", ':']
        # for i in range(len(task_data)):
        #     if task_data[i]['golden-event-mentions']:
        #         events = task_data[i]['golden-event-mentions']
        #
        #         try:
        #             flag = len(task_data[i + 1]['golden-event-mentions']) == 0
        #         except:
        #             flag = False
        #
        #         p = task_data[i - 1]['sentence'][-1]
        #         if len(p) == 0:
        #             p = task_data[i - 1]['sentence'][-2]
        #
        #         if len(task_data[i - 1]['golden-event-mentions']) == 0 \
        #                 and ('(' not in task_data[i - 1]['sentence'] and ')' not in task_data[i - 1]['sentence']) \
        #                 and task_tag[i - 1] == task_tag[i - 2] and p in punct:
        #             task_data_new.append({'sentence': task_data[i - 1]['sentence'] + ' <---- ' +
        #                                     task_data[i]['sentence'], 'golden-event-mentions': events})
        #
        #         elif flag and ('(' not in task_data[i + 1]['sentence'] and ')' not in task_data[i + 1]['sentence']) \
        #                 and task_tag[i] == task_tag[i + 1] and p in punct:
        #             task_data_new.append({'sentence': task_data[i]['sentence'] + ' ----> ' +
        #                                     task_data[i + 1]['sentence'], 'golden-event-mentions': events})
        #
        #         else:
        #             task_data_new.append(task_data[i])

        # with open(config.previous_path + '/data/' + task + '_translate_de.pkl', 'rb') as f:
        #     train_translate = pickle.load(f)
        # with open(config.previous_path + '/data/train_translate_zh.pkl', 'rb') as f:
        #     train_translate_zh = pickle.load(f)

        num_1, num_2, num_3 = [0  for _  in range(3)]
        task_data_new_2 = []
        # task_data = task_data_new
        for i, line in tqdm(enumerate(task_data)):
            sent = line['sentence'].replace('\\', '').replace('``', '"').replace("''", '"').lower()
            task_data_new_2.append({'sentence': sent, 'golden-event-mentions': line['golden-event-mentions']})
            trigger_list = [event['trigger']['text'].replace('\n', ' ').lower() for event in line['golden-event-mentions']]

            # assert line['sentence'] == train_translate[i][0]
            # sent_aug = train_translate[i][2].replace('``', '').lower()
            #
            # flag = judge_trigger(sent_aug, trigger_list)
            #
            # # zh as supplement
            # if flag:
            #     num_1 += 1
            #
            #     # sent_aug_zh = train_translate_zh[i][2].replace('``', '').lower()
            #     # flag = judge_trigger(sent_aug_zh, trigger_list)
            #
            #     # if not flag:
            #     #     sent_aug = sent_aug_zh
            #
            # # synonym
            # if flag:
            #     num_2 += 1
            #
            #     # sent_aug_token = [l.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace('"', '') \
            #     #                 .replace('(', '').replace(')', '').replace(':', '').replace('`', '').replace("'", '') \
            #     #                   for l in sent_aug.split(' ')]
            #     # sent_aug_token_new = [l for l in sent_aug.split(' ')]
            #     # n = 0
            #     # for tri in trigger_list:
            #     #     if tri not in sent_aug_token:
            #     #         for j, l in enumerate(sent_aug_token):
            #     #             l_l = get_synonyms(l)
            #     #             if tri in l_l:
            #     #                 sent_aug_token_new[j] = tri
            #     #                 n += 1
            #     #                 break
            #     #     else:
            #     #         n += 1
            #     # if len(trigger_list) == n:
            #     #     # sent_aug = ' '.join(sent_aug_token_new)
            #     #     flag = False
            #
            # if flag:
            #     num_3 += 1
            #     # print(sent)
            #     # print(sent_aug)
            #     # print(trigger_list)
            #     # print()
            #     # assert 1 == -1

            # # continue
            # if train_translate[i][2] == train_translate[i][0] or flag:
            #     sent_aug = \
            #         eda(sent, alpha_sr=0.1, alpha_ri=0, alpha_rs=0, p_rd=0, num_aug=1, trigger_list=trigger_list)[0]
            #     while sent_aug == sent:
            #         sent_aug = \
            #             eda(sent, alpha_sr=0, alpha_ri=0, alpha_rs=0, p_rd=0.1, num_aug=1,
            #                 trigger_list=trigger_list)[0]

            # task_data_new_2.append({'sentence': sent_aug, 'golden-event-mentions': line['golden-event-mentions']})

            # data augment
            # sent_aug = eda(sent, alpha_sr=0, alpha_ri=0, alpha_rs=0, p_rd=0.1, num_aug=2, trigger_list=trigger_list)
            # for sent_a in sent_aug:
            #     task_data_new.append({'sentence': sent_a, 'golden-event-mentions': line['golden-event-mentions']})

            '''positive sample construction'''
            # sent_l = [sent] + [sent_aug]
            sent_l = [sent]
            for s in sent_l:
                idx_l, sent_a_id = [[] for _ in range(2)]
                if s not in ['nomination.']:
                    # sent_aug = train_translate[i][2].replace('``', '').lower()
                    # flag = judge_trigger(sent_aug, trigger_list)
                    # if train_translate[i][2] == train_translate[i][0] or flag:
                    sent_aug = eda(s, alpha_sr=0.1, alpha_ri=0, alpha_rs=0, p_rd=0, num_aug=1, trigger_list=trigger_list)
                    sent_aug = [sent_a.replace('\\', '') for sent_a in sent_aug][0]
                    sent_a_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_aug))

                    trigger_s, trigger_e = [np.zeros((len(sent_a_id), config.event_type_num), int) for _ in range(2)]
                    for event in line['golden-event-mentions']:
                        e_t = config.event_type_t2i[event['event_type']]
                        trigger_text = event['trigger']['text'].replace('\n', ' ')
                        trigger_token = tokenizer.tokenize(trigger_text)
                        trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)

                        # Todo insert 
                        for k in range(len(sent_a_id)):
                            if sent_a_id[k: k + len(trigger_id)] == trigger_id:
                                idx_l.append([k, k + len(trigger_id) - 1])
                                break

                        # for k in range(len(sent_a_id)):
                        #     if sent_a_id[k: k + len(trigger_id)] == trigger_id:
                        #         if 1 in trigger_s[k, :] and 1 in trigger_e[k + len(trigger_id) - 1, :]:
                        #             continue
                        #         trigger_s[k][e_t] = 1
                        #         trigger_e[k + len(trigger_id) - 1][e_t] = 1
                        #
                        #         idx_l.append([k, k + len(trigger_id) - 1])
                        #         # Todo how to preprocess the same trigger
                        #         break

                for name in config.event_upper_type_t2id:
                    sent_a_id = sent_a_id + config.event_upper_type_t2id[name] + [102]
                sent_id_add.append(sent_a_id)
                trigger_idx_add.append(idx_l)

        # print('trigger cannot be found: %.4f (de), %.4f (+zh), %.4f(self)'
        #       % (num_1 / len(task_data), num_2 / len(task_data), num_3 / len(task_data)))
        task_data_new = task_data_new_2
    else:
        # task_data_new = [l for l in task_data if len(l['golden-event-mentions']) == 1]
        task_data_new = task_data

    # assert 1==-1

    sent_id_list, trigger_start, trigger_end, argument_start, argument_end, sent_com_id_list, sent_tag_list, trigger_span, \
    trigger_num, trigger_sent_start, trigger_sent_end, sent_id_add_list, trigger_idx_add_list, event_type_list = \
        [[] for _ in range(14)]
    num, num_2 = [0 for _ in range(2)]
    # for k, line in tqdm(enumerate(task_data)):
    for k, line in tqdm(enumerate(task_data_new)):
        sent = line['sentence']

        '''construct sequential sentences'''
        # Todo replace with [SEP]
        if ' <---- ' in sent:
            sent_token = sent.split(' ')
            idx = [l for l in range(len(sent_token)) if sent_token[l] == '<----'][0]
            sent_t = tokenizer.tokenize(' '.join(sent_token[: idx]))
            idx_sel = (-1, len(sent_t))
        elif ' ----> ' in sent:
            sent_token = sent.split(' ')
            idx = [l for l in range(len(sent_token)) if sent_token[l] == '---->'][0]
            sent_t = tokenizer.tokenize(' '.join(sent_token[: idx]))
            idx_sel = (len(sent_t), -1)
        else:
            idx_sel = []
        sent = sent.replace(' <---- ', ' ').replace(' ----> ', ' ')

        sent_token = tokenizer.tokenize(sent)
        sent_token.insert(0, '[CLS]')
        sent_token.append('[SEP]')
        sent_id = tokenizer.convert_tokens_to_ids(sent_token)

        if len(idx_sel) == 0:
            idx_range = [0, len(sent_id)]
        elif idx_sel[0] == -1:
            idx_range = [idx_sel[1], len(sent_id)]
        elif idx_sel[1] == -1:
            idx_range = [0, idx_sel[0]]

        trigger_s, trigger_e  = [np.zeros((len(sent_id), config.event_type_num), int) for _ in range(2)]
        trigger_sp, trigger_s_s, trigger_s_e = [[0] * len(sent_id) for _ in range(3)]
        trigger_sp[0] = 1
        trigger_sp[-1] = 1
        for event in line['golden-event-mentions']:
            e_t = config.event_type_t2i[event['event_type']]
            trigger_text = event['trigger']['text'].replace('\n', ' ')
            trigger_token = tokenizer.tokenize(trigger_text)
            trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)

            for i in range(idx_range[0], idx_range[1]):
                if sent_id[i: i + len(trigger_id)] == trigger_id:
                    if 1 in trigger_s[i, :] and 1 in trigger_e[i + len(trigger_id) - 1, :]:
                        continue
                    trigger_s[i][e_t] = 1
                    trigger_e[i + len(trigger_id) - 1][e_t] = 1
                    trigger_sp[i: i + len(trigger_id)] = [1] * len(trigger_id)

                    trigger_s_s[i] = 1
                    trigger_s_e[i + len(trigger_id) - 1] = 1
                    # Todo how to preprocess the same trigger
                    break

        # if 1 in trigger_s:
        # if task == 'train' and 1 in trigger_s and len(trigger_idx_add[k]) == len(np.where(trigger_s == 1)[0]):
        #     for name in config.event_upper_type_t2id:
        if task != 'test' and 1 in trigger_s and len(trigger_idx_add[k]) == len(np.where(trigger_s == 1)[0]):
            for name in config.event_upper_type_t2id:
                sent_id = sent_id + config.event_upper_type_t2id[name] + [102]
            sent_id_list.append(sent_id)
            trigger_start.append(list(trigger_s))
            trigger_end.append(list(trigger_e))
            sent_tag_list.append([1] * len(sent_id))
            trigger_span.append(trigger_sp)

            sent_id_add_list.append(sent_id_add[k])
            trigger_idx_add_list.append(trigger_idx_add[k])

            event_type_list.append(list(set([config.event_type_t2i[event['event_type']] + 1
                                             for event in line['golden-event-mentions']]))) # 0 represents padding

        # elif task != 'train' and 1 in trigger_s:
        elif task == 'test' and 1 in trigger_s:
            for name in config.event_upper_type_t2id:
                sent_id = sent_id + config.event_upper_type_t2id[name] + [102]
            sent_id_list.append(sent_id)
            trigger_start.append(list(trigger_s))
            trigger_end.append(list(trigger_e))
            sent_tag_list.append([1] * len(sent_id))
            trigger_span.append(trigger_sp)

        else:
            num += 1

    # print(max([len(l) for l in sent_id_list]))
    # assert 1==-1

    if subtask == 'trigger':
        print('trigger words cannot be labeled in the sentence: %.4f' % (num / len(task_data)))

        # if task == 'train':
        if task != 'test':
            sent_id_list_new, trigger_start_new, trigger_end_new, trigger_start_com, trigger_start_tag, trigger_end_com, \
            trigger_end_tag, trigger_start_new_1, trigger_end_new_1, trigger_start_com_1, trigger_end_com_1, \
            event_type_list_new = [[] for _ in range(12)]
            for i in tqdm(range(len(trigger_start))):
                sent_id = sent_id_list[i]
                # todo int-->float 前者只要有negative就0,后者变成float
                tri_s_1, tri_e_1, tri_s_2, tri_e_2 = [np.zeros((len(sent_id), config.event_type_num), int) for _ in range(4)]
                # tri_s_1, tri_e_1, tri_s_2, tri_e_2 = [np.zeros((len(sent_id), config.event_type_num), float) for _ in range(4)]
                tri_s_s_1, tri_s_e_1 = [[0] * len(sent_id) for _ in range(2)]
                sent_s_c, sent_e_c, sent_s_t_1, sent_e_t_1, sent_s_c_1, sent_e_c_1, sent_s_t_2, sent_e_t_2, sent_s_c_2, \
                sent_e_c_2 = [[] for _ in range(10)]
                s_idx_l = np.where(np.array(trigger_start[i]) == 1)
                e_idx_l = np.where(np.array(trigger_end[i]) == 1)
                for j in range(len(s_idx_l[0])):
                    '''start'''
                    p = random.random()

                    s_idx = s_idx_l[0][j]
                    s_t = s_idx_l[1][j]
                    sent_s_c.append([1, 0])

                    '''select the neighbor'''
                    sent_s_c_1.append([p, 1 - p])
                    if s_idx == 1:
                        sent_s_t_1.append([s_idx, s_idx + 1])
                    else:
                        sent_s_t_1.append([s_idx, s_idx - 1])
                    tri_s_1[s_idx][s_t] = p

                    # 另一个随机选
                    # sent_s_c_1.append([p, 1 - p])
                    # sent_s_t_1.append([s_idx, s_idx + 1])
                    # tri_s_1[s_idx][s_t] = p
                    #
                    # p_1 = random.random()
                    # sent_s_c_2.append([p_1, 1 - p_1])
                    # if s_idx != 1:
                    #     sent_s_t_2.append([s_idx, s_idx - 1])
                    # else:
                    #     sent_s_t_2.append([s_idx, s_idx + 1])
                    # tri_s_2[s_idx][s_t] = p_1

                    '''random select'''
                    # if 0 not in trigger_span[i]:
                    #     continue
                    # idx = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # while trigger_span[i][idx] == 1:
                    #     idx = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # sent_s_c_1.append([p, 1 - p])
                    # sent_s_t_1.append([s_idx, idx])
                    # tri_s_1[s_idx][s_t] = p
                    # tri_s_s_1[s_idx] = p

                    # p_1 = random.random()
                    # idx_1 = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # while trigger_span[i][idx_1] == 1 or idx_1 == idx:
                    #     idx_1 = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # sent_s_c_2.append([p_1, 1 - p_1])
                    # sent_s_t_2.append([s_idx, idx_1])
                    # tri_s_2[s_idx][s_t] = p_1

                    '''end'''
                    p = random.random()

                    e_idx = e_idx_l[0][j]
                    e_t = e_idx_l[1][j]
                    sent_e_c.append([1, 0])

                    '''select the neighbor'''
                    sent_e_c_1.append([p, 1 - p])
                    if e_idx == len(sent_id) - 2:
                        sent_e_t_1.append([e_idx, e_idx - 1])
                    else:
                        sent_e_t_1.append([e_idx, e_idx + 1])
                    tri_e_1[e_idx][e_t] = p

                    # p_1 = random.random()
                    # sent_e_c_2.append([p_1, 1- p_1])
                    # if e_idx != len(sent_id) - 2:
                    #     sent_e_t_2.append([e_idx, e_idx + 1])
                    # else:
                    #     sent_e_t_2.append([e_idx, e_idx - 1])
                    # tri_e_2[e_idx][e_t] = p_1

                    '''random select'''
                    # # if 0 in trigger_span[i]:
                    # idx = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # while trigger_span[i][idx] == 1:
                    #     idx = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # sent_e_c_1.append([p, 1 - p])
                    # sent_e_t_1.append([e_idx, idx])
                    # tri_e_1[e_idx][e_t] = p
                    # tri_s_e_1[s_idx] = p

                    # p_1 = random.random()
                    # idx_1 = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # while trigger_span[i][idx_1] == 1 or idx_1 == idx:
                    #     idx_1 = random.sample(list(range(len(trigger_span[i]))), 1)[0]
                    # sent_e_c_2.append([p_1, 1 - p_1])
                    # sent_e_t_2.append([e_idx, idx_1])
                    # tri_e_2[e_idx][e_t] = p_1

                if 0 in trigger_span[i]:
                    # original
                    sent_id_list_new.append(sent_id)
                    trigger_start_new.append(list(trigger_start[i]))
                    trigger_end_new.append(list(trigger_end[i]))
                    # trigger_start_com.append(sent_s_c)
                    trigger_start_tag.append(sent_s_t_1)
                    # trigger_end_com.append(sent_e_c)
                    trigger_end_tag.append(sent_e_t_1)

                    trigger_start_new_1.append(list(tri_s_1))
                    trigger_end_new_1.append(list(tri_e_1))
                    trigger_start_com_1.append(sent_s_c_1)
                    trigger_end_com_1.append(sent_e_c_1)

                    event_type_list_new.append(event_type_list[i])

                    '''attention'''
                    assert len(sent_e_t_1) == len(trigger_idx_add_list[i])

                    # trigger_sent_start_new.append(trigger_sent_start[i])
                    # trigger_sent_end_new.append(trigger_sent_end[i])

                    # left soft label
                    # if random.random() > 0.5:
                    # sent_id_list_new.append(sent_id)
                    # trigger_start_new.append(list(tri_s_1))
                    # trigger_end_new.append(list(tri_e_1))
                    # trigger_start_com.append(sent_s_c_1)
                    # trigger_start_tag.append(sent_s_t_1)
                    # trigger_end_com.append(sent_e_c_1)
                    # trigger_end_tag.append(sent_e_t_1)
                    # trigger_sent_start_new.append(tri_s_s_1)
                    # trigger_sent_end_new.append(tri_s_e_1)

                        # # right soft label
                    # sent_id_list_new.append(sent_id)
                    # trigger_start_new.append(list(tri_s_2))
                    # trigger_end_new.append(list(tri_e_2))
                    # trigger_start_com.append(sent_s_c_2)
                    # trigger_start_tag.append(sent_s_t_2)
                    # trigger_end_com.append(sent_e_c_2)
                    # trigger_end_tag.append(sent_e_t_2)

            return sent_id_list_new, trigger_start_new, trigger_end_new, trigger_start_com, trigger_start_tag, \
                   trigger_end_com, trigger_end_tag, trigger_start_new_1, trigger_end_new_1, trigger_start_com_1, \
                   trigger_end_com_1, sent_id_add_list, trigger_idx_add_list, event_type_list_new

        else:
            return sent_id_list, trigger_start, trigger_end

def preprocess_tac_data(task, subtask):
    task_data = []
    # for task in ['train', 'dev']:
    if task == 'test':
        with open(config.previous_path + '/data/tac2015/dev.pkl', 'rb') as f:
            task_data.extend(pickle.load(f))
    else:
        with open(config.previous_path + '/data/tac2015/' + task + '.pkl', 'rb') as f:
            task_data.extend(pickle.load(f))
    task_data = [l for l in task_data if l['event']]


    trigger_start, trigger_end, sent_id_list = [[] for _ in range(3)]
    num_1 = 0
    for line in task_data:
        sent = line['sentence'].replace('\\', '').replace('``', '"').replace("''", '"').lower()
        sent_token = tokenizer.tokenize(sent)
        sent_id = tokenizer.convert_tokens_to_ids(sent_token)

        trigger_s, trigger_e = [np.zeros((len(sent_id), config_tac.event_type_num), int) for _ in range(2)]
        for event in line['event']:
            trigger = event[: event.index('_')]
            trigger_token = tokenizer.tokenize(trigger)
            trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)

            event_type = event[event.index('_') + 1: ]
            e_t = config_tac.event_type_t2i[event_type]
            for i in range(len(sent_id)):
                if sent_id[i: i + len(trigger_id)] == trigger_id:
                    trigger_s[i][e_t] = 1
                    trigger_e[i + len(trigger_id) - 1][e_t] = 1
                    # break

        if 1 in trigger_s:
            for key in config_tac.event_upper_type_t2id:
                sent_id  = sent_id + config_tac.event_upper_type_t2id[key] + [102]

            sent_id_list.append(sent_id)
            trigger_start.append(list(trigger_s))
            trigger_end.append(list(trigger_e))
        else:
            num_1 += 1

    print('trigger cannot be tagged: %.4f' % (num_1/ len(task_data)))
    return sent_id_list, trigger_start, trigger_end

def preprocess_argument_data(task):
    task_data = []
    # for task in ['train', 'dev', 'test']:
    with open(config.previous_path + '/data/' + task + '.json', 'rb') as f:
        task_data.extend(json.load(f))
    task_data = [line for line in task_data if line['golden-event-mentions']]

    sent_com_id_list, argument_start, argument_end, trigger_position, sent_com_id_add_list, argument_add_list, \
    argument_start_idx, argument_end_idx, argument_start_mix, argument_end_mix, trigger_position_add, \
    argument_start_prob, argument_end_prob = [[] for _ in range(13)]
    num_1, num_2, num_3, num_4 = [0 for _ in range(4)]
    for _, line in tqdm(enumerate(task_data)):
        sent = line['sentence'].replace('\\', '').replace('``', '"').replace("''", '"').lower()
        sent_token = tokenizer.tokenize(sent)
        sent_token.insert(0, '[CLS]')
        sent_token.append('[SEP]')
        sent_id = tokenizer.convert_tokens_to_ids(sent_token)

        trigger_l = []
        trigger_s, trigger_e, trigger_s_add, trigger_e_add \
            = [np.zeros((len(sent_id) + 30, config.event_type_num), int) for _ in range(4)]
        for event in line['golden-event-mentions']:
            trigger_text = event['trigger']['text'].replace('\n', ' ')
            trigger_token = tokenizer.tokenize(trigger_text)
            trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)

            event_type_id = config.event_type_t2id[event['event_type']]
            span_id = trigger_id + event_type_id
            sent_com_id = [101] + span_id + [102] + sent_id[1:]
            # sent_com_id = sent_id

            # trigger_info
            e_t = config.event_type_t2i[event['event_type']]
            trigger_sp, trigger_sp_add = [[0] * (len(sent_id) + 30) for _ in range(2)]
            for k in range(len(span_id) + 2, len(sent_com_id)):
            # for k in range(len(sent_com_id)):
                if sent_com_id[k: k + len(trigger_id)] == trigger_id:
                    # if 1 in trigger_s[k, :] and 1 in trigger_e[k + len(trigger_id) - 1, :]:
                    #     continue
                    # trigger_s[k][e_t] = 1
                    # trigger_e[k + len(trigger_id) - 1][e_t] = 1
                    trigger_sp[k: k + len(trigger_id)] = [1] * len(trigger_id)
                    tri_idx = [k, k + len(trigger_id) - 1]
                    break

            # sent_com_id.insert(tri_idx[0], tokenizer.convert_tokens_to_ids(['<' + event['event_type'] + '>'])[0])
            # sent_com_id.insert(tri_idx[1] + 2, tokenizer.convert_tokens_to_ids(['</' + event['event_type'] + '>'])[0])
            # trigger_sp[tri_idx[0] + 1: tri_idx[1] + 2] = [1] * len(trigger_id)
            # print(tri_idx[0], tri_idx[1] + 2)

            # argument
            arg_s, arg_e = [np.zeros((len(sent_com_id), config.role_type_num), int) for _ in range(2)]
            arg_dict, arg_dict_add = [{} for _ in range(2)]
            arg_list, arg_list_add = [[] for _ in range(2)]
            trigger_list = [event['trigger']['text'].replace('\n', ' ').lower() for event in
                            line['golden-event-mentions']]
            trigger_t = trigger_text + '_' + str(event['trigger']['start']) + '_' + str(event['trigger']['end'])
            idx_l = []
            if trigger_t not in trigger_l:
                '''construct positive samples'''
                argument_list = [revise_argument(arg['text']) for arg in event['arguments']]
                sent_aug = eda(sent, alpha_sr=0.1, alpha_ri=0, alpha_rs=0, p_rd=0, num_aug=1,
                               trigger_list=argument_list + trigger_list)[0]
                if sent == sent_aug and sent and argument_list \
                        and max([len(a.split(' ')) for a in argument_list]) > len(sent.split(' ')) - 4:
                    sent_aug = sent
                    num_3 += 1
                elif sent == sent_aug:
                    sent_aug = sent
                    num_3 += 1
                    # sent_aug = eda(sent, alpha_sr=0, alpha_ri=0.1, alpha_rs=0, p_rd=0, num_aug=1,
                    #                trigger_list=argument_list + trigger_list)[0]
                sent_a_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_aug))
                sent_com_id_add = [101] + span_id + [102] + sent_a_id + [102]
                # sent_com_id_add = [101] + sent_a_id + [102]

                e_t = config.event_type_t2i[event['event_type']]
                for k in range(len(span_id) + 2, len(sent_com_id_add)):
                # for k in range(len(sent_com_id_add)):
                    if sent_com_id_add[k: k + len(trigger_id)] == trigger_id:
                        # if 1 in trigger_s_add[k, :] and 1 in trigger_e_add[k + len(trigger_id) - 1, :]:
                        #     continue
                        # trigger_s_add[k][e_t] = 1
                        # trigger_e_add[k + len(trigger_id) - 1][e_t] = 1
                        trigger_sp_add[k: k + len(trigger_id)] = [1] * len(trigger_id)
                        tri_idx = [k, k + len(trigger_id) - 1]
                        break

                # sent_com_id_add.insert(tri_idx[0],
                #                        tokenizer.convert_tokens_to_ids(['<' + event['event_type'] + '>'])[0])
                # sent_com_id_add.insert(tri_idx[1] + 2,
                #                        tokenizer.convert_tokens_to_ids(['</' + event['event_type'] + '>'])[0])
                # trigger_sp_add[tri_idx[0] + 1: tri_idx[1] + 2] = [1] * len(trigger_id)

                for arg in event['arguments']:
                    r_t = config.role_type_t2i[arg['role']]
                    arg_text = revise_argument(arg['text'])
                    arg_token = tokenizer.tokenize(arg_text)
                    arg_id = tokenizer.convert_tokens_to_ids(arg_token)

                    # judge whether argument appears more than once
                    flag = True
                    # if len(arg_id) == 1 and sum([1 for t in sent_a_id if t == arg_id[0]]) > 1:
                    #     flag = False
                    # elif len(arg_id) != 1 and len(re.findall(' '.join([str(a) for a in arg_id]),
                    #                                          ' '.join([str(s) for s in sent_a_id]))) > 1:
                    #     flag = False

                    for k in range(len(span_id) + 2, len(sent_com_id_add)):
                    # for k in range(len(sent_com_id_add)):
                        if sent_com_id_add[k: k + len(arg_id)] == arg_id:
                            if flag:
                                idx_l.append((k, k + len(arg_id) - 1, r_t))
                                arg_list_add.append((k, k + len(arg_id) - 1))
                                break
                            else:
                                key = arg_text + '_' + str(r_t)
                                if key not in arg_dict_add:
                                    arg_dict_add[key] = []
                                arg_dict_add[key].append((k, k + len(arg_id) - 1, r_t))

                # select the nearest one
                if arg_dict_add:
                    if len(arg_list_add) == 0:
                        # Todo 全标
                        for key in arg_dict_add:
                            idx = random.sample(list(range(len(arg_dict_add[key]))), 1)[0]
                            arg_sel = arg_dict_add[key][idx]
                            idx_l.append(arg_sel)
                    else:
                        for key in arg_dict_add:
                            arg_l = [l for l in arg_dict_add[key]]
                            anc_l = [l[0] for l in arg_list_add]
                            arg_d = [sum([abs(l[0] - x) for x in anc_l]) for l in arg_l]
                            idx = arg_d.index(min(arg_d))
                            arg_sel = arg_l[idx]
                            idx_l.append(arg_sel)

                '''tag index'''
                for arg in event['arguments']:
                    r_t = config.role_type_t2i[arg['role']]
                    arg_text = arg['text'].replace('\\', '').replace('``', '"').replace("''", '"').lower()
                    arg_token = tokenizer.tokenize(arg_text)
                    arg_id = tokenizer.convert_tokens_to_ids(arg_token)

                    # judge whether argument appears more than once
                    flag = True
                    # if len(arg_id) == 1 and sum([1 for t in sent_id if t == arg_id[0]]) > 1:
                    #     flag = False
                    # elif len(arg_id) != 1 and len(re.findall(' '.join([str(a) for a in arg_id]),
                    #                                          ' '.join([str(s) for s in sent_id]))) > 1:
                    #     flag = False

                    for k in range(len(span_id) + 2, len(sent_com_id)):
                    # for k in range(len(sent_com_id)):
                        if sent_com_id[k: k + len(arg_id)] == arg_id:
                            if flag:
                                arg_s[k][r_t] = 1
                                arg_e[k + len(arg_id) - 1][r_t] = 1
                                arg_list.append((k, k + len(arg_id) - 1))
                                break
                            else:
                                key = arg_text + '_' + str(r_t)
                                if key not in arg_dict:
                                    arg_dict[key] = []
                                arg_dict[key].append((k, k + len(arg_id) - 1, r_t))
                trigger_l.append(trigger_t)

                # select the nearest one
                if arg_dict:
                    if len(arg_list) == 0:
                        # Todo 全标
                        for key in arg_dict:
                            idx = random.sample(list(range(len(arg_dict[key]))), 1)[0]
                            # for idx in range(len(arg_dict[key])):
                            arg_sel = arg_dict[key][idx]
                            # while 1 in arg_s[arg_sel[0]]:
                            #     idx = random.sample(list(range(len(arg_dict[key]))), 1)[0]
                            #     arg_sel = arg_dict[key][idx]

                            arg_s[arg_sel[0]][arg_sel[2]] = 1
                            arg_e[arg_sel[1]][arg_sel[2]] = 1
                    else:
                        for key in arg_dict:
                            arg_l = [l for l in arg_dict[key]]
                            anc_l = [l[0] for l in arg_list]
                            arg_d = [sum([abs(l[0] - x) for x in anc_l]) for l in arg_l]
                            idx = arg_d.index(min(arg_d))
                            arg_sel = arg_l[idx]

                            # while 1 in arg_s[arg_sel[0]]:
                            #     idx = random.sample(list(range(len(arg_dict[key]))), 1)[0]
                            #     arg_sel = arg_l[idx]

                            arg_s[arg_sel[0]][arg_sel[2]] = 1
                            arg_e[arg_sel[1]][arg_sel[2]] = 1

                arg_list =set([(revise_argument(arg['text']), arg['role']) for arg in event['arguments']])
                flag_tag = True
                if sum(sum(arg_s)) != len(arg_list):
                    num_2 += 1
                    flag_tag = False
                #     print(len(argument_list), sum(sum(arg_s)))
                #     print(sent)
                #     print([(arg['text'], arg['role']) for arg in  event['arguments']])
                #     # print(sent_aug[0])
                #     print()

                flag_equal = True
                if len(set(idx_l)) != max(len(np.where(arg_s == 1)[0]), len(np.where(arg_e == 1)[0])):
                    num_4 += 1
                    flag_equal = False
                    # s = np.where(arg_s == 1)
                    # e = np.where(arg_e == 1)
                    #
                    # print(sent)
                    # print(sent_aug)
                    # print(len(set(idx_l)), len(np.where(arg_s == 1)[0]))
                    # print([(arg['text'], arg['role']) for arg in event['arguments']])
                    #
                    #
                    # print(set(idx_l))
                    # # print([[s[l], e[l]] for l in range(len(s))])
                    # print(s, e)
                    # print()

                '''mixup'''
                idx_n = sent_com_id.index(102) + 1
                # idx_n = 1
                arg_s_l, arg_e_l, arg_s_p, arg_e_p = [[] for _ in range(4)]
                arg_s_t, arg_e_t = [np.zeros((len(sent_com_id), config.role_type_num), float) for _ in range(2)]
                # start
                s_idx_l = np.where(arg_s == 1)
                for j in range(len(s_idx_l[0])):
                    p = random.random()
                    s_idx = s_idx_l[0][j]
                    s_t = s_idx_l[1][j]
                    '''select the neighbor'''
                    arg_s_p.append([p, 1 - p])
                    if s_idx == idx_n:
                        arg_s_l.append([s_idx, s_idx + 1])
                    else:
                        arg_s_l.append([s_idx, s_idx - 1])
                    arg_s_t[s_idx][s_t] = p
                if len(arg_s_l) == 0:
                    arg_s_l.append([0, 0])
                    arg_s_p.append([0, 0])

                # end
                e_idx_l = np.where(arg_e == 1)
                for j in range(len(e_idx_l[0])):
                    p = random.random()
                    e_idx = e_idx_l[0][j]
                    e_t = e_idx_l[1][j]
                    '''select the neighbor'''
                    arg_e_p.append([p, 1 - p])
                    if e_idx == len(sent_com_id) - 2:
                        arg_e_l.append([e_idx, e_idx - 1])
                    else:
                        arg_e_l.append([e_idx, e_idx + 1])
                    arg_e_t[e_idx][e_t] = p
                if len(arg_e_l) == 0:
                    arg_e_l.append([0, 0])
                    arg_e_p.append([0, 0])

                if (1 in arg_s and 1 in trigger_sp and flag_tag and flag_equal and 1 in trigger_sp_add) \
                    or (len(event['arguments']) == 0 and 1 in trigger_sp and 1 in trigger_sp_add):
                    sent_com_id_list.append(sent_com_id)
                    argument_start.append(list(arg_s))
                    argument_end.append(list(arg_e))
                    trigger_position.append(trigger_sp)

                    sent_com_id_add_list.append(sent_a_id)
                    argument_add_list.append([l[: -1] for l in list(set(idx_l))])

                    argument_start_idx.append(arg_s_l)
                    argument_end_idx.append(arg_e_l)
                    argument_start_mix.append(list(arg_s_t))
                    argument_end_mix.append(list(arg_e_t))
                    trigger_position_add.append(trigger_sp_add)

                    argument_start_prob.append(arg_s_p)
                    argument_end_prob.append(arg_e_p)
                else:
                    num_1 += 1

    print('augement sentences do not change: %.4f' % (num_3/ len(sent_com_id_list)))
    print('arguments cannot be totally tagged in the augmentation: %.4f' % (num_4/ len(sent_com_id_list)))
    print('tag index num != set entity num: %.4f' % (num_2/ len(sent_com_id_list)))
    print('arguments cannot be labeled in the sentence: %.4f' % (num_1 /
                                                sum([len(line['golden-event-mentions']) for line in task_data])))
    # if task == 'dev':
    #     assert 1==-1
    return sent_com_id_list, argument_start, argument_end, trigger_position, sent_com_id_add_list, argument_add_list, \
           argument_start_idx, argument_end_idx, argument_start_mix, argument_end_mix, trigger_position_add, \
           argument_start_prob, argument_end_prob

class gen_data(Dataset):
    def __init__(self, task, subtask, input=None):
        super(gen_data, self).__init__()
        self.task = task
        self.subtask = subtask

        if self.subtask == 'other':
            self.sent_id, self.sent_tag = preprocess_data(task, subtask)

        # if self.task == 'train' and self.subtask == 'trigger':
        if self.task != 'test' and self.subtask == 'trigger':
            self.sent_id, self.trigger_start, self.trigger_end, _, self.trigger_start_tag, \
            _, self.trigger_end_tag, self.trigger_start_1, self.trigger_end_1, \
            self.trigger_start_com_1, self.trigger_end_com_1, self.sent_add_id, self.trigger_add_idx, \
            self.event_type_list = preprocess_data(task, subtask)
        # elif self.task == 'dev' and self.subtask == 'trigger':
        #     self.sent_id, self.trigger_start, self.trigger_end = preprocess_data(task, subtask)
        elif self.task == 'test' and self.subtask == 'trigger':
            self.sent_id, _, _ = preprocess_data(task, subtask)

        if self.task != 'test' and self.subtask == 'argument':
            self.sent_com_id, self.argument_start, self.argument_end, self.trigger_position, self.sent_com_id_add, \
            self.argument_add_idx, self.argument_start_idx, self.argument_end_idx, self.argument_start_mix, \
            self.argument_end_mix, self.trigger_position_add, self.argument_start_prob, self.argument_end_prob \
                = preprocess_argument_data(task)
        elif self.task == 'test' and self.subtask == 'argument':
            self.sent_com_id, self.sent_com_id_mask, self.trigger_position = input

    def __len__(self):
        if self.subtask != 'argument':
            return len(self.sent_id)
        elif self.subtask == 'argument':
            return len(self.sent_com_id)

    def __getitem__(self, item):
        # other

        if self.subtask == 'other':
            max_len = [config.text_max_len, config.event_type_num]
            sent_id_item = padding(self.sent_id[item], max_len, 1)
            return torch.LongTensor(sent_id_item), torch.FloatTensor([self.sent_tag[item]])

        # trigger
        # if self.task == 'train' and self.subtask == 'trigger':
        if self.task != 'test' and self.subtask == 'trigger':
            max_len = [config.text_max_len, config.event_type_num]
            sent_id_item = padding(self.sent_id[item], max_len, 1)
            trigger_start_item = padding(self.trigger_start[item], max_len, 2)
            trigger_end_item = padding(self.trigger_end[item], max_len, 2)

            # trigger_start_com_item = padding(self.trigger_start_com[item], [config.trigger_num, 2], 2)
            trigger_start_tag_item = padding(self.trigger_start_tag[item], [config.trigger_num, 2], 2)
            # trigger_end_com_item = padding(self.trigger_end_com[item], [config.trigger_num, 2], 2)
            trigger_end_tag_item = padding(self.trigger_end_tag[item], [config.trigger_num, 2], 2)

            trigger_start_1_item = padding(self.trigger_start_1[item], max_len, 2)
            trigger_end_1_item = padding(self.trigger_end_1[item], max_len, 2)

            trigger_start_com_1_item = padding(self.trigger_start_com_1[item], [config.trigger_num, 2], 2)
            trigger_end_com_1_item = padding(self.trigger_end_com_1[item], [config.trigger_num, 2], 2)

            # trigger_sent_start_item = padding(self.trigger_sent_start[item], max_len, 1)
            # trigger_sent_end_item = padding(self.trigger_sent_end[item], max_len, 1)

            sent_id_add_item = padding(self.sent_add_id[item], max_len, 1)
            trigger_add_idx_item = padding(self.trigger_add_idx[item], [config.trigger_num, 2], 2)

            event_type_item = padding(self.event_type_list[item], [7, 0], 1)

            return torch.LongTensor(sent_id_item), torch.FloatTensor(trigger_start_item), \
                   torch.FloatTensor(trigger_end_item), torch.LongTensor(trigger_start_tag_item), \
                   torch.LongTensor(trigger_end_tag_item), torch.FloatTensor(trigger_start_1_item), \
                   torch.FloatTensor(trigger_end_1_item), torch.FloatTensor(trigger_start_com_1_item), \
                   torch.FloatTensor(trigger_end_com_1_item), torch.LongTensor(sent_id_add_item), \
                   torch.LongTensor(trigger_add_idx_item), torch.LongTensor(event_type_item)

        elif self.task == 'test' and self.subtask == 'trigger':
            max_len = [config.text_max_len, config.event_type_num]
            sent_id_item = padding(self.sent_id[item], max_len, 1)
            return torch.LongTensor(sent_id_item)

        # argument
        if self.task != 'test' and self.subtask == 'argument':
            max_len = [config.text_com_max_len, config.role_type_num]
            sent_com_id_item = padding(self.sent_com_id[item], max_len, 1)
            argument_start_item = padding(self.argument_start[item], max_len, 2)
            argument_end_item = padding(self.argument_end[item], max_len, 2)
            trigger_position_item = padding(self.trigger_position[item], max_len, 1)

            sent_com_id_add_item = padding(self.sent_com_id_add[item], max_len, 1)
            argument_add_idx_item = padding(self.argument_add_idx[item], [config.argument_num, 2], 2)

            argument_start_idx_item = padding(self.argument_start_idx[item], [config.argument_num, 2], 2)
            argument_end_idx_item = padding(self.argument_end_idx[item], [config.argument_num, 2], 2)
            argument_start_mix_item = padding(self.argument_start_mix[item], max_len, 2)
            argument_end_mix_item = padding(self.argument_end_mix[item], max_len, 2)
            trigger_position_add_item = padding(self.trigger_position_add[item], max_len, 1)

            argument_start_prob_item = padding(self.argument_start_prob[item], max_len, 2)
            argument_end_prob_item = padding(self.argument_end_prob[item], max_len, 2)

            return torch.LongTensor(sent_com_id_item), torch.FloatTensor(argument_start_item), \
                   torch.FloatTensor(argument_end_item), torch.LongTensor(trigger_position_item), \
                   torch.LongTensor(sent_com_id_add_item), torch.LongTensor(argument_add_idx_item), \
                   torch.LongTensor(argument_start_idx_item), torch.LongTensor(argument_end_idx_item), \
                   torch.FloatTensor(argument_start_mix_item), torch.FloatTensor(argument_end_mix_item), \
                   torch.LongTensor(trigger_position_add_item), torch.FloatTensor(argument_start_prob_item), \
                   torch.FloatTensor(argument_end_prob_item)

        elif self.task == 'test' and self.subtask == 'argument':
            max_len = [config.text_com_max_len, config.role_type_num]
            sent_com_id_item = padding(self.sent_com_id[item], max_len, 1)
            sent_com_id_mask_item = padding(self.sent_com_id_mask[item], max_len, 1)
            trigger_position_item = padding(self.trigger_position[item], max_len, 1)
            return torch.LongTensor(sent_com_id_item), torch.LongTensor(sent_com_id_mask_item), \
                   torch.LongTensor(trigger_position_item)

class gen_tac_data(Dataset):
    def __init__(self, task, subtask, input=None):
        super(gen_tac_data, self).__init__()
        self.task = task
        self.subtask = subtask

        # if self.subtask == 'other':
        #     self.sent_id, self.sent_tag = preprocess_tac_data(task, subtask)

        # if self.task == 'train' and self.subtask == 'trigger':
        if self.task != 'test':
            self.sent_id, self.trigger_start, self.trigger_end = preprocess_tac_data(task, subtask)
        elif self.task == 'test':
            self.sent_id, _, _ = preprocess_tac_data(task, subtask)

    def __len__(self):
        return len(self.sent_id)

    def __getitem__(self, item):
        # other
        # if self.subtask == 'other':
        #     max_len = [config.text_max_len, config.event_type_num]
        #     sent_id_item = padding(self.sent_id[item], max_len, 1)
        #     return torch.LongTensor(sent_id_item), torch.FloatTensor([self.sent_tag[item]])

        # trigger
        if self.task != 'test':
            max_len = [config_tac.text_max_len, config_tac.event_type_num]
            sent_id_item = padding(self.sent_id[item], max_len, 1)
            trigger_start_item = padding(self.trigger_start[item], max_len, 2)
            trigger_end_item = padding(self.trigger_end[item], max_len, 2)

            # trigger_start_tag_item = padding(self.trigger_start_tag[item], [config.trigger_num, 2], 2)
            # trigger_end_tag_item = padding(self.trigger_end_tag[item], [config.trigger_num, 2], 2)
            #
            # trigger_start_1_item = padding(self.trigger_start_1[item], max_len, 2)
            # trigger_end_1_item = padding(self.trigger_end_1[item], max_len, 2)
            #
            # trigger_start_com_1_item = padding(self.trigger_start_com_1[item], [config.trigger_num, 2], 2)
            # trigger_end_com_1_item = padding(self.trigger_end_com_1[item], [config.trigger_num, 2], 2)
            #
            # sent_id_add_item = padding(self.sent_add_id[item], max_len, 1)
            # trigger_add_idx_item = padding(self.trigger_add_idx[item], [config.trigger_num, 2], 2)
            #
            # event_type_item = padding(self.event_type_list[item], [7, 0], 1)

            return torch.LongTensor(sent_id_item), torch.FloatTensor(trigger_start_item), \
                   torch.FloatTensor(trigger_end_item), \
                   # torch.LongTensor(trigger_start_tag_item), \
                   # torch.LongTensor(trigger_end_tag_item), torch.FloatTensor(trigger_start_1_item), \
                   # torch.FloatTensor(trigger_end_1_item), torch.FloatTensor(trigger_start_com_1_item), \
                   # torch.FloatTensor(trigger_end_com_1_item), torch.LongTensor(sent_id_add_item), \
                   # torch.LongTensor(trigger_add_idx_item), torch.LongTensor(event_type_item)

        elif self.task == 'test':
            max_len = [config.text_max_len, config.event_type_num]
            sent_id_item = padding(self.sent_id[item], max_len, 1)
            return torch.LongTensor(sent_id_item)


def data_loader(dataset, batch_size, shuffle, drop_last):
    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return data_iter