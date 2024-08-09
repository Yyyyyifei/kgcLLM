import argparse
import json
import os

from collections import defaultdict


def split_line(line):
    first_split = line.strip().find("\t")

    if first_split == -1:
        return None, None
    
    return line[:first_split], line[first_split+1:]


def indexify(train_path, valid_path, test_path):
    id_2_ent = {}
    ent_2_id = {}

    id_2_rel = {}
    rel_2_id = {}

    num_entity = 0
    num_relation = 0

    for file in (train_path, valid_path, test_path):
        with open(file, "r") as f:
            lines = f.readlines()

            for l in lines:
                # Fix: strip
                head, relation, tail = l.split()
                head, relation, tail = head.strip(), relation.strip(), tail.strip()
                if head not in ent_2_id:
                    id_2_ent[num_entity] = head
                    ent_2_id[head] = num_entity
                    num_entity += 1

                if tail not in ent_2_id:
                    id_2_ent[num_entity] = tail
                    ent_2_id[tail] = num_entity
                    num_entity += 1
                
                if relation not in rel_2_id:
                    id_2_rel[num_relation] = relation
                    rel_2_id[relation] = num_relation
                    num_relation += 1

    return id_2_ent, ent_2_id, id_2_rel, rel_2_id

def group_degrees(train_path, valid_path, test_path):
    ent_to_degree = {}
    rel_to_degree = {}
    
    for file in (train_path, valid_path, test_path):
        with open(file, "r") as f:
            lines = f.readlines()

            for l in lines:
                head, relation, tail = [ele.strip() for ele in l.split()]
                
                ent_to_degree[head] = ent_to_degree.get(head, 0) + 1
                ent_to_degree[tail] = ent_to_degree.get(tail, 0) + 1
                
                rel_to_degree[relation] = rel_to_degree.get(relation, 0) + 1
                
    return {k: v for k, v in sorted(ent_to_degree.items(), reverse=True, key=lambda x: x[1])}, \
                {k: v for k, v in sorted(rel_to_degree.items(), reverse=True, key=lambda x: x[1])}

def id2text(id2ent, ent2text, id2rel, rel2text):
    id2ent_text = {id : ent2text[id2ent[id]] for id in id2ent}
    id2rel_text = {id : rel2text[id2rel[id]] for id in id2rel}

    return id2ent_text, id2rel_text

def ent_to_text(datapath):
    ent2text = {}

    with open(os.path.join(datapath, "entity2text.txt"), "r") as f:
        for line in f.readlines():
            key, value = split_line(line)
            ent2text[key] = value

    return ent2text

def parse_file(file, entities, relations):
    all_tuples = []
    predict_tail = defaultdict(list)
    predict_head = defaultdict(list)

    with open(file, "r") as f:
        lines = f.readlines()

        for l in lines:
            head, relation, tail = l.split()
            index_head, index_relation, index_tail = entities[head.strip()], relations[relation.strip()], entities[tail.strip()] 

            all_tuples.append((index_head, index_relation, index_tail))
            
            predict_tail[(index_head, index_relation)].append(index_tail)
            predict_head[(index_relation, index_tail)].append(index_head)

    return all_tuples, predict_tail, predict_head

def get_data(datapath):
    # default datapath as "./data"
    ent_path = os.path.join(datapath, "id2ent.json")
    rel_path = os.path.join(datapath, "id2rel.json")

    id2symbol_ent_path = os.path.join(datapath, "id2symbol_ent.json")
    id2symbol_rel_path = os.path.join(datapath, "id2symbol_rel.json")

    train_path = os.path.join(datapath, "train.txt")
    valid_path = os.path.join(datapath, "valid.txt")
    test_path = os.path.join(datapath, "test.txt")

    if os.path.exists(ent_path):
        id2ent = json.load(open(ent_path, "r"))
        id2rel = json.load(open(rel_path, "r"))

        id2symbol_ent = json.load(open(id2symbol_ent_path, "r"))
        id2symbol_rel = json.load(open(id2symbol_rel_path, "r"))

        symbol_ent2id = {v : k for k,v in id2symbol_ent.items()}
        symbol_rel2id = {v : k for k,v in id2symbol_rel.items()}
    else:
        id2symbol_ent, symbol_ent2id, id2symbol_rel, symbol_rel2id = indexify(train_path, valid_path, test_path)
        
        with open(id2symbol_ent_path, "w") as f:
            json.dump(id2symbol_ent, f)
            f.close()
        
        with open(id2symbol_rel_path, "w") as f:
            json.dump(id2symbol_rel, f)
            f.close()

        relation2text = json.load(open(os.path.join(datapath, "alignment_clean.json"), "r"))
        ent2text = ent_to_text(datapath)
        
        id2ent, id2rel = id2text(id2symbol_ent, ent2text, id2symbol_rel, relation2text)

        with open(ent_path, "w") as f:
            json.dump(id2ent, f)
            f.close()
        
        with open(rel_path, "w") as f:
            json.dump(id2rel, f)
            f.close()
    
    all_tuples_train, predict_tail_train, predict_head_train = parse_file(train_path, symbol_ent2id, symbol_rel2id)
    all_tuples_valid, predict_tail_valid, predict_head_valid = parse_file(valid_path, symbol_ent2id, symbol_rel2id)
    all_tuples_test, predict_tail_test, predict_head_test = parse_file(test_path, symbol_ent2id, symbol_rel2id)

    return all_tuples_train, predict_tail_train, predict_head_train, \
                all_tuples_valid, predict_tail_valid, predict_head_valid, \
                    all_tuples_test, predict_tail_test, predict_head_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse the number of queries.')
    parser.add_argument("-p", dest="data_path", type=str, default="./data/kgc_data/FB15k-237", help="path where data is stored")
    
    args = parser.parse_args()

    path = args.data_path
    
    train_path = os.path.join(path, "train.txt")
    valid_path = os.path.join(path, "valid.txt")
    test_path = os.path.join(path, "test.txt")
    
    ent_degree, rel_degree = group_degrees(train_path, valid_path, test_path)
    
    json.dump(ent_degree, open(os.path.join(path,"ent_degree.json"), "w"), indent=2)
    json.dump(rel_degree, open(os.path.join(path,"rel_degree.json"), "w"), indent=2)
    
    tuples_train, predict_tail_train, predict_head_train, tuples_valid, \
        predict_tail_valid, predict_head_valid, tuples_test, predict_tail_test, predict_head_test = get_data(path)

    max = 0
    max_test = 0
    non_known = 0
    
    for question in predict_tail_test:
        known_answers = len(predict_tail_train.get(question, [])) + len(predict_tail_valid.get(question, []))
        test_answers = len(predict_tail_test.get(question, []))
        
        if known_answers == 0:
            non_known += 1
        
        if test_answers > max_test:
            max_test = test_answers
        
        if known_answers > max:
            max = known_answers
            print(known_answers)
            print(question)
    print(max_test)
    print(non_known)
    print(non_known / len(predict_head_test))
    # ent_path = os.path.join(path, "id2ent.json")
    # rel_path = os.path.join(path, "id2rel.json")

    # id2symbol_ent_path = os.path.join(path, "id2symbol_ent.json")
    # id2symbol_rel_path = os.path.join(path, "id2symbol_rel.json")
    
    # id2ent = json.load(open(ent_path, "r"))
    # id2rel = json.load(open(rel_path, "r"))

    # id2symbol_ent = json.load(open(id2symbol_ent_path, "r"))
    # id2symbol_rel = json.load(open(id2symbol_rel_path, "r"))
