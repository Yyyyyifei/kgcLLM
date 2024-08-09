import pickle
import os
import json

def indexify(train_path, valid_path, test_path):
    entities_to_id = {}
    relations_to_id = {}
    
    id_to_ent = {}
    id_to_rel = {}
    
    num_entity = 0
    num_relations = 0
    for file in (train_path, valid_path, test_path):
        with open(file, "r") as f:
            lines = f.readlines()

            for l in lines:
                head, relation, tail = [ele.strip() for ele in l.split()]
                
                if head not in entities_to_id:
                    id_to_ent[num_entity] = head
                    entities_to_id[head] = num_entity
                    num_entity += 1
                
                if tail not in entities_to_id:
                    id_to_ent[num_entity] = tail
                    entities_to_id[tail] = num_entity
                    num_entity += 1
                
                if relation not in relations_to_id:
                    id_to_rel[num_relations] = relation
                    relations_to_id[relation] = num_relations
                    num_relations += 1
                    
    return id_to_ent, id_to_rel


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


def truncate_text(text_path):
    with open(text_path, "r") as f:
        with open("/scratch/ssd004/scratch/zjt/yifei/kgc_pipeline/data/kgc_data/WIN18RR/truncated_text.txt", "w") as g:
            lines = f.readlines()
            
            truncated_lines = []
            for l in lines:
                id, description = l.split("\t", 1)
                description = description[:description.find(",")]
                
                g.write(f"{id}\t{description}\n")
        
if __name__ == "__main__":
    base_path = "/scratch/ssd004/scratch/zjt/yifei/kgc_pipeline/data/kgc_data/WIN18RR"
    train_path = os.path.join(base_path, "train.txt")
    valid_path = os.path.join(base_path, "valid.txt")
    test_path = os.path.join(base_path, "test.txt")
    
    id_to_ent, id_to_rel = indexify(train_path, valid_path, test_path)
    
    # pickle.dump(id_to_ent, open(os.path.join(base_path, "id2ent.pkl"), "wb"))
    # pickle.dump(id_to_rel, open(os.path.join(base_path, "id2rel.pkl"), "wb"))
    
    # json.dump(id_to_ent, open(os.path.join(base_path, "id2symbol_ent.json"), "w"), indent=2)
    # json.dump(id_to_rel, open(os.path.join(base_path, "id2symbol_rel.json"), "w"), indent=2)
                
    # ent_to_degree, rel_to_degree = group_degrees(train_path, valid_path, test_path)
    
    # json.dump(ent_to_degree, open(os.path.join(base_path,"ent_degree.json"), "w"), indent=2)
    # json.dump(rel_to_degree, open(os.path.join(base_path,"rel_degree.json"), "w"), indent=2)
    
    truncate_text(os.path.join(base_path, "entity2text.txt"))
    
# def parse_file(path):