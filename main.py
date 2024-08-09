import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pdb
import pandas as pd
import numpy as np
import torch
import random
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import tqdm

import logging

def seed_everything(seed: int):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(cfg):

    ent2text = pd.read_csv(os.path.join(cfg.data_root, "entity2text.txt"), sep="\t", header=None, names=["entity", "text"], dtype={'entity': str, "text": str})
    ent2text = ent2text.set_index("entity").to_dict()["text"]
    
    id2rel = json.load(open(os.path.join(cfg.data_root, "id2symbol_rel.json")))
    id2rel = {int(k): v for k, v in id2rel.items()}
    rel2id = {v: int(k) for k, v in id2rel.items()}
    
    id2ent = json.load(open(os.path.join(cfg.data_root, "id2symbol_ent.json")))
    id2ent = {int(k): v for k, v in id2ent.items()}
    ent2id = {v: int(k) for k, v in id2ent.items()}

    data_train = pd.read_csv(os.path.join(cfg.data_root, "train.txt"), sep="\t", header=None, names=["head", "relation", "tail"], dtype={'head': str, "relation": str, "tail": str})
    data_valid = pd.read_csv(os.path.join(cfg.data_root, "valid.txt"), sep="\t", header=None, names=["head", "relation", "tail"], dtype={'head': str, "relation": str, "tail": str})
    data_test = pd.read_csv(os.path.join(cfg.data_root, "test.txt"), sep="\t", header=None, names=["head", "relation", "tail"], dtype={'head': str, "relation": str, "tail": str})
    data_known = pd.concat([data_train, data_valid], axis=0)

    data_known["head_text"] = data_known["head"].map(ent2text)
    data_known["tail_text"] = data_known["tail"].map(ent2text)
    data_known["relation_id"] = data_known["relation"].map(rel2id).astype(int)
    data_known["head_id"] = data_known["head"].map(ent2id).astype(int)
    data_known["tail_id"] = data_known["tail"].map(ent2id).astype(int)
    
    data_test["relation_id"] = data_test["relation"].map(rel2id).astype(int)
    data_test["head_id"] = data_test["head"].map(ent2id).astype(int)
    data_test["tail_id"] = data_test["tail"].map(ent2id).astype(int)
    
    return data_known, data_test, id2rel, rel2id, id2ent, ent2id, ent2text


def construct_prompt_head(data_known, line, all_candidates, id2ent, ent2text, id2rel):
    
    tail_id, relation_id, head_ids = line
    
    known_heads = data_known[(data_known["tail_id"] == tail_id) & (data_known["relation_id"] == relation_id)]["head_text"].values.tolist()
    
    tail_text = ent2text[id2ent[tail_id]]
    relation_text = id2rel[relation_id]
    answer_texts = [ent2text[id2ent[head_id]] for head_id in head_ids]
    
    candidates_head = list(all_candidates - set(known_heads))
    candidates_head = sorted(candidates_head, key=lambda x: len(tokenizer(x).input_ids))
    
    prompt_head = f"Complete this triple. Tail entity: {tail_text}, Relation: {relation_text}, Head Entity: "
    
    return prompt_head, candidates_head, answer_texts


def construct_prompt_tail(data_known, line, all_candidates, id2ent, ent2text, id2rel):
    
    head_id, relation_id, tail_ids = line
    
    known_tails = data_known[(data_known["head_id"] == head_id) & (data_known["relation_id"] == relation_id)]["tail_text"].values.tolist()
    
    head_text = ent2text[id2ent[head_id]]
    relation_text = id2rel[relation_id]
    answer_texts = [ent2text[id2ent[tail_id]] for tail_id in tail_ids]
    
    candidates_tail = list(all_candidates - set(known_tails))
    candidates_tail = sorted(candidates_tail, key=lambda x: len(tokenizer(x).input_ids))
    
    prompt_tail = f"Complete this triple. Head entity: {head_text}, Relation: {relation_text}, Tail Entity: "
    
    return prompt_tail, candidates_tail, answer_texts


def format_q_and_a(question, answer):
    
    messages = [
        {"role": "user", "content": 
            f"{question}"
        },
        {"role": "assistant", "content": 
            f"{answer}"
        }
        
    ]
    
    return messages


def format_q(question):
    
    messages = [
        {"role": "user", "content": 
            f"{question}"
        },
    ]
    
    return messages


class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, prompt, candidates, ent2text, id2ent, tokenizer):
        
        self.prompt = prompt
        self.candidates = candidates
        self.ent2text = ent2text
        self.id2ent = id2ent
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.candidates)
    
    def __getitem__(self, idx):
        candidate = self.candidates[idx]
        
        prompt = format_q(self.prompt)
        prompt_and_answer = format_q_and_a(self.prompt, candidate)
        
        prompt_template = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_and_answer_template = self.tokenizer.apply_chat_template(prompt_and_answer, tokenize=False, add_generation_prompt=False)
        
        input_prompt = self.tokenizer(prompt_template, return_tensors="pt")
        input_prompt_and_answer = self.tokenizer(prompt_and_answer_template, return_tensors="pt")
        prompt_length = len(input_prompt["input_ids"][0])
        answer_length = len(input_prompt_and_answer["input_ids"][0]) - prompt_length
        
        return prompt_and_answer_template, candidate, prompt_length, answer_length        


def get_first_answer_probs(question, answer, model, tokenizer, get_cache=False, use_cache=None):
    
    prompt = format_q(question)
    prompt_and_answer = format_q_and_a(question, answer)
    
    prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_and_answer_template = tokenizer.apply_chat_template(prompt_and_answer, tokenize=False, add_generation_prompt=False)
    
    input_prompt = tokenizer(prompt_template, return_tensors="pt")
    input_prompt_and_answer = tokenizer(prompt_and_answer_template, return_tensors="pt")
    answer_length = len(input_prompt_and_answer["input_ids"][0]) - len(input_prompt["input_ids"][0])
    answer_ids = input_prompt_and_answer["input_ids"][0, -answer_length:]
    
    # TODO: Check if this is necessary, multi gpu?
    input_prompt_and_answer = input_prompt_and_answer.to(model.device)
    answer_ids = answer_ids.to(model.device)
    
    outputs = model(**input_prompt_and_answer, use_cache=True)
    
    logits = outputs.logits[0, -answer_length-1:-1]
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    if get_cache:
        kv_cache = tuple(
                    tuple(kv[:, :, :-answer_length-1, :] for kv in layer)
                    for layer in outputs.past_key_values
                )

    answer_logprob = logprobs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).mean().item()
    
    return answer_logprob, kv_cache

def get_ranks(records):
    
    records = sorted(records, key=lambda x: x[1], reverse=True)
    ranks = []
    
    for i in range(len(records)):
        if records[i][2] == 1:
            ranks.append(i + 1 - len(ranks))
    
    return ranks


def inference(data_known, data_test, id2ent, model, tokenizer, ent2text, id2rel, cfg):
    
    all_candidates = set(ent2text.values())
    
    data_test_tail = data_test.groupby(["head_id", "relation_id"]).agg({"tail_id": list}).reset_index().values.tolist()
    
    for line in tqdm.tqdm(data_test_tail):
        prompt, candidates, answers = construct_prompt_tail(data_known, line, all_candidates, id2ent, ent2text, id2rel)
        first_answer_logprob, tail_kv_cache = get_first_answer_probs(prompt, answers[0], model, tokenizer, get_cache=True)
        dataset = TestDataset(prompt, candidates, ent2text, id2ent, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=cfg.batch_size, 
                                                    shuffle=False, 
                                                    drop_last=False,
                                                    num_workers=cfg.num_workers)
        tail_kv_cache = tuple(tuple(past_kv.repeat(cfg.batch_size, 1, 1, 1) for past_kv in layer) for layer in tail_kv_cache)
        logging.info(prompt)
        logging.info(answers)
        
        saves = []
        
        answer_index = []
        for i, c in enumerate(candidates):
            if c in answers:
                answer_index.append(i)
            if len(answer_index) == len(answers):
                break
        
        answer_index = torch.tensor(answer_index)
        
        all_results = torch.zeros(len(candidates))
        acc = 0
        
        for batch in dataloader:
            prompt_and_answer_templates, candidates, prompt_lengths, answer_lengths = batch
            inputs = tokenizer(prompt_and_answer_templates, return_tensors="pt", padding=True)
            
            bs = len(prompt_and_answer_templates)
            if bs != cfg.batch_size:
                tail_kv_cache = tuple(tuple(past_kv[:bs] for past_kv in layer) for layer in tail_kv_cache)
            
            inputs["input_ids"] = inputs["input_ids"][:, prompt_lengths[0]-1:]
            inputs = inputs.to(model.device)
            logits = model(**inputs, past_key_values=tail_kv_cache).logits
            
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # make this faster by batch operations
            # for i in range(len(logprobs)):
            #     candidate_logprob = logprobs[i][:answer_lengths[i]].gather(dim=-1, index=inputs["input_ids"][i, 1:answer_lengths[i]+1].unsqueeze(-1)).mean().item()
            #     # batch_probs.append(candidate_logprob)
            #     if candidates[i] in answers:
            #         saves.append([candidates[i], candidate_logprob, 1])
            #     else:
            #         saves.append([candidates[i], candidate_logprob, 0])
            #     if candidates[i] == answers[0]:
            #         pass
            # #         # TODO: why different from first_answer_logprob?
            # #         assert candidate_logprob == first_answer_logprob            
            mask = torch.cat((inputs["attention_mask"][:, prompt_lengths[0]:], torch.zeros(bs, 1).to(model.device)), dim=1).unsqueeze(-1)
            logprobs.masked_fill_(~mask.to(torch.bool), 0)
            gathering_index = inputs["input_ids"][:, 1:].unsqueeze(-1)
            aggregated_results = logprobs[:, :-1, :].gather(dim=-1, index=gathering_index).sum(dim=1).squeeze() / answer_lengths.to(model.device)
            
            all_results[acc: acc+bs] = aggregated_results
            acc += bs
        
        argsort = torch.argsort(all_results, descending=True)
        rankings = argsort.scatter_(0, argsort.clone(), torch.arange(argsort.size(0)))
        ranks = torch.sort(rankings[answer_index])[0] + 1 - torch.arange(answer_index.size(0))
        
        logging.info(ranks)
        
        # ranks = get_ranks(saves)
        # logging.info(ranks)
    
    
    data_test_head = data_test.groupby(["tail_id", "relation_id"]).agg({"head_id": list}).reset_index().values.tolist()
    print("N_data_test_head", len(data_test_head))
    
    for line in tqdm.tqdm(data_test_head):
        prompt, candidates, answers = construct_prompt_head(data_known, line, all_candidates, id2ent, ent2text, id2rel)
        first_answer_logprob, head_kv_cache = get_first_answer_probs(prompt, answers[0], model, tokenizer, get_cache=True)
        dataset = TestDataset(prompt, candidates, ent2text, id2ent, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg.batch_size, 
                                                 shuffle=False, 
                                                 drop_last=False,
                                                 num_workers=cfg.num_workers)
        head_kv_cache = tuple(tuple(past_kv.repeat(cfg.batch_size, 1, 1, 1) for past_kv in layer) for layer in head_kv_cache)
        logging.info(prompt)
        logging.info(answers)
        
        saves = []
        for batch in dataloader:
            prompt_and_answer_templates, candidates, prompt_lengths, answer_lengths = batch
            inputs = tokenizer(prompt_and_answer_templates, return_tensors="pt", padding=True)
            
            bs = len(prompt_and_answer_templates)
            if bs != cfg.batch_size:
                head_kv_cache = tuple(tuple(past_kv[:bs] for past_kv in layer) for layer in head_kv_cache)
            
            inputs["input_ids"] = inputs["input_ids"][:, prompt_lengths[0]-1:]
            inputs = inputs.to(model.device)
            logits = model(**inputs, past_key_values=head_kv_cache).logits
            
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            for i in range(len(logprobs)):
                candidate_logprob = logprobs[i][:answer_lengths[i]].gather(dim=-1, index=inputs["input_ids"][i, 1:answer_lengths[i]+1].unsqueeze(-1)).mean().item()
                if candidates[i] in answers:
                    saves.append([candidates[i], candidate_logprob, 1])
                else:
                    saves.append([candidates[i], candidate_logprob, 0])
                if candidates[i] == answers[0]:
                    pass

        ranks = get_ranks(saves)
        logging.info(ranks)


def prepare_llm(cfg):
    
    model_id = cfg.model_root
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    vocab = tokenizer.get_vocab()
    vocab_rev = {v: k for k, v in vocab.items()}
    
    return model, tokenizer, vocab, vocab_rev


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # parser.add_argument("--model_root", default="/datadrive/josephtang/LLaMA-Factory/models/llama3_lora_FB15k-237", type=str)
    # parser.add_argument("--model_root", default="/datadrive/josephtang/kg-llm/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa", type=str)
    parser.add_argument("--data_root", default="./data/kgc_data/FB15k-237", type=str)
    parser.add_argument("--model_root", default="/scratch/ssd004/scratch/zjt/LLaMA-Factory/models/FB15k-237-4N", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)
    seed_everything(cfg.seed)
    
    data_known, data_test, id2rel, rel2id, id2ent, ent2id, ent2text = load_data(cfg)
    model, tokenizer, vocab, vocab_rev = prepare_llm(cfg)
    with torch.no_grad():
        inference(data_known, data_test, id2ent, model, tokenizer, ent2text, id2rel, cfg)
    