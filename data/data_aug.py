import random
from collections import defaultdict

def add_noise(triples):
    noisy_triples = []
    for head, relation, tail in triples:
        if random.random() < 0.1:  # 10%的概率添加噪声
            head += '_noise'
            noisy_triples.append([head, relation, tail])  # 使用列表而不是元组
            print(f"添加噪声三元组: {[head, relation, tail]}")  # 记录被添加噪声的三元组
    return noisy_triples

def augment_low_frequency_entities(triples, threshold=500):
    augmented_triples = []
    entity_frequency = defaultdict(int)

    for head, relation, tail in triples:
        entity_frequency[head] += 1
        entity_frequency[tail] += 1

    for head, relation, tail in triples:
        if entity_frequency[head] < threshold:
            augmented_triples.append([head + "_noise", relation, tail])  # 使用列表
        if entity_frequency[tail] < threshold:
            augmented_triples.append([head, relation, tail + "_noise"])  # 使用列表
        if entity_frequency[head] < threshold and entity_frequency[tail] < threshold:
            new_entity = random.choice(list(entity_frequency.keys()))
            if new_entity != head and new_entity != tail:
                augmented_triples.append([head + "_" + new_entity, relation, tail])  # 使用列表
                augmented_triples.append([head, relation, tail + "_" + new_entity])  # 使用列表

    print(f"增强低频实体的三元组数量: {len(augmented_triples)}")  # 打印增加的三元组数量
    return augmented_triples

def augment_low_frequency_relations(triples, threshold=100):
    augmented_triples = []
    relation_frequency = defaultdict(int)

    for head, relation, tail in triples:
        relation_frequency[relation] += 1
            
    print("关系频率统计:", dict(relation_frequency))  # 打印关系频率

    for head, relation, tail in triples:
        if relation_frequency[relation] < threshold:
            print(f"增强关系: {relation}")  # 记录增强的关系
            augmented_triples.append([head, relation + "_noise", tail])  # 使用列表
            new_relation = relation + "_augmented"  
            augmented_triples.append([head, new_relation, tail])  # 使用列表

    return augmented_triples

def data_ag(triple_list):
    augmented_triples = []
    noisy_triples = add_noise(triple_list)
    augmented_triples.extend(noisy_triples)

    low_freq_entity_triples = augment_low_frequency_entities(triple_list)
    augmented_triples.extend(low_freq_entity_triples)

    low_freq_relation_triples = augment_low_frequency_relations(triple_list)
    augmented_triples.extend(low_freq_relation_triples)

    print("增强后的三元组数量:", len(augmented_triples))  # 打印增强后的三元组数量
    print("增强的三元组示例:", augmented_triples[:5])  # 打印前5个增强的三元组
    return augmented_triples




