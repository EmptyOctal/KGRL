import pandas as pd
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

def process_data(input_path, output_dir):
    # 读取数据
    data = pd.read_csv(input_path, sep='\t', header=None, usecols=range(8), encoding='utf-8')
    data.columns = ['id', 'orgin_id', 'start_lang', 'end_lang', 'weight', 'start_entity', 'relation', 'end_entity']
    
    # 过滤start_lang和end_lang为'zh'的行
    data = data[(data['start_lang'] == 'zh') & (data['end_lang'] == 'zh')]
    
    # 提取三元组和实体、关系
    triples = data[['start_entity', 'relation', 'end_entity']].values
    entities = set(data['start_entity']).union(set(data['end_entity']))
    relations = set(data['relation'])

    # 创建实体和关系的映射
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    # 将entity2id和relation2id写入文件
    with open(f"{output_dir}/entity2id.txt", 'w', encoding='utf-8') as e_file:
        for entity, idx in entity2id.items():
            e_file.write(f"{entity} {idx}\n")

    with open(f"{output_dir}/relation2id.txt", 'w', encoding='utf-8') as r_file:
        for relation, idx in relation2id.items():
            r_file.write(f"{relation} {idx}\n")

    # 训练集和测试集的分割
    train_data, test_data = train_test_split(data, test_size=0.18, random_state=42)

    # 写入训练集和测试集文件
    train_data[['start_entity', 'relation', 'end_entity']].to_csv(f"{output_dir}/train.txt", sep=' ', index=False, header=False, encoding='utf-8')
    test_data[['start_entity', 'relation', 'end_entity']].to_csv(f"{output_dir}/test.txt", sep=' ', index=False, header=False, encoding='utf-8')

    return triples, entity2id, relation2id

def aug_data(input_path, output_dir):
    # 读取数据
    data = pd.read_csv(input_path, sep='\t', header=None, usecols=range(8), encoding='utf-8')
    data.columns = ['id', 'orgin_id', 'start_lang', 'end_lang', 'weight', 'start_entity', 'relation', 'end_entity']
    
    # 过滤start_lang和end_lang为'zh'的行
    data = data[(data['start_lang'] == 'zh') & (data['end_lang'] == 'zh')]
    
    # 计算每个关系和实体的出现次数
    relation_counts = data['relation'].value_counts()
    entity_counts = pd.concat([data['start_entity'], data['end_entity']]).value_counts()

    # 提取三元组和实体、关系
    triples = data[['start_entity', 'relation', 'end_entity']].values
    entities = set(data['start_entity']).union(set(data['end_entity']))
    relations = set(data['relation'])

    # 创建实体和关系的映射
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    # 将entity2id和relation2id写入文件
    with open(f"{output_dir}/entity2id.txt", 'w', encoding='utf-8') as e_file:
        for entity, idx in entity2id.items():
            e_file.write(f"{entity} {idx}\n")

    with open(f"{output_dir}/relation2id.txt", 'w', encoding='utf-8') as r_file:
        for relation, idx in relation2id.items():
            r_file.write(f"{relation} {idx}\n")
     # 复制并增强数据
    augmented_rows = []
    for i in tqdm(range(len(data))):
        row = data.iloc[i]
        # 如果某个实体或关系出现次数较少，则增强该数据
        if entity_counts[row['start_entity']] < 100 or entity_counts[row['end_entity']] < 100 or relation_counts[row['relation']] < 10000:
            augmented_rows.append(row)
            # 对于稀有的实体或关系进行更多增强
            if entity_counts[row['start_entity']] < 10 or entity_counts[row['end_entity']] < 10 or relation_counts[row['relation']] < 100:
                augmented_rows.append(row)
                augmented_rows.append(row)
    
    # 使用 pd.concat 将原始数据与增强数据合并
    augmented_data = pd.concat([data, pd.DataFrame(augmented_rows)], ignore_index=True)

    # 训练集和测试集的分割
    train_data, test_data = train_test_split(augmented_data, test_size=0.18, random_state=42)

    # 写入训练集和测试集文件
    train_data[['start_entity', 'relation', 'end_entity']].to_csv(f"{output_dir}/train.txt", sep=' ', index=False, header=False, encoding='utf-8')
    test_data[['start_entity', 'relation', 'end_entity']].to_csv(f"{output_dir}/test.txt", sep=' ', index=False, header=False, encoding='utf-8')

    return triples, entity2id, relation2id