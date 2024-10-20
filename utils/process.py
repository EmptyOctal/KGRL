import pandas as pd
from sklearn.model_selection import train_test_split

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
