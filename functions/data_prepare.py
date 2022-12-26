from tika import parser
import re
import pandas as pd
from os import listdir
from os.path import isfile, join, splitext
from tqdm import tqdm

exts = ['.pdf', '.doc', '.docx', '.rtf']


def get_doc_text(file_obj, return_meta=False):
    parsed_file = parser.from_file(file_obj)
    return parsed_file['content'] if not return_meta else parsed_file


def prepare_dataset(contracts_path):
    # Подготовка датасета из текстов договоров
    contract_files = [join(contracts_path, f) for f in listdir(contracts_path)
                      if isfile(join(contracts_path, f)) and splitext(f)[1] in exts]
    print(f'Total documents: {len(contract_files)}')

    docs_data = []

    for doc_file in tqdm(contract_files, desc='DOC/DOCX/PDF/RTF Reading'):
        _, file_ext = splitext(doc_file)
        file_name = doc_file.split('\\')[-1]
        with open(doc_file, 'rb'):
            try:
                doc_text = get_doc_text(doc_file).strip()
            except:
                doc_text = ''
            docs_data.append({'FileName': file_name, 'Format': file_ext[1:], 'Content': doc_text})

    return pd.DataFrame(docs_data)


def doc_sentences_split(txt, min_sym_cnt=50):
    # Разбивка текста договора на предложения
    txt_after = re.sub(r'\sгр\.', ' гражданин ', txt, flags=re.IGNORECASE)
    txt_after = re.sub(r'\sп\.', ' пункт ', txt_after, flags=re.IGNORECASE)
    txt_after = re.sub(r'\sг\.', ' ', txt_after, flags=re.IGNORECASE)
    txt_after = re.sub(r'\sкв\.', ' квадратный ', txt_after, flags=re.IGNORECASE)
    txt_after = re.sub(r'\sм\.п\.', ' место печати ', txt_after, flags=re.IGNORECASE)
    txt_after = re.sub(r'\sм\.', ' метр ', txt_after, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', txt_after, flags=re.M)
    sents = re.split(r'(?<=[.!?…]) ', s)
    
    # Склеиваем короткие строки с последующими длинными (например номер пункта + пункт)
    sents_clipped = []
    st_tmp =''
    for i in range(len(sents)):
        st_tmp += sents[i] + ' '
        # Получили минимально необходимую длину предложения и следующее предложение не начинается с маленькой буквы
        if len(st_tmp) >= min_sym_cnt and ((i == (len(sents) - 1)) or (not sents[i+1][0].islower())):
            sents_clipped.append(st_tmp.strip())
            st_tmp = ''
    if len(st_tmp):
        sents_clipped[-1] = (sents_clipped[-1] + ' ' + st_tmp).strip()
    
    return sents_clipped


def clear_sents(sents_clipped):
    # Очистка списка предложений
    sents_clear = [re.sub(r'\s+|\\n', ' ', sent).lower() for sent in sents_clipped]
    sents_clear_2 = [re.sub(r'[^a-zA-Zа-яА-я]', ' ', sent) for sent in sents_clear]
    sents_clear_3 = [re.sub(r'\s+', ' ', sent).strip() for sent in sents_clear_2]
    return sents_clear_3


def prepare_sents_dataset(docs_df, min_sym_cnt=50, inference=False):
    # Подготовка датасета из очищенных предложений
    sents_data = []

    for doc_row in tqdm(docs_df.iterrows()):
        doc_series = doc_row[1]
        snts = doc_sentences_split(doc_series.Content, min_sym_cnt)
        snts_to_process = clear_sents(snts)
        for i, (snt, snt_cl) in enumerate(zip(snts, snts_to_process)):
            sents_data.append({
                'file_name': doc_series.FileName,
                'snt_order': i + 1,
                'sentence': snt,
                'sentence_clear': snt_cl,
                'data_split': doc_series.Split if not inference else '',
                'target': doc_series.target.split('/')[1]  if not inference else ''
            })
    return pd.DataFrame(sents_data)
