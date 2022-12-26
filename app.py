import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from os.path import splitext
import seaborn as sns
import matplotlib.pyplot as plt
import json
import datetime
from ast import literal_eval

from functions.utils import get_file, save_file
from functions.data_prepare import get_doc_text, prepare_sents_dataset
from functions.bert import Dataset, BertClassifier, inference_model, get_similarity_level
from functions.consts import MODEL_PATH, DATA_PATH, OUTPUT_PATH, \
    MODEL_DICT_FN, TOKENIZER_FN, IDX2ITEM_FN, TARGET2EMB_FN, RESP_FN, HISTORY_FN, \
    PROB_THR, POSS_TYPE_THR, N_PHRASES, OUTP_DIM, BATCH_SIZE, N_CATS, MAX_LEN


def get_doc_info(file_obj, model, tokenizer, item2idx, target2emb):
    # Определение типа документа и возврат аналитики по нему
    doc_text = get_doc_text(file_obj).strip()
    _, file_ext = splitext(file_obj.name)
    docs_data = pd.DataFrame([{'FileName': file_obj.name, 'Format': file_ext[1:], 'Content': doc_text}])
    data = prepare_sents_dataset(docs_data, inference=True)
    test_data = Dataset(data[['sentence_clear', 'file_name', 'snt_order', 'sentence']], tokenizer, MAX_LEN, item2idx)
    data['target_pred'], data['pred_proba'] = inference_model(model, test_data, BATCH_SIZE, return_proba=True)
    data['target_pred'] = data['target_pred'].map({v: k for k, v in item2idx.items()})

    # Возможный тип и его вероятность
    ret = data.target_pred.value_counts(normalize=True).to_frame().reset_index()
    ret.columns = ['Тип договора', 'Вероятность']

    # Степень схожести с документами данного типа
    sim_level = get_similarity_level(test_data, model, target2emb[ret['Тип договора'].values[0]], BATCH_SIZE)

    # Ключевые фрагменты для каждого возможного типа
    data.sort_values(by='pred_proba', ascending=False, inplace=True)
    possible_types = ret[ret['Вероятность'] >= POSS_TYPE_THR]['Тип договора'].values
    imp_phrases = {}
    for doc_type in possible_types:
        tmp_df = data[data.target_pred == doc_type].head(N_PHRASES)
        imp_phrases[doc_type] = [f'{sent} ({prob :.1%})' for sent, prob in
                                 zip(tmp_df.sentence.values, tmp_df.pred_proba.values)]
        # imp_phrases[doc_type] = tmp_df.sentence.values

    return ret.round(3), imp_phrases, sim_level


@st.cache(allow_output_mutation=True)
def load_model():
    bert_model = AutoModel.from_pretrained('cointegrated/rubert-tiny2')
    model = BertClassifier(bert_model, N_CATS, OUTP_DIM)
    model.load_state_dict(torch.load(MODEL_PATH / MODEL_DICT_FN, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = get_file(MODEL_PATH / TOKENIZER_FN)
    idx2item = get_file(MODEL_PATH / IDX2ITEM_FN)
    target2emb = get_file(MODEL_PATH / TARGET2EMB_FN)
    item2idx = {v: k for k, v in idx2item.items()}
    resp_df = pd.read_csv(DATA_PATH / RESP_FN, sep=';')
    return model, tokenizer, item2idx, resp_df, target2emb


def doc_check():
    model, tokenizer, item2idx, resp_df, target2emb = load_model()
    st.title('Определение вида договора')
    with st.form('Загрузка договора'):
        file_obj = st.file_uploader('Выберите файл с договором (*.doc, *.docx, *.pdf)',
                                    type=['doc', 'docx', 'pdf'],
                                    accept_multiple_files=False)
        submitted = st.form_submit_button("Определить тип договора")

    if submitted and file_obj is not None:
        type_vc, imp_phrases, sim_level = get_doc_info(file_obj, model, tokenizer, item2idx, target2emb)

        doc_class = type_vc['Тип договора'][0]
        pred_proba = type_vc['Вероятность'][0]

        st.subheader(f'Тип договора:')
        st.markdown(f'**{doc_class}**')
        st.markdown(f'Уверенность: {pred_proba :.1%}')
        st.markdown(f'Степень схожести с другими договорами этого типа: {sim_level :.1%}')

        fig = plt.figure(figsize=(8, 2))
        ax = sns.barplot(data=type_vc, y='Тип договора', x='Вероятность', orient='h')
        ax.set(xlabel=None, ylabel=None)
        ax.set(xlim=(0.0, 1.05))
        for val in ax.containers:
            ax.bar_label(val)
        st.pyplot(fig)

        col1, col2, = st.columns(2)

        with col1:
            if pred_proba > PROB_THR:
                st.markdown('**Высокая** уверенность модели в ответе.')
                st.markdown('Рекомендация: отправить ответственному.')
                if st.button('Отправить ответственному'):
                    st.write('Договор отправлен')
            else:
                st.markdown('**Низкая уверенность модели в ответе!**')
                st.markdown('**Рекомендация: отправить на ручную проверку.**')
                if st.button('Отправить на ручную проверку'):
                    st.write('Договор отправлен')

        with col2:
            if pred_proba > PROB_THR:
                resp = resp_df[resp_df['Тип договора'] == doc_class]
                in_charge = resp['Ответственный'].values[0]
                st.text(f'Ответственный: {in_charge}')
                st.text(resp['email'].values[0])
                st.text(resp['Телефон'].values[0])

            else:
                lawyer = resp_df[resp_df['Тип договора'] == 'не определено']
                in_charge = lawyer['Ответственный'].values[0]
                st.text(f'Юрист: {in_charge}')
                st.text(lawyer['email'].values[0])
                st.text(lawyer['Телефон'].values[0])

        st.subheader('Ключевые фрагменты:')
        for doc_type in imp_phrases:
            type_proba = type_vc[type_vc['Тип договора'] == doc_type]['Вероятность'].values[0]
            st.markdown(f'**{doc_type}:** ({type_proba :.1%})')
            for phrase in imp_phrases[doc_type]:
                st.caption(phrase)

        json_result = dict()

        json_result['doc_id'] = file_obj.name
        json_result['class'] = type_vc['Тип договора'].to_list()
        json_result['confidence_level'] = type_vc['Вероятность'].to_list()
        json_result['interpretation'] = imp_phrases[doc_class]
        json_result['similarity'] = sim_level

        json_fname = str(file_obj.name).replace('.', '_')
        with open(OUTPUT_PATH / f'{json_fname}.json', 'w') as outfile:
            outfile.write(json.dumps(json_result))

        json_result['date_time'] = datetime.datetime.now()
        # history = pd.DataFrame([json_result])
        history = pd.read_csv(OUTPUT_PATH / HISTORY_FN)
        history = pd.concat((history, pd.DataFrame([json_result])))
        history.to_csv(OUTPUT_PATH / HISTORY_FN, index=False)


def docs_list():
    st.title("Список проверенных договоров")
    history = pd.read_csv(OUTPUT_PATH / HISTORY_FN).sort_values(by='date_time', ascending=False)

    history['Дата'] = pd.to_datetime(history.date_time).dt.strftime("%Y-%m-%d")
    history['Время'] = pd.to_datetime(history.date_time).dt.strftime("%H:%M:%S")
    history['Тип договора'] = history['class'].map(lambda x: literal_eval(x)[0])
    history.rename(columns={'doc_id': 'Имя файла'}, inplace=True)
    output_cols = ['Дата', 'Время', 'Имя файла', 'Тип договора']

    gb = GridOptionsBuilder.from_dataframe(history[output_cols])
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode='single')
    gridOptions = gb.build()

    grid_response = AgGrid(
        history,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        height=360,
        width='100%',
        reload_data=True,
    )

    selected = grid_response['selected_rows']
    if selected:
        df = pd.DataFrame(selected).iloc[0, 1:]

        st.caption(f'{df["Дата"]} {df["Время"]}')
        st.markdown(f'**{df["Имя файла"]} : {literal_eval(df["class"])[0]}**')
        st.markdown(f'Уверенность: {literal_eval(df["confidence_level"])[0] :.1%}')
        st.markdown(f'Степень схожести с другими договорами этого типа: {df["similarity"]:.1%}')

        fig = plt.figure(figsize=(8, 2))
        ax = sns.barplot(y=literal_eval(df["class"]), x=literal_eval(df["confidence_level"]), orient='h')
        ax.set(xlabel=None, ylabel=None)
        ax.set(xlim=(0.0, 1.05))
        for val in ax.containers:
            ax.bar_label(val)
        st.pyplot(fig)

        st.markdown('**Ключевые фрагменты:**')
        for phrase in literal_eval(df["interpretation"]):
            st.caption(phrase)


if __name__ == "__main__":

    with st.sidebar:
        page = option_menu('', ['Проверить договор', 'История проверок'],
                           icons=['upload', 'list-check'], default_index=0,
                           styles={"nav-link-selected": {"font-weight": "400"}}
                           )
    if page == 'Проверить договор':
        doc_check()
    else:
        docs_list()
