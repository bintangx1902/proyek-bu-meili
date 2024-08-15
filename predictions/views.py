import base64

from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse
import time
import seaborn as sns
from .forms import *
# from .utils import *
from .new_utils import *
from io import BytesIO
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# df = read_data_train()
# df_status, df_mhs = pivots(df)
# df_mhs = table_melting(df_mhs)
# G, graphs_nim, graph_embedding, nim_list, i = create_graph(df_mhs)
# train_graphs1, train_graphs, test_graphs1, test_graphs = split_data(i, df, graphs_nim, graph_embedding)
# comparison_table = get_ged_score(train_graphs1, train_graphs, test_graphs1, test_graphs)
# df_merged = merging_table(comparison_table, df_status)
# model = SVC()
# x, y = df_merged['GED'], df_merged['STATUS']
# x = np.reshape(x, (-1, 1))
# model.fit(x, y)


df_gbg = read_data_train('2020.xlsx')
df_status, df_mhs = pivots(df_gbg)
df_mhs = melting_table(df_mhs)
G, graphs_nim, graphs_embedding, nimList, i = create_graphs(df_mhs)
train_graphs1, test_graphs1 = split_data(i, df_gbg, graphs_nim)
comparison_table = get_ged_score(train_graphs1, test_graphs1)
df = merge_table(comparison_table, df_status)
x, y = xy_split(df)
svm = SVC()
svm.fit(x, y)


def main_menu(request):
    form = FileForms()
    if request.method == 'POST':
        form = FileForms(request.POST, request.FILES)
        if request.FILES.get('file'):
            to_shown = int(request.POST.get('what'))
            file = request.FILES.get('file')
            df_year = pd.read_excel(file)
            start_time = time.time()
            # start here
            df_status_test, df_mhs_test = pivots(df_year)
            df_mhs_test = melting_table(df_mhs_test)
            G_test, graphs_nim_test, graphs_embedding_test, nim_list_test, i_test = create_graphs(df_mhs_test)
            train_graphs_test, test_graphs_test = split_data(i_test, df_year, graphs_nim_test)
            comparison_table_test = get_ged_score(train_graphs_test, test_graphs_test)
            compare_result_test = comparison_table_test.merge(df_status_test, how='left', left_on='GRAPH_TRAIN',
                                                              right_on='NIM')
            min_ged = compare_result_test.groupby('GRAPH_TRAIN')['GED'].min().reset_index()
            merged_df = comparison_table_test.merge(min_ged, on='GRAPH_TRAIN', suffixes=('', '_min'))

            averages = []
            for value in merged_df['GRAPH_TRAIN'].unique():
                average_value = merged_df[merged_df['GRAPH_TRAIN'] == value]['GED'].min()
                averages.append({'GRAPH_TRAIN': value, 'GED': average_value})

            for value in merged_df['GRAPH_TEST'].unique():
                average_value = merged_df[merged_df['GRAPH_TEST'] == value]['GED'].min()
                averages.append({'GRAPH_TRAIN': value, 'GED': average_value})

            df_test = pd.DataFrame(averages)

            df_test['STATUS'] = df_status_test['STATUS']
            df_test['STATUS'] = df_test['STATUS'].map(lambda s: 1 if s == 'LULUS' else 0)

            x_test, y_test = df_test['GED'], df_test['STATUS']
            x_test = np.reshape(x_test, (-1, 1))

            pred = svm.predict(x_test)
            print(len(pred))
            df_print = pd.DataFrame({
                'NIM': df_test['GRAPH_TRAIN'].to_numpy(),
                'PREDICTION': pred
            })

            df_print['PREDICTION'] = df_print['PREDICTION'].apply(mapping)

            excel_file = BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                df_print.to_excel(writer, index=False, sheet_name='Sheet1')
            end_time = time.time()

            if to_shown == 1:
                count = df_print['PREDICTION'].value_counts().reset_index()
                sns.barplot(x='PREDICTION', y='count', data=count)
                plt.xlabel('Prediksi Lulus')
                plt.ylabel('Count')
                plt.title('Count of Lulus vs Drop-Out')

                for index, row in count.iterrows():
                    plt.text(index, row['count'], row['count'], color='black', ha="center")

                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()

                context = {
                    'chart': image_base64,
                    'form': form
                }
                messages.success(request, f"Lama Waktu prediksi : {end_time - start_time}")
                return render(request, 'index.html', context)
            else:
                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                excel_file.seek(0)
                response = HttpResponse(excel_file.read(), content_type=content_type)
                response['Content-Disposition'] = 'attachment; filename="Hasil-Prediksi.xlsx"'
                return response

    context = {'form': form}
    return render(request, 'index.html', context)


def download_file(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'template.xlsx')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fp:
            response = HttpResponse(fp.read(), content_type='application/file')
            response['Content-Disposition'] = f"inline; filename={os.path.basename(file_path)}"
            return response
    raise Http404
