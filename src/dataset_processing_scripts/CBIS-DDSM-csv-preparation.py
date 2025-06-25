import os

import pandas as pd


# def main() -> None:
#     """
#     Initial dataset pre-processing for the CBIS-DDSM dataset:
#         * Retrieves the path of all images and filters out the cropped images.
#         * Imports original CSV files with the full image information (patient_id, left or right breast, pathology,
#         image file path, etc.).
#         * Filters out the cases with more than one pathology in the original csv files.
#         * Merges image path which extracted on GPU machine and image pathology which is in the original csv file on image id
#         * Outputs 4 CSV files.

#     Generate CSV file columns:
#       img: image id (e.g Calc-Test_P_00038_LEFT_CC => <case type>_<patient id>_<left or right breast>_<CC or MLO>)
#       img_path: image path on the GPU machine
#       label: image pathology (BENIGN or MALIGNANT)
    
#     Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
    
#     :return: None
#     """
#     csv_root = '/data/CBIS-DDSM/manifest-ZkhPvrLo5216730872708713142'               # original csv folder (change as needed)
#     img_root = '/data/CBIS-DDSM/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM'     # dataset folder (change as needed)
#     csv_output_path = '../data/CBIS-DDSM'                                           # csv output folder (change as needed)

#     folders = os.listdir(img_root)

#     cases_dict = dict()  # save image id and path

#     for f in folders:
#         if f.endswith('_CC') or f.endswith('_MLO'):  # filter out the cropped images
#             path = list()

#             for root, dirs, files in os.walk(img_root + '/' + f):  # retrieve the path of image
#                 for d in dirs:
#                     path.append(d)
#                 for filename in files:
#                     path.append(filename)

#             img_path = img_root + '/' + f + '/' + '/'.join(path)  # generate image path
#             cases_dict[f] = img_path

#     df = pd.DataFrame(list(cases_dict.items()), columns=['img', 'img_path'])  # transform image dictionary to dataframe

#     # image type and csv file name mapping
#     type_dict = {'Calc-Test': 'calc_case_description_test_set.csv',
#                  'Calc-Training': 'calc_case_description_train_set.csv',
#                  'Mass-Test': 'mass_case_description_test_set.csv',
#                  'Mass-Training': 'mass_case_description_train_set.csv'}

#     for t in type_dict.keys():  # handle images based on the type
#         df_subset = df[df['img'].str.startswith(t)]

#         df_csv = pd.read_csv(csv_root + '/' + type_dict[t],
#                              usecols=['pathology', 'image file path'])  # read original csv file
#         df_csv['img'] = df_csv.apply(lambda row: row['image file path'].split('/')[0],
#                                      axis=1)  # extract image id from the path
#         df_csv['pathology'] = df_csv.apply(
#             lambda row: 'BENIGN' if row['pathology'] == 'BENIGN_WITHOUT_CALLBACK' else row['pathology'],
#             axis=1)  # replace pathology 'BENIGN_WITHOUT_CALLBACK' to 'BENIGN'

#         df_cnt = pd.DataFrame(
#             df_csv.groupby(['img'])['pathology'].nunique()).reset_index()  # count dictict pathology for each image id
#         df_csv = df_csv[~df_csv['img'].isin(
#             list(df_cnt[df_cnt['pathology'] != 1]['img']))]  # filter out the image with more than one pathology
#         df_csv = df_csv.drop(columns=['image file path'])
#         df_csv = df_csv.drop_duplicates(
#             keep='first')  # remove duplicate data (because original csv list all abnormality area, that make one image id may have more than one record)

#         df_subset_new = pd.merge(df_subset, df_csv, how='inner',
#                                  on='img')  # merge image path and image pathology on image id
#         df_subset_new = df_subset_new.rename(columns={'pathology': 'label'})  # rename column 'pathology' to 'label'
#         df_subset_new.to_csv(csv_output_path + '/' + t.lower() + '.csv',
#                              index=False)  # output merged dataframe in csv format

#         print(t)
#         print('data_cnt: %d' % len(df_subset_new))
#         print('multi pathology case:')
#         print(list(df_cnt[df_cnt['pathology'] != 1]['img']))
#         print()

#     print('Finished pre-processing CSV for the CBIS-DDSM dataset.')


# if __name__ == '__main__':
#     main()
import os
import pandas as pd

# def main() -> None:
#     """
#     Script tiền xử lý cho bộ dữ liệu CBIS-DDSM (đã điều chỉnh cho cấu trúc thư mục mới).
#     - Đọc các file CSV gốc chứa thông tin bệnh lý.
#     - Tạo một DataFrame ánh xạ từ patient_id và các thông tin khác tới đường dẫn ảnh jpeg thực tế.
#     - Hợp nhất hai nguồn thông tin trên để tạo ra các file CSV cuối cùng cho việc huấn luyện và kiểm thử.
#     """
#     # --- THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI MÁY CỦA BẠN ---
#     base_path = '/home/neeyuhuynh/Desktop/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication'
    
#     # Đường dẫn đến thư mục chứa các file CSV gốc (mass_case_description_... và calc_case_description_...)
#     csv_root = os.path.join(base_path, 'data/CBIS-DDSM/csv')
    
#     # Đường dẫn đến thư mục chứa các ảnh đã chuyển đổi sang jpeg
#     img_root = os.path.join(base_path, 'data/CBIS-DDSM/jpeg')
    
#     # Đường dẫn để lưu các file CSV đầu ra (sẽ được tạo nếu chưa có)
#     csv_output_path = os.path.join(base_path, 'data/CBIS-DDSM/processed_csv')
#     if not os.path.exists(csv_output_path):
#         os.makedirs(csv_output_path)
#     # --- KẾT THÚC PHẦN THAY ĐỔI ---

#     # Đọc file dicom_info.csv để ánh xạ thông tin bệnh nhân tới đường dẫn ảnh
#     # Giả sử file này chứa thông tin liên kết giữa mô tả và ảnh thực tế
#     try:
#         dicom_info_df = pd.read_csv(os.path.join(csv_root, 'dicom_info.csv'))
#         # Tạo cột 'image file path' để khớp với các file CSV khác
#         dicom_info_df['image file path'] = dicom_info_df['image_path'].apply(
#             lambda x: x.replace('CBIS-DDSM/', '').replace('.dcm', '')
#         )
#         # Tạo cột 'full_jpeg_path'
#         dicom_info_df['full_jpeg_path'] = dicom_info_df['image_path'].apply(
#             lambda x: os.path.join(img_root, x.replace('.dcm', '.jpg'))
#         )
#     except FileNotFoundError:
#         print(f"Lỗi: Không tìm thấy file 'dicom_info.csv' tại '{csv_root}'. File này rất quan trọng để liên kết dữ liệu.")
#         return

#     # Từ điển ánh xạ loại dữ liệu và tên file CSV
#     type_dict = {'Calc-Test': 'calc_case_description_test_set.csv',
#                  'Calc-Training': 'calc_case_description_train_set.csv',
#                  'Mass-Test': 'mass_case_description_test_set.csv',
#                  'Mass-Training': 'mass_case_description_train_set.csv'}

#     for t, csv_file in type_dict.items():
#         print(f"Đang xử lý: {t}")
#         try:
#             df_csv = pd.read_csv(os.path.join(csv_root, csv_file))
#         except FileNotFoundError:
#             print(f"Cảnh báo: Không tìm thấy file {csv_file}. Bỏ qua...")
#             continue

#         # Chuẩn hóa các giá trị bệnh lý
#         df_csv['pathology'] = df_csv['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')

#         # Lọc ra các trường hợp có nhiều hơn một bệnh lý
#         df_cnt = df_csv.groupby('image file path')['pathology'].nunique().reset_index()
#         multi_pathology_cases = df_cnt[df_cnt['pathology'] != 1]['image file path']
#         df_csv = df_csv[~df_csv['image file path'].isin(multi_pathology_cases)]
        
#         # Xóa các bản ghi trùng lặp
#         df_csv = df_csv.drop_duplicates(subset=['image file path'], keep='first')

#         # Hợp nhất với thông tin đường dẫn ảnh đầy đủ
#         # Chúng ta cần một cột chung để hợp nhất, ở đây là 'image file path'
#         df_merged = pd.merge(df_csv, dicom_info_df[['image file path', 'full_jpeg_path']], 
#                              how='inner', on='image file path')

#         if df_merged.empty:
#             print(f"Cảnh báo: Không có dữ liệu nào được hợp nhất cho {t}. Kiểm tra lại nội dung các file CSV.")
#             continue
            
#         # Tạo cột 'label' và chỉ giữ lại các cột cần thiết
#         final_df = df_merged[['full_jpeg_path', 'pathology']].copy()
#         final_df.rename(columns={'full_jpeg_path': 'img_path', 'pathology': 'label'}, inplace=True)

#         # Lưu file CSV đầu ra
#         output_filename = os.path.join(csv_output_path, t.lower() + '.csv')
#         final_df.to_csv(output_filename, index=False)

#         print(f'  -> Đã tạo file: {output_filename}')
#         print(f'  -> Số lượng mẫu: {len(final_df)}')
#         if not multi_pathology_cases.empty:
#             print(f'  -> Các trường hợp có nhiều bệnh lý đã bị loại bỏ: {len(multi_pathology_cases)}')
#         print()

#     print('Hoàn tất tiền xử lý CSV cho bộ dữ liệu CBIS-DDSM.')


# if __name__ == '__main__':
#     main()
# --- Cấu hình đường dẫn ---
# Để đơn giản, hãy đảm bảo TẤT CẢ các tệp CSV của bạn (mass, calc, meta, dicom_info)
# đều nằm trong cùng một thư mục này.
BASE_PATH = "/home/neeyuhuynh/Desktop/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication"
CSV_FILES_DIRECTORY = os.path.join(BASE_PATH, "data/CBIS-DDSM/csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "data/CBIS-DDSM")
try:
    # --- Bước 1: Tạo bảng tra cứu chỉ từ dicom_info.csv ---
    print("1. Đang tạo bảng tra cứu từ dicom_info.csv...")
    dicom_info_df = pd.read_csv(os.path.join(CSV_FILES_DIRECTORY, 'dicom_info.csv'))
    
    # Tạo đường dẫn ảnh tuyệt đối từ cột 'image_path'
    dicom_info_df['full_image_path'] = dicom_info_df['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))
    
    # Bảng tra cứu chỉ cần 2 cột: UID và đường dẫn ảnh
    lookup_table = dicom_info_df[['SeriesInstanceUID', 'full_image_path']]
    lookup_table.drop_duplicates(subset=['SeriesInstanceUID'], inplace=True)
    
    print(f"   -> Bảng tra cứu được tạo thành công với {len(lookup_table)} UID duy nhất.")

    # --- Bước 2: Hàm xử lý cho từng bộ dữ liệu (training/testing) ---
    def process_and_save(mass_csv_filename, calc_csv_filename, dataset_name):
        print(f"\n2. Bắt đầu xử lý bộ dữ liệu '{dataset_name}'...")
        
        mass_df = pd.read_csv(os.path.join(CSV_FILES_DIRECTORY, mass_csv_filename))
        calc_df = pd.read_csv(os.path.join(CSV_FILES_DIRECTORY, calc_csv_filename))
        df_combined = pd.concat([mass_df, calc_df], ignore_index=True)
        
        # Trích xuất SeriesInstanceUID từ cột 'cropped image file path'
        df_combined['SeriesInstanceUID'] = df_combined['cropped image file path'].apply(lambda x: x.split('/')[2])
        
        print(f"   -> Tổng số ca '{dataset_name}' ban đầu: {len(df_combined)}")

        # Hợp nhất chỉ dựa trên 'SeriesInstanceUID'
        final_df = pd.merge(df_combined, lookup_table, on='SeriesInstanceUID', how='inner')
        
        print(f"   -> Số lượng mẫu khớp sau khi hợp nhất: {len(final_df)}")

        # Chọn và đổi tên các cột cuối cùng
        result_df = final_df[['patient_id', 'pathology', 'full_image_path']].copy()
        result_df.rename(columns={'full_image_path': 'image_file_path'}, inplace=True)
        
        result_df.drop_duplicates(inplace=True)

        output_filepath = os.path.join(OUTPUT_PATH, f'{dataset_name}.csv')
        result_df.to_csv(output_filepath, index=False)
        
        print(f"   -> Đã tạo thành công tệp: '{dataset_name}.csv' với {len(result_df)} mẫu.")
        if len(result_df) == 0:
            print(f"   CẢNH BÁO: Tệp '{dataset_name}.csv' rỗng.")
            
    # --- Chạy quá trình cho training và testing ---
    process_and_save(
        'mass_case_description_train_set.csv',
        'calc_case_description_train_set.csv',
        'training'
    )
    
    process_and_save(
        'mass_case_description_test_set.csv',
        'calc_case_description_test_set.csv',
        'testing'
    )

    print("\n--- Hoàn tất quá trình tiền xử lý ---")

except Exception as e:
    print(f"\n!!! Đã xảy ra lỗi không mong muốn: {e}")



# Calc-Test
# data_cnt: 282
# multi pathology case:
# ['Calc-Test_P_00353_LEFT_CC', 'Calc-Test_P_00353_LEFT_MLO']

# Calc-Training
# data_cnt: 1220
# multi pathology case:
# ['Calc-Training_P_00600_LEFT_CC', 'Calc-Training_P_00600_LEFT_MLO', 'Calc-Training_P_00937_RIGHT_CC', 'Calc-Training_P_00937_RIGHT_MLO', 'Calc-Training_P_01284_RIGHT_MLO', 'Calc-Training_P_01819_LEFT_CC', 'Calc-Training_P_01819_LEFT_MLO']

# Mass-Test
# data_cnt: 359
# multi pathology case:
# ['Mass-Test_P_00969_LEFT_CC', 'Mass-Test_P_00969_LEFT_MLO']

# Mass-Training
# data_cnt: 1225
# multi pathology case:
# ['Mass-Training_P_00419_LEFT_CC', 'Mass-Training_P_00419_LEFT_MLO', 'Mass-Training_P_00797_LEFT_CC', 'Mass-Training_P_00797_LEFT_MLO', 'Mass-Training_P_01103_RIGHT_CC', 'Mass-Training_P_01103_RIGHT_MLO']
