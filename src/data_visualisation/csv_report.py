import pandas as pd
from sklearn.metrics import classification_report

import config
import os

# Tự động tạo thư mục ../output nếu chưa tồn tại
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

def generate_csv_report(y_true_inv, y_pred_inv, label_encoder, accuracy) -> None:
    # """
    # Print and save classification report for accuracy, precision, recall and f1 score metrics.
    # :return: None.
    # """

    # # Classification report.
    # print(classification_report(y_true_inv, y_pred_inv, target_names=label_encoder.classes_))
    # report_df = pd.DataFrame(classification_report(y_true_inv, y_pred_inv, target_names=label_encoder.classes_,
    #                                                output_dict=True)).transpose()
    # 1) Tập các nhãn chuỗi xuất hiện trong y_true_inv
    # 1) Lấy tập nhãn chuỗi xuất hiện
    unique_labels = sorted(set(y_true_inv))

    # 2) target_names chính là unique_labels theo thứ tự của label_encoder
    target_names = [lbl for lbl in label_encoder.classes_ if lbl in unique_labels]

    # 3) Dùng labels = target_names (chuỗi) khi y_true_inv là chuỗi
    labels = target_names

    # 4) In classification report
    report_str = classification_report(
        y_true_inv,
        y_pred_inv,
        labels=labels,
        target_names=target_names
    )
    print(report_str)

    # 5) DataFrame từ báo cáo
    report_df = pd.DataFrame(
        classification_report(
            y_true_inv,
            y_pred_inv,
            labels=labels,
            target_names=target_names,
            output_dict=True
        )
    ).transpose()

    # 6) Thêm hàng accuracy
    # Append accuracy.
    # report_df.append({'accuracy': accuracy}, ignore_index=True)
    # Append accuracy as a new row (using concat thay thế append)
    new_row = pd.DataFrame([{'accuracy': accuracy}])
    report_df = pd.concat([report_df, new_row], ignore_index=True)

# Save report.
    report_df.to_csv(
        os.path.join(
            output_dir,
            "{}_dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_report.csv".format(
                config.run_mode,
                config.dataset,
                config.mammogram_type,
                config.model,
                config.learning_rate,
                config.batch_size,
                config.max_epoch_frozen,
                config.max_epoch_unfrozen,
                config.is_roi
            )
        ),
        index=False,
        header=True
)



def generate_csv_metadata(runtime) -> None:
    """
    Print and save CLI arguments and training runtime.
    :return: None.
    """
    metadata_dict = {
        'dataset': config.dataset,
        'mammogram_type': config.mammogram_type,
        'model': config.model,
        'run_mode': config.run_mode,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'max_epoch_frozen': config.max_epoch_frozen,
        'max_epoch_unfrozen': config.max_epoch_unfrozen,
        'is_roi': config.is_roi,
        'experiment_name': config.name,
        'training runtime (s)': runtime
    }

    # Convert to dataframe.
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index')

    # Save report.
    metadata_df.to_csv(
        os.path.join(
            output_dir,
            "{}_dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_metadata.csv".format(
                config.run_mode,
                config.dataset,
                config.mammogram_type,
                config.model,
                config.learning_rate,
                config.batch_size,
                config.max_epoch_frozen,
                config.max_epoch_unfrozen,
                config.is_roi
            )
        ),
        index=False,
        header=True
    )
