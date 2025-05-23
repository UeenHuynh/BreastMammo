# src/main.py

import os
import sys
import time
import argparse

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from data_operations.data_preprocessing import (
    import_inbreast_roi_dataset,
    import_inbreast_full_dataset
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# allow imports from project root
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT_BREAST = '/kaggle/input/breastdata'
DATA_ROOT_CMMD = '/kaggle/input/cmmddata/CMMD'
from collections import Counter
import config
from data_operations.data_transformations import generate_image_transforms
from data_operations import data_preprocessing, data_transformations, dataset_feed
from cnn_models.cnn_model import CnnModel
from tensorflow.keras import layers, models, applications, optimizers

def build_cnn(input_shape, num_classes):
    """Simple grayscale CNN."""
    m = models.Sequential(name="CustomCNN")
    m.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    m.add(layers.MaxPooling2D((2,2)))
    m.add(layers.Conv2D(64, (3,3), activation='relu'))
    m.add(layers.MaxPooling2D((2,2)))
    m.add(layers.Flatten())
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(64, activation='relu'))
    if num_classes == 2:
        m.add(layers.Dense(1, activation='sigmoid'))
    else:
        m.add(layers.Dense(num_classes, activation='softmax'))
    return m

def make_class_weights_from_labels(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

def build_pretrained_model(model_name, input_shape, num_classes):
    """Pretrained ImageNet base + GAP→Dropout→Dense."""
    mn = model_name.lower()
    if mn.startswith("vgg"):
        base = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "resnet":
        base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "inception":
        base = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "densenet":
        base = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "mobilenet":
        base = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(x)
    else:
        out = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=base.input, outputs=out, name=model_name), base

def main():
    # 1) CLI args
    parser = argparse.ArgumentParser(description="Mammogram DL pipeline")
    parser.add_argument("-d", "--dataset",
                        choices=["mini-MIAS","mini-MIAS-binary","CBIS-DDSM","CMMD","INbreast"],
                        required=True,
                        help="Dataset to use")
    parser.add_argument("-mt", "--mammogram_type",
                        choices=["calc","mass","all"], default="all",
                        help="For CBIS-DDSM only")
    parser.add_argument("-m", "--model",
                        choices=["CNN","VGG","VGG-common","ResNet","Inception","DenseNet","MobileNet"],
                        required=True,
                        help="Model backbone")
    parser.add_argument("-r", "--runmode",
                        choices=["train","test"], default="train",
                        help="train or test")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=config.learning_rate, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=config.batch_size, help="Batch size")
    parser.add_argument("-e1", "--max_epoch_frozen", type=int,
                        default=config.max_epoch_frozen, help="Frozen epochs")
    parser.add_argument("-e2", "--max_epoch_unfrozen", type=int,
                        default=config.max_epoch_unfrozen, help="Unfrozen epochs")
    parser.add_argument("--roi", action="store_true",
                        help="Use ROI mode for INbreast / mini-MIAS")
    parser.add_argument("--augment", action="store_true",
                        help="Apply augmentation transforms")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("-n", "--name", default=config.name,
                        help="Experiment name")
    args = parser.parse_args()

    # 2) Override config from args
    config.dataset            = args.dataset
    config.mammogram_type     = args.mammogram_type
    config.model              = args.model
    config.run_mode           = args.runmode
    config.learning_rate      = args.learning_rate
    config.batch_size         = args.batch_size
    config.max_epoch_frozen   = args.max_epoch_frozen
    config.max_epoch_unfrozen = args.max_epoch_unfrozen
    config.is_roi             = args.roi
    config.augment_data       = args.augment
    config.verbose_mode       = args.verbose
    config.name               = args.name
    
    if config.verbose_mode:
        print(f"[DEBUG] Config: dataset={config.dataset}, model={config.model}, roi={config.is_roi}, augment={config.augment_data}")

    # 3) Load & preprocess data
    # le = LabelEncoder()
    # X_train = X_test = y_train = y_test = None

    # if config.dataset in ["mini-MIAS","mini-MIAS-binary"]:
    #     d = os.path.join("/kaggle/input/breastdata", config.dataset)
    #     X, y = data_preprocessing.import_minimias_dataset(d, le)
    #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    # elif config.dataset == "CBIS-DDSM":
    #     X_train, y_train = data_preprocessing.import_cbisddsm_training_dataset(le)
    #     X_test,  y_test  = data_preprocessing.import_cbisddsm_testing_dataset(le)

    # elif config.dataset == "CMMD":
    #     d = os.path.join("/kaggle/input/breastdata", "CMMD-binary")
    #     # if you reverted to the original 3-arg loader:
    #     #     X, y = data_preprocessing.import_cmmd_dataset(d, csv_path, le)
    #     X, y = data_preprocessing.import_cmmd_dataset(d, le)
    #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    # # elif config.dataset == "INbreast":
    # #     d = os.path.join("/kaggle/input/breastdata", "INbreast")
    # #     X, y = data_preprocessing.import_inbreast_dataset(d, le)
    # #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)
    # elif config.dataset.upper() == "INBREAST":
    #     data_dir = os.path.join("/kaggle/input/breastdata","INbreast")
    #     X, y = data_preprocessing.import_inbreast_dataset(data_dir, le)
    #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)
    #     if getattr(config, "augment_data", False):
    #         X_train, y_train = data_transformations.generate_image_transforms(X_train, y_train)

    # else:
    #     raise ValueError(f"Unsupported dataset: {config.dataset}")

    le = LabelEncoder()
    X_train = X_test = y_train = y_test = None
    train_data = val_data = None

    if config.dataset in ["mini-MIAS", "mini-MIAS-binary"]:
        d = os.path.join(DATA_ROOT_BREAST, config.dataset)
        X, y = data_preprocessing.import_minimias_dataset(d, le)
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    elif config.dataset == "CBIS-DDSM":
        X_train, y_train = data_preprocessing.import_cbisddsm_training_dataset(le)
        X_test, y_test = data_preprocessing.import_cbisddsm_testing_dataset(le)

    elif config.dataset == "CMMD":
        # Nếu muốn dùng CMMD-binary:
        # d = os.path.join(DATA_ROOT, "CMMD-binary")
        # Nếu muốn dùng CMMD gốc (ảnh và clinical):
        # d = os.path.join(DATA_ROOT, "CMMD", "CMMD")
        d = DATA_ROOT_CMMD
        X, y = data_preprocessing.import_cmmd_dataset(d, le)
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)
        # --- Gán train/val và input_shape, num_classes ---
        train_data = (X_train, y_train)
        val_data   = (X_test,  y_test)
        input_shape = X_train.shape[1:]
        # nếu y_train one-hot thì ndim>1, ngược lại binary (ndim==1)
        num_classes = y_train.shape[1] if y_train.ndim > 1 else 2
    # elif config.dataset.upper() == "INBREAST":
    #     data_dir = os.path.join(DATA_ROOT_BREAST, "INbreast", "INbreast")
    #     if config.is_roi:
    #         # --- ROI‐mode: tf.data.Dataset on‐the‐fly ---
    #         ds = import_inbreast_roi_dataset(
    #             data_dir, le,
    #             target_size=(config.INBREAST_IMG_SIZE["HEIGHT"],
    #                          config.INBREAST_IMG_SIZE["WIDTH"])
    #         )
    #         # Shuffle + split
    #         ds = ds.shuffle(buffer_size=1000)
    #         split = int(0.8 * 1000)
    #         ds_train = ds.take(split).batch(config.batch_size)
    #         ds_val   = ds.skip(split).batch(config.batch_size)

    #         train_data, val_data = ds_train, ds_val
    #         # Lấy input_shape từ element_spec của Dataset
    #         input_shape = train_data.element_spec[0].shape[1:]
    #         # Số lớp bằng số classes của LabelEncoder
    #         num_classes = le.classes_.size

    #     else:
    #         # --- Full‐image mode: load all into numpy ---
    #         X, y = import_inbreast_full_dataset(
    #             data_dir, le,
    #             target_size=(config.INBREAST_IMG_SIZE["HEIGHT"],
    #                          config.INBREAST_IMG_SIZE["WIDTH"])
    #         )
    #         # Loại bỏ class 'Normal' khỏi dataset INbreast
    #         normal_label = 'Normal'
    #         if normal_label in le.classes_:
    #             normal_idx = np.where(le.classes_ == normal_label)[0][0]
    #             mask = (y != normal_idx)
    #             X = X[mask]
    #             y = y[mask]
    #         X_train, X_test, y_train, y_test = \
    #             data_preprocessing.dataset_stratified_split(0.2, X, y)

    #         if config.augment_data:
    #             X_train, y_train = generate_image_transforms(X_train, y_train)
    #          # ------ New: expand grayscale → RGB for pretrained nets ------
    #         if config.model != "CNN" and X_train.shape[-1] == 1:
    #             # lặp kênh cuối 3 lần
    #             X_train = np.repeat(X_train, 3, axis=-1)
    #             X_test  = np.repeat(X_test,  3, axis=-1)
    #             # # cập nhật lại train_data, val_data và input_shape
    #             # train_data = (X_train, y_train)
    #             # val_data   = (X_test,  y_test)
    #             # input_shape = (input_shape[0], input_shape[1], 3)
    #             # cập nhật lại train_data, val_data và input_shape từ X_train
    #             train_data = (X_train, y_train)
    #             val_data   = (X_test,  y_test)
    #             input_shape = (X_train.shape[1], X_train.shape[2], 3)                
    #         else:
    #             train_data = (X_train, y_train)
    #             val_data   = (X_test,  y_test)
    #             input_shape = X_train.shape[1:]
    #         # ------------------------------------------------------------
    #         # num_classes = 2 if y_train.ndim == 1 else y_train.shape[1]
    #         num_classes = y_train.shape[1] if y_train.ndim > 1 else 2

    # else:
    #     raise ValueError(f"Unsupported dataset: {config.dataset}")
    elif config.dataset.upper() == "INBREAST":
        data_dir = os.path.join(DATA_ROOT_BREAST, "INbreast", "INbreast")
        if config.is_roi:
            # --- ROI-mode: tf.data.Dataset on-the-fly ---
            ds = import_inbreast_roi_dataset(
                data_dir, le,
                target_size=(
                    config.INBREAST_IMG_SIZE["HEIGHT"],
                    config.INBREAST_IMG_SIZE["WIDTH"]
                )
            )
            # 1) Lọc bỏ class "Normal"
            normal_label = "Normal"
            if normal_label in le.classes_:
                normal_idx = int(np.where(le.classes_ == normal_label)[0][0])
                ds = ds.filter(lambda img, label: tf.not_equal(label, normal_idx))

            # 2) Expand grayscale→RGB cho các model pre-trained
            if config.model.upper() != "CNN":
                ds = ds.map(
                    lambda img, label: (tf.repeat(img, repeats=3, axis=-1), label),
                    num_parallel_calls=tf.data.AUTOTUNE
                )

            # 3) Shuffle, split, batch
            ds = ds.shuffle(buffer_size=1000)
            split = int(0.8 * 1000)
            ds_train = ds.take(split).batch(config.batch_size)
            ds_val   = ds.skip(split).batch(config.batch_size)

            train_data, val_data = ds_train, ds_val

            # 4) Input shape và số class
            input_shape = train_data.element_spec[0].shape[1:]  # (H, W, C)
            num_classes = 2  # Benign & Malignant

        else:
            # --- Full-image mode: load all into numpy ---
            # X, y = import_inbreast_full_dataset(
            #     data_dir, le,
            #     target_size=(
            #         config.INBREAST_IMG_SIZE["HEIGHT"],
            #         config.INBREAST_IMG_SIZE["WIDTH"]
            #     )
            # )
            # print("After filtering out Normal:")
            # print("  X.shape =", X.shape)
            # print("  y.shape =", y.shape)
            # # 1) Lọc bỏ class "Normal"
            # normal_label = "Normal"
            # if normal_label in le.classes_:
            #     normal_idx = int(np.where(le.classes_ == normal_label)[0][0])
            #     # Nếu y one-hot (N,3) → y_int (N,)
            #     if y.ndim > 1:
            #         y_int = np.argmax(y, axis=1)
            #     else:
            #         y_int = y
            #     mask = (y_int != normal_idx)
            #     X, y = X[mask], y[mask]
            # print("After filtering out Normal:", X.shape, y.shape)

            # # 2) Chuyển y về vector 1-chiều nếu cần (model CNN dùng sigmoid)
            # if y.ndim > 1:
            #     y = np.argmax(y, axis=1)
            # # 3) Stratified split đúng cú pháp
            # # X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(
            # #     0.2,
            # #     X,
            # #     y
            # # )
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y,
            #     test_size=0.2,
            #     stratify=y,
            #     random_state=42
            # )
            # # 4) Augmentation nếu bật
            # if config.augment_data:
            #     X_train, y_train = generate_image_transforms(X_train, y_train)

            # # 5) Expand grayscale→RGB cho pretrained nets
            # if config.model.upper() != "CNN" and X_train.shape[-1] == 1:
            #     X_train = np.repeat(X_train, 3, axis=-1)
            #     X_test  = np.repeat(X_test,  3, axis=-1)

            # train_data = (X_train, y_train)
            # val_data   = (X_test,  y_test)
            # input_shape = X_train.shape[1:]   # (H, W, C)
            # num_classes = 2                   # Benign & Malignant
# --- Full-image mode: load all into numpy ---
            X, y = import_inbreast_full_dataset(
                data_dir, le,
                target_size=(
                    config.INBREAST_IMG_SIZE["HEIGHT"],
                    config.INBREAST_IMG_SIZE["WIDTH"]
                )
            )

            # 1) In ra shape lúc mới load
            print("Loaded INbreast full-image:", X.shape, y.shape)

            # 2) Chuyển y (one-hot) → y_int (labels 1-d)
            if y.ndim > 1:
                y = np.argmax(y, axis=1)
            print(" After converting y to 1-d labels:", X.shape, y.shape)
            print(" Label counts:", Counter(y))

            # 3) Lọc bỏ class "Normal"
            normal_idx = int(np.where(le.classes_ == "Normal")[0][0])
            mask = (y != normal_idx)
            X, y = X[mask], y[mask]
            print(" After filtering out Normal:", X.shape, y.shape)
            print(" Remaining label counts:", Counter(y))

            # 4) Stratified split (sklearn đảm bảo mỗi lớp có test sample)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=config.RANDOM_SEED
            )
            print(" Final split:", X_train.shape, X_test.shape)
            print(" Train labels:", Counter(y_train))
            print(" Test  labels:", Counter(y_test))

            # 5) Expand grayscale → RGB nếu pretrained
            if config.model.upper() != "CNN" and X_train.shape[-1] == 1:
                X_train = np.repeat(X_train, 3, axis=-1)
                X_test  = np.repeat(X_test,  3, axis=-1)

            # 6) Gán lại train/val, input_shape, num_classes
            train_data  = (X_train, y_train)
            val_data    = (X_test,  y_test)
            input_shape = X_train.shape[1:]
            num_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    # 4) Build & compile model
    #    Nếu dùng pretrained và ảnh grayscale thì cần convert sang 3-channel trước
    # if config.model != "CNN" and isinstance(train_data, tuple):
    #     Xtr, _ = train_data
    #     if Xtr.shape[-1] == 1:
    #         Xtr = np.repeat(Xtr, 3, axis=-1)
    #         Xte = np.repeat(val_data[0], 3, axis=-1)
    #         train_data = (Xtr, train_data[1])
    #         val_data   = (Xte, val_data[1])
    #         X_train, y_train = train_data
    #         X_test,  y_test  = val_data
    #         input_shape = (input_shape[0], input_shape[1], 3)

    if config.model == "CNN":
        keras_model = build_cnn(input_shape, num_classes)
    else:
        keras_model, _ = build_pretrained_model(
            config.model, input_shape, num_classes
        )

    loss_fn = "binary_crossentropy" if num_classes == 2 else "categorical_crossentropy"
    keras_model.compile(
        loss=loss_fn,
        optimizer=Adam(learning_rate=config.learning_rate),
        metrics=["accuracy"]
    )

    cnn = CnnModel(config.model, num_classes)
    cnn._model = keras_model

    # # 5) Train
    # if config.is_roi:
    #     cnn.train_model(train_data, val_data, class_weights=None)
    # else:
    #     Xtr, Ytr = train_data
    #     Xt, Yt   = val_data
    #     cnn.train_model(Xtr, Xt, Ytr, Yt, class_weights=None)

    # 6) Test mode
    start_time = time.time()
    if config.run_mode == "test":
        # load model and evaluate
        model_fname = f"{config.dataset}_{config.model}.h5"
        model_path = os.path.join(PROJECT_ROOT, "saved_models", model_fname)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")
        keras_model = tf.keras.models.load_model(model_path)
        cnn = CnnModel(config.model, num_classes)
        cnn._model = keras_model

        # prediction & evaluation
        if config.dataset == "CBIS-DDSM":
            ds = dataset_feed.create_dataset(X_test, y_test)
            cnn.make_prediction(ds)
        else:
            cnn.make_prediction(X_test)
        cnn.evaluate_model(y_test, le, "binary", time.time() - start_time)
        return

    # # 7) Build & compile a fresh model for training
    # #   convert grayscale→RGB if necessary for pretrained
    # if config.model != "CNN" and isinstance(X_train, np.ndarray) and X_train.ndim==4 and X_train.shape[-1]==1:
    #     X_train = np.repeat(X_train, 3, axis=-1)
    #     X_test  = np.repeat(X_test,  3, axis=-1)

    # if config.model == "CNN":
    #     in_shape = X_train.shape[1:]
    #     keras_model = build_cnn(in_shape, num_classes)
    # else:
    #     # figure out correct height/width
    #     if config.dataset == "CMMD":
    #         h,w = config.CMMD_IMG_SIZE.values()
    #     else:
    #         size_attr = f"{config.model.upper()}_IMG_SIZE"
    #         h,w = getattr(config, size_attr).values()
    #     keras_model, _ = build_pretrained_model(config.model, (h,w,3), num_classes)

    # loss_fn = "binary_crossentropy" if num_classes==2 else "categorical_crossentropy"
    # keras_model.compile(
    #     loss=loss_fn,
    #     optimizer=optimizers.Adam(learning_rate=config.learning_rate),
    #     metrics=["accuracy"]
    # )

    # 8) Wrap and train
    cnn = CnnModel(config.model, num_classes)
    cnn._model = keras_model

    # if config.dataset == "CBIS-DDSM":
    #     ds_train = dataset_feed.create_dataset(X_train, y_train)
    #     ds_val   = dataset_feed.create_dataset(X_test,  y_test)
    #     cnn.train_model(ds_train, ds_val, y_train, y_test, class_weights=None)
    # else:
    #     cnn.train_model(X_train, X_test, y_train, y_test, class_weights=None)
    # 8) Wrap and train (INBREAST-ROI riêng, CBIS-DDSM riêng, còn lại giữ nguyên)
    # if config.dataset.upper() == "INBREAST" and config.is_roi:
    #     # INBREAST ROI-mode: train trên tf.data.Dataset đã chuẩn bị ở bước load data
    #     cnn.train_model(ds_train, ds_val, class_weights=None)
    if config.dataset.upper() == "INBREAST":
        if config.is_roi:
            # ROI-mode: ds_train, ds_val là tf.data.Dataset có (img,label)
            # Tính class_weights
            labels = [int(l) for _, l in ds_train.unbatch().as_numpy_iterator()]
            class_weights = make_class_weights_from_labels(np.array(labels))
            print("INBREAST ROI class_weights:", class_weights)

            cnn.train_model(ds_train, ds_val, None, None, class_weights)

        else:
            # Full-image mode: X_train, y_train là numpy (1-D), X_test, y_test tương tự
            class_weights = make_class_weights_from_labels(y_train)
            print("INBREAST full-img class_weights:", class_weights)

            cnn.train_model(X_train, X_test, y_train, y_test, class_weights)
    elif config.dataset == "CBIS-DDSM":
        # CBIS-DDSM: build dataset rồi train (như cũ)
        ds_train = dataset_feed.create_dataset(X_train, y_train)
        ds_val   = dataset_feed.create_dataset(X_test,  y_test)
        cnn.train_model(ds_train, ds_val, y_train, y_test, class_weights=None)
    else:
        cnn.train_model(X_train, X_test, y_train, y_test, class_weights=None)
        # Các dataset còn lại (mini-MIAS, CMMD, INBREAST full-mode, etc.)

    # 9) Save & evaluate
    cnn.save_model()

    runtime  = time.time() - start_time
    # cls_type = 'B-M' if num_classes==2 else 'N-B-M'
    cls_type = 'B-M' if num_classes==2 else 'multiclass'
    # cnn.save_weights()
    # cnn.make_prediction(X_test)
    cnn.evaluate_model(X_test, y_test, le, cls_type, runtime)

    if config.verbose_mode:
        print(f"[DONE] Training + evaluation completed in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()
