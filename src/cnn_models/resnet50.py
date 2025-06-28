# import ssl

# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
# from tensorflow.python.keras import Sequential

# import config

# # Needed to download pre-trained weights for ImageNet
# ssl._create_default_https_context = ssl._create_unverified_context


# def create_resnet50_model(num_classes: int):
#     """
#     Function to create a ResNet50 model pre-trained with custom FC Layers.
#     If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
#     larger images.
#     :param num_classes: The number of classes (labels).
#     :return: The ResNet50 model.
#     """
#     # Reconfigure single channel input into a greyscale 3 channel input
#     img_input = Input(shape=(config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE['WIDTH'], 1))
#     img_conc = Concatenate()([img_input, img_input, img_input])

#     # Generate a ResNet50 model with pre-trained ImageNet weights, input as given above, excluding fully connected
#     # layers.
#     model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=img_conc)

#     # Add fully connected layers
#     model = Sequential()
#     # Start with base model consisting of convolutional layers
#     model.add(model_base)

#     # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
#     model.add(Flatten())

#     fully_connected = Sequential(name="Fully_Connected")
#     # Fully connected layers.
#     fully_connected.add(Dropout(0.2, seed=config.RANDOM_SEED, name="Dropout_1"))
#     fully_connected.add(Dense(units=512, activation='relu', name='Dense_1'))
#     # fully_connected.add(Dropout(0.2, name="Dropout_2"))
#     fully_connected.add(Dense(units=32, activation='relu', name='Dense_2'))

#     # Final output layer that uses softmax activation function (because the classes are exclusive).
#     if num_classes == 2:
#         fully_connected.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
#     else:
#         fully_connected.add(
#             Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output'))

#     model.add(fully_connected)

#     # Print model details if running in debug mode.
#     if config.verbose_mode:
#         print("CNN Model used:")
#         print(model.summary())
#         print("Fully connected layers:")
#         print(fully_connected.summary())

#     return model

import ssl
import tensorflow as tf # Thêm import này
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, GlobalAveragePooling2D # Thêm GlobalAveragePooling2D
from tensorflow.keras.models import Model # Sửa từ tensorflow.python.keras sang tensorflow.keras.models

# import config

# # Needed to download pre-trained weights for ImageNet
# ssl._create_default_https_context = ssl._create_unverified_context


# def create_resnet50_model(num_classes: int):
#     """
#     Function to create a ResNet50 model pre-trained with custom FC Layers.
#     :param num_classes: The number of classes (labels).
#     :return: The ResNet50 model.
#     """
#     # Sử dụng giá trị từ config một cách an toàn
#     img_height = getattr(config, 'RESNET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'RESNET_IMG_SIZE', {}).get('WIDTH', 224)

#     # Reconfigure single channel input into a greyscale 3 channel input
#     img_input = Input(shape=(img_height, img_width, 1), name="Input_Grayscale")
#     img_conc = Concatenate(name="Input_RGB_Grayscale")([img_input, img_input, img_input])

#     # Generate a ResNet50 model with pre-trained ImageNet weights
#     model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=img_conc)

#     x = model_base.output
#     x = GlobalAveragePooling2D(name="GlobalAvgPool")(x) # Hoặc Flatten()

#     # Fully connected layers.
#     # Sử dụng giá trị từ config một cách an toàn
#     random_seed_val = getattr(config, 'RANDOM_SEED', None)
#     x = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(x)
#     x = Dense(units=512, activation='relu', name='Dense_1')(x)
#     x = Dense(units=32, activation='relu', name='Dense_2')(x)

#     # Final output layer - Đã sửa đổi
#     if num_classes == 2:
#         outputs = Dense(num_classes, activation='softmax', name='Output')(x)
#     elif num_classes > 2:
#         outputs = Dense(num_classes, activation='softmax', name='Output')(x)
#     else: # num_classes = 1 hoặc < 1
#         print(f"[WARNING] resnet50: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
#         outputs = Dense(1, activation='sigmoid', name='Output')(x)
        
#     model = Model(inputs=img_input, outputs=outputs, name="ResNet50_Custom")

#     verbose_mode_val = getattr(config, 'verbose_mode', False)
#     if verbose_mode_val:
#         print("CNN Model used (ResNet50_Custom):")
#         model.summary()

#     return model

from tensorflow.keras.applications import ResNet50
import config

def create_resnet50_model(num_classes: int, input_shape: tuple):
    """
    Hàm tạo model ResNet50 có khả năng chấp nhận đầu vào 1 kênh (ảnh xám)
    hoặc 3 kênh (ảnh màu) một cách linh hoạt.

    :param num_classes: Số lượng lớp (nhãn).
    :param input_shape: Tuple xác định kích thước đầu vào, ví dụ: (224, 224, 1) hoặc (224, 224, 3).
    :return: Model ResNet50 đã được tạo.
    """
    if len(input_shape) != 3:
        raise ValueError("input_shape phải là một tuple có 3 phần tử: (height, width, channels)")

    img_height, img_width, channels = input_shape
    
    # Khởi tạo Input layer với shape được cung cấp
    img_input = Input(shape=input_shape, name="Input_Layer")

    # --- ĐÂY LÀ LOGIC IF MÀ BẠN YÊU CẦU ---
    # Kiểm tra số kênh của ảnh đầu vào để xử lý tương ứng
    if channels == 1:
        # Nếu là ảnh xám (1 kênh), tự động nhân 3 lần để tạo thành ảnh 3 kênh giả
        print("[INFO ResNet50] Đầu vào là 1 kênh. Tự động chuyển thành 3 kênh.")
        x = Concatenate(name="Replicate_Grayscale_To_3_Channels")([img_input, img_input, img_input])
    elif channels == 3:
        # Nếu đã là ảnh 3 kênh, sử dụng trực tiếp
        print("[INFO ResNet50] Đầu vào là 3 kênh. Sử dụng trực tiếp.")
        x = img_input
    else:
        # Ném ra lỗi nếu số kênh không phải là 1 hoặc 3
        raise ValueError(f"Kênh đầu vào mong muốn là 1 hoặc 3, nhưng nhận được {channels} kênh.")

    # Tạo model ResNet50 với pre-trained ImageNet weights,
    # và ĐẦU VÀO là tensor 'x' (đã đảm bảo là 3 kênh)
    model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=x)

    # Các lớp Fully Connected (giữ nguyên logic của bạn)
    y = model_base.output
    y = GlobalAveragePooling2D(name="GlobalAvgPool")(y)
    
    random_seed_val = getattr(config, 'RANDOM_SEED', None)
    y = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(y)
    y = Dense(units=512, activation='relu', name='Dense_1')(y)
    y = Dense(units=32, activation='relu', name='Dense_2')(y)

    # Lớp Output cuối cùng (giữ nguyên logic của bạn)
    if num_classes >= 2:
        outputs = Dense(num_classes, activation='softmax', name='Output')(y)
    else:
        outputs = Dense(1, activation='sigmoid', name='Output')(y)
        
    # Tạo model hoàn chỉnh, với đầu vào là img_input gốc
    model = Model(inputs=img_input, outputs=outputs, name="ResNet50_Flexible_Input")

    if getattr(config, 'verbose_mode', False):
        print("CNN Model used (ResNet50_Flexible_Input):")
        model.summary()

    return model