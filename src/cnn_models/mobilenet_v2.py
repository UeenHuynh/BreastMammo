import ssl
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, GlobalAveragePooling2D
from tensorflow.python.keras import Sequential
from tensorflow.keras import regularizers

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


# def create_mobilenet_model(num_classes: int):
#     """
#     Function to create a MobileNetV2 model pre-trained with custom FC Layers.
#     If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
#     larger images.
#     :param num_classes: The number of classes (labels).
#     :return: The MobileNetV2 model.
#     """
#     # Reconfigure single channel input into a greyscale 3 channel input
#     img_input = Input(shape=(config.DENSE_NET_IMG_SIZE['HEIGHT'], config.DENSE_NET_IMG_SIZE['WIDTH'], 1))
#     img_conc = Concatenate()([img_input, img_input, img_input])

#     # Generate a MobileNetV2 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
#     model_base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=img_conc)

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

# def create_mobilenet_model(num_classes: int):
#     # 1) Input grayscale → 3 channels
#     inp = Input(shape=(config.MOBILE_NET_IMG_SIZE['HEIGHT'],
#                        config.MOBILE_NET_IMG_SIZE['WIDTH'], 1))
#     x = Concatenate()([inp, inp, inp])

#     # 2) Base MobileNetV2
#     base = MobileNetV2(include_top=False,
#                        weights='imagenet',
#                        input_tensor=x)
#     x = base.output

#     # 3) Head
#     x = Flatten()(x)
#     x = Dropout(0.2, seed=config.RANDOM_SEED)(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#     if num_classes == 2:
#         out = Dense(1, activation='sigmoid', name='Output')(x)
#     else:
#         out = Dense(num_classes, activation='softmax', name='Output')(x)

#     return Model(inputs=inp, outputs=out, name='MobileNetV2_Custom')

# def create_mobilenet_model(num_classes: int):
#     # Sử dụng giá trị từ config một cách an toàn
#     img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)
    
#     # 1) Input grayscale → 3 channels
#     # inp = Input(shape=(img_height, img_width, 1), name="Input_Grayscale")
#     # x_conc = Concatenate(name="Input_RGB_Grayscale")([inp, inp, inp]) # Đổi tên biến để không trùng inp

#     # # 2) Base MobileNetV2
#     # base = MobileNetV2(include_top=False,
#     #                    weights='imagenet',
#     #                    input_tensor=x_conc) # Sử dụng x_conc
#     # x = base.output
#     # Input của toàn bộ mô hình vẫn là ảnh xám 1 kênh
#     inp_gray = Input(shape=(img_height, img_width, 1), name="Input_Grayscale")
#     print(f"[DEBUG create_mobilenet] inp_gray shape: {inp_gray.shape}") # Mong đợi (None, H, W, 1)

#     # Lặp kênh để tạo ảnh 3 kênh
#     x_conc = Concatenate(name="Input_RGB_From_Grayscale")([inp_gray, inp_gray, inp_gray]) # Shape: (None, H, W, 3)
#     print(f"[DEBUG create_mobilenet] x_conc shape (after Concatenate): {x_conc.shape}") # Mong đợi (None, H, W, 3)

#     # Tạo MobileNetV2 base, nó sẽ tự tạo Input layer (None, H, W, 3) nếu không có input_tensor
#     # Sau đó, chúng ta truyền x_conc (đã có 3 kênh) vào nó.
#     base_mobilenet = MobileNetV2(input_shape=(img_height, img_width, 3), # Định nghĩa input_shape cho base model
#                                  include_top=False,
#                                  weights='imagenet')
#     print(f"[DEBUG create_mobilenet] base_mobilenet.input_shape (expected by base): {base_mobilenet.input_shape}") # Mong đợi (None, H, W, 3)

#     # Truyền output của Concatenate vào base model
#     x = base_mobilenet(x_conc) # x_conc có shape (None, H, W, 3)
#     print(f"[DEBUG create_mobilenet] x shape (output of base_mobilenet(x_conc)): {x.shape}")

#     # inp = Input(shape=(img_height, img_width, 3), name="Input_RGB") # THAY ĐỔI Ở ĐÂY: từ 1 thành 3 kênh

#     # base = MobileNetV2(include_top=False,
#     #                    weights='imagenet',
#     #                    input_tensor=inp) # input_tensor bây giờ là inp (3 kênh)
#     # x = base.output
#     # 3) Head
#     # Có thể chọn GlobalAveragePooling2D thay vì Flatten tùy theo hiệu năng mong muốn
#     x = GlobalAveragePooling2D(name="GlobalAvgPool")(x) 
#     # x = Flatten()(x) # Nếu giữ Flatten

#     # Sử dụng giá trị từ config một cách an toàn
#     random_seed_val = getattr(config, 'RANDOM_SEED', None)
#     x = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(x) # Đặt tên để dễ debug
#     x = Dense(512, activation='relu', name="Dense_1")(x)
#     x = Dense(32, activation='relu', name="Dense_2")(x)

#     # Lớp output - Đã sửa đổi
#     if num_classes == 2:
#         out = Dense(num_classes, activation='softmax', name='Output')(x) # 2 units, softmax
#     elif num_classes > 2:
#         out = Dense(num_classes, activation='softmax', name='Output')(x)
#     else: # num_classes = 1 hoặc < 1
#         print(f"[WARNING] mobilenet_v2: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
#         out = Dense(1, activation='sigmoid', name='Output')(x)

#     final_model = Model(inputs=inp_gray, outputs=out, name='MobileNetV2_Custom') # Đổi tên biến model

#     verbose_mode_val = getattr(config, 'verbose_mode', False)
#     if verbose_mode_val:
#         print("CNN Model used (MobileNetV2_Custom):")
#         final_model.summary()
        
#     return final_model


# def create_mobilenet_model(num_classes: int):
#     img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)
    
#     # Input của toàn bộ mô hình vẫn là ảnh xám 1 kênh
#     inp_gray = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_MobileNet")
    
#     # Lặp kênh để tạo ảnh 3 kênh
#     x_conc = Concatenate(name="MobileNet_Grayscale_to_RGB")([inp_gray, inp_gray, inp_gray]) 
#     # x_conc bây giờ có shape (None, H, W, 3)
    
#     # Tạo một Input layer mới CỤ THỂ cho base_mobilenet
#     mobilenet_input = Input(shape=(img_height, img_width, 3), name="MobileNet_Base_Input")
    
#     # Khởi tạo MobileNetV2 base, sử dụng Input layer mới này
#     base_mobilenet_model = MobileNetV2(input_tensor=mobilenet_input, # Dùng input_tensor ở đây
#                                  include_top=False,
#                                  weights='imagenet',
#                                  name="MobileNetV2_Base_Explicit_Input")
    
#     # Lấy output của base_mobilenet_model
#     base_output = base_mobilenet_model.output 
    
#     # Tạo một Model trung gian từ mobilenet_input đến base_output
#     # Điều này "đóng gói" base_mobilenet với input 3 kênh rõ ràng của nó
#     intermediate_base_model = Model(inputs=mobilenet_input, outputs=base_output, name="Wrapped_MobileNet_Base")

#     # Bây giờ, truyền x_conc (3 kênh từ dữ liệu của bạn) vào intermediate_base_model
#     x = intermediate_base_model(x_conc)

#     x = GlobalAveragePooling2D(name="GlobalAvgPool")(x) 
#     # x = Flatten()(x) # Nếu giữ Flatten

#     # Sử dụng giá trị từ config một cách an toàn
#     random_seed_val = getattr(config, 'RANDOM_SEED', None)
#     x = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(x) # Đặt tên để dễ debug
#     x = Dense(512, activation='relu', name="Dense_1")(x)
#     x = Dense(32, activation='relu', name="Dense_2")(x)

#     # Lớp output - Đã sửa đổi
#     if num_classes == 2:
#         out = Dense(num_classes, activation='softmax', name='Output')(x) # 2 units, softmax
#     elif num_classes > 2:
#         out = Dense(num_classes, activation='softmax', name='Output')(x)
#     else: # num_classes = 1 hoặc < 1
#         print(f"[WARNING] mobilenet_v2: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
#         out = Dense(1, activation='sigmoid', name='Output')(x)

#     final_model = Model(inputs=inp_gray, outputs=out, name='MobileNetV2_Custom') # Đổi tên biến model

#     verbose_mode_val = getattr(config, 'verbose_mode', False)
#     if verbose_mode_val:
#         print("CNN Model used (MobileNetV2_Custom):")
#         final_model.summary()
        
#     return final_model

# def create_mobilenet_model(num_classes: int):
#     img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)
    
#     # Input của toàn bộ mô hình custom là ảnh xám 1 kênh
#     inp_gray = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_MobileNet")
#     if config.verbose_mode: print(f"    [MobileNet Create] inp_gray.shape: {inp_gray.shape}")
    
#     # Lớp Concatenate để chuyển từ 1 kênh sang 3 kênh
#     x_conc = Concatenate(name="MobileNet_Grayscale_to_RGB")([inp_gray, inp_gray, inp_gray])
#     if config.verbose_mode: print(f"    [MobileNet Create] x_conc.shape (after concat): {x_conc.shape}") # Phải là (None, H, W, 3)
    
#     # Khởi tạo MobileNetV2 base, chỉ định input_shape là 3 kênh.
#     # KHÔNG sử dụng input_tensor ở đây.
#     base_mobilenet = MobileNetV2(input_shape=(img_height, img_width, 3), 
#                                  include_top=False,
#                                  weights='imagenet',
#                                  name="MobileNetV2_Base") # Đặt tên để dễ theo dõi
#     if config.verbose_mode: 
#         print(f"    [MobileNet Create] base_mobilenet (MobileNetV2_Base) is created.")
#         print(f"    [MobileNet Create] base_mobilenet.input_shape (expected by base): {base_mobilenet.input_shape}") # Phải là (None, H, W, 3)

#     # Gọi base_mobilenet như một hàm (layer) với x_conc (3 kênh) làm đầu vào.
#     x = base_mobilenet(x_conc) 
#     if config.verbose_mode: print(f"    [MobileNet Create] x.shape (output of base_mobilenet(x_conc)): {x.shape}")
    
#     # Các lớp custom phía trên
#     x = GlobalAveragePooling2D(name="MobileNet_GlobalAvgPool")(x)
# def create_mobilenet_model(num_classes: int):
#     img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)

#     # Input của mô hình bây giờ là ảnh 3 kênh
#     inp_rgb = Input(shape=(img_height, img_width, 3), name="Input_RGB_MobileNet")
#     if config.verbose_mode: print(f"    [MobileNet Create] inp_rgb.shape: {inp_rgb.shape}")

#     # Không cần Concatenate nữa

#     base_mobilenet = MobileNetV2(input_tensor=inp_rgb, # Truyền trực tiếp inp_rgb
#                                  include_top=False,
#                                  weights='imagenet',
#                                  name="MobileNetV2_Base")
#     if config.verbose_mode: 
#         print(f"    [MobileNet Create] base_mobilenet.input_shape (expected by base): {base_mobilenet.input_shape}")

#     x = base_mobilenet.output # Output của base_mobilenet
#     if config.verbose_mode: print(f"    [MobileNet Create] x.shape (output of base_mobilenet): {x.shape}")

#     x = GlobalAveragePooling2D(name="MobileNet_GlobalAvgPool")(x)
    
#     random_seed_val = getattr(config, 'RANDOM_SEED', None)
#     x = Dropout(0.2, seed=random_seed_val, name="MobileNet_Dropout_1")(x)
#     x = Dense(512, activation='relu', name="MobileNet_Dense_1")(x)
#     x = Dense(32, activation='relu', name="MobileNet_Dense_2")(x)

#     # Lớp output
#     if num_classes == 2:
#         out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
#     elif num_classes > 2:
#         out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
#     else: # num_classes <= 1 (trường hợp fallback, ít khi xảy ra nếu num_classes được xác định đúng)
#         print(f"[WARNING create_mobilenet_model] num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid.")
#         out = Dense(1, activation='sigmoid', name='MobileNet_Output')(x)
            
#     # final_model = Model(inputs=inp_gray, outputs=out, name='MobileNetV2_Custom_Corrected')
#     final_model = Model(inputs=inp_rgb, outputs=out, name='MobileNetV2_Custom_Corrected')

#     if getattr(config, 'verbose_mode', False):
#         print("--- MobileNetV2_Custom_Corrected Summary ---")
#         final_model.summary(line_length=150) # Tăng line_length
#         # Nếu bạn muốn xem summary của base_mobilenet riêng:
#         # print("\n--- MobileNetV2_Base (internal) Summary ---")
#         # base_mobilenet.summary(line_length=150)
            
#     return final_model

# def create_mobilenet_model(num_classes: int):
#     img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)
    
#     final_model_input_tensor = None # Tensor Input cho toàn bộ Model cuối cùng
#     processed_input_for_base = None # Tensor sẽ được đưa vào MobileNetV2 base

#     dataset_name_upper = getattr(config, 'dataset', '').upper()

#     if dataset_name_upper == "INBREAST":
#         # INbreast được giả định là đã cung cấp ảnh 3 kênh từ hàm load dữ liệu
#         # Do đó, Input của model này phải là 3 kênh.
#         inp_rgb_inbreast = Input(shape=(img_height, img_width, 3), name="Input_RGB_INbreast_MobileNet")
#         if config.verbose_mode: 
#             print(f"    [MobileNet Create INBREAST] Input layer is 3-channel: {inp_rgb_inbreast.shape}")
#         processed_input_for_base = inp_rgb_inbreast # Dùng trực tiếp, không Concatenate
#         final_model_input_tensor = inp_rgb_inbreast

#     elif dataset_name_upper == "CMMD":
#         # CMMD được giả định là cung cấp ảnh 1 kênh từ hàm load dữ liệu
#         # Model sẽ nhận 1 kênh và Concatenate thành 3 kênh bên trong.
#         inp_gray_cmmd = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_CMMD_MobileNet")
#         if config.verbose_mode: 
#             print(f"    [MobileNet Create CMMD] Input layer is 1-channel: {inp_gray_cmmd.shape}")
        
#         x_conc_cmmd = Concatenate(name="CMMD_MobileNet_Grayscale_to_RGB")([inp_gray_cmmd, inp_gray_cmmd, inp_gray_cmmd])
#         if config.verbose_mode: 
#             print(f"    [MobileNet Create CMMD] Concatenated to 3-channel: {x_conc_cmmd.shape}")
#         processed_input_for_base = x_conc_cmmd
#         final_model_input_tensor = inp_gray_cmmd
#     else:
#         # Xử lý mặc định hoặc cho các dataset khác: giả định 1 kênh và concat
#         if config.verbose_mode:
#             print(f"    [MobileNet Create DEFAULT] Dataset '{config.dataset}' not explicitly INBREAST or CMMD. Assuming 1-channel input and concatenating.")
#         inp_gray_default = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_Default_MobileNet")
#         x_conc_default = Concatenate(name="Default_MobileNet_Grayscale_to_RGB")([inp_gray_default, inp_gray_default, inp_gray_default])
#         processed_input_for_base = x_conc_default
#         final_model_input_tensor = inp_gray_default

#     if processed_input_for_base is None or final_model_input_tensor is None:
#         raise ValueError("Input tensor for MobileNet base or final model could not be determined based on dataset.")
    
#     if config.verbose_mode:
#         print(f"    [MobileNet Create] Effective input tensor to MobileNetV2 base has shape: {processed_input_for_base.shape}")

#     # Sử dụng input_tensor khi khởi tạo MobileNetV2 base
#     # input_tensor này phải là 3 kênh
#     if processed_input_for_base.shape[-1] != 3:
#         raise ValueError(f"Tensor 'processed_input_for_base' going into MobileNetV2 base must have 3 channels, but got shape {processed_input_for_base.shape}")

#     base_mobilenet = MobileNetV2(input_tensor=processed_input_for_base, 
#                                  include_top=False,
#                                  weights='imagenet',
#                                  name="MobileNetV2_Base")
#     if config.verbose_mode: 
#         print(f"    [MobileNet Create] base_mobilenet.input.shape (actual tensor passed): {base_mobilenet.input.shape}") # Phải là (None, H, W, 3)
    
#     x = base_mobilenet.output 
#     if config.verbose_mode: 
#         print(f"    [MobileNet Create] x.shape (output of base_mobilenet): {x.shape}")
    
#     x = GlobalAveragePooling2D(name="MobileNet_GlobalAvgPool")(x)
    
#     random_seed_val = getattr(config, 'RANDOM_SEED', None)
#     x = Dropout(0.2, seed=random_seed_val, name="MobileNet_Dropout_1")(x)
#     x = Dense(512, activation='relu', name="MobileNet_Dense_1")(x)
#     x = Dense(32, activation='relu', name="MobileNet_Dense_2")(x)

#     # Lớp output
#     if num_classes == 2:
#         out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
#     elif num_classes > 2:
#         out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
#     else: 
#         print(f"[WARNING create_mobilenet_model] num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid.")
#         out = Dense(1, activation='sigmoid', name='MobileNet_Output')(x)
            
#     final_model = Model(inputs=final_model_input_tensor, outputs=out, name='MobileNetV2_Custom_DatasetSpecificInput')

#     if getattr(config, 'verbose_mode', False):
#         print(f"--- MobileNetV2_Custom_DatasetSpecificInput ({config.dataset}) Summary ---")
#         final_model.summary(line_length=150)
            
#     return final_model

# def create_mobilenet_model(num_classes: int):
#     img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)
    
#     final_model_input_tensor = None # Tensor Input cho toàn bộ Model cuối cùng
#     processed_input_for_base = None # Tensor sẽ được đưa vào MobileNetV2 base

#     dataset_name_upper = getattr(config, 'dataset', '').upper()

#     if config.verbose_mode:
#         print(f"    [MobileNet Create] Detected dataset: {config.dataset}, Model: {config.model}")

#     # Xử lý kênh đầu vào dựa trên dataset
#     if dataset_name_upper == "INBREAST" and config.model.upper() == "MOBILENET":
#         # INbreast cho MobileNet: Giả định dữ liệu đầu vào đã là 3 kênh từ hàm load
#         # Do đó, Input của model này phải là 3 kênh.
#         inp_rgb_inbreast = Input(shape=(img_height, img_width, 3), name="Input_RGB_INbreast_MobileNet")
#         if config.verbose_mode: 
#             print(f"    [MobileNet Create INBREAST] Input layer is 3-channel: {inp_rgb_inbreast.shape}")
#         processed_input_for_base = inp_rgb_inbreast # Dùng trực tiếp, không Concatenate
#         final_model_input_tensor = inp_rgb_inbreast

#     elif dataset_name_upper == "CMMD" and config.model.upper() == "MOBILENET":
#         # CMMD cho MobileNet: Giả định dữ liệu đầu vào là 1 kênh, cần Concatenate
#         inp_gray_cmmd = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_CMMD_MobileNet")
#         if config.verbose_mode: 
#             print(f"    [MobileNet Create CMMD] Input layer is 1-channel: {inp_gray_cmmd.shape}")
        
#         x_conc_cmmd = Concatenate(name="CMMD_MobileNet_Grayscale_to_RGB")([inp_gray_cmmd, inp_gray_cmmd, inp_gray_cmmd])
#         if config.verbose_mode: 
#             print(f"    [MobileNet Create CMMD] Concatenated to 3-channel: {x_conc_cmmd.shape}")
#         processed_input_for_base = x_conc_cmmd
#         final_model_input_tensor = inp_gray_cmmd
#     else:
#         # Trường hợp mặc định cho MobileNet (nếu dataset không phải INbreast hoặc CMMD nhưng model là MobileNet)
#         # Hoặc nếu hàm này được gọi bởi một model khác không phải MobileNet (dù tên hàm là create_mobilenet_model)
#         # Để an toàn, nếu model là MobileNet, ta giả định 1 kênh và concat.
#         # Nếu model không phải MobileNet, logic này không nên được chạy (CnnModel.__init__ sẽ chọn đúng hàm create).
#         if config.model.upper() == "MOBILENET":
#             if config.verbose_mode:
#                 print(f"    [MobileNet Create DEFAULT] Dataset '{config.dataset}' with MobileNet. Assuming 1-channel input and concatenating.")
#             inp_gray_default = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_Default_MobileNet")
#             x_conc_default = Concatenate(name="Default_MobileNet_Grayscale_to_RGB")([inp_gray_default, inp_gray_default, inp_gray_default])
#             processed_input_for_base = x_conc_default
#             final_model_input_tensor = inp_gray_default
#         else:
#             # Fallback này ít khi xảy ra nếu CnnModel chọn đúng hàm.
#             # Nếu model không phải MobileNet, nó nên có hàm create riêng.
#             # Tuy nhiên, để code chạy được, ta mặc định input 3 kênh cho các model pretrain khác.
#             print(f"    [MobileNet Create FALLBACK] Model is {config.model} (not MobileNet but this function was called). Assuming 3-channel input directly.")
#             inp_rgb_fallback = Input(shape=(img_height, img_width, 3), name="Input_RGB_Fallback_OtherModel")
#             processed_input_for_base = inp_rgb_fallback
#             final_model_input_tensor = inp_rgb_fallback


#     if processed_input_for_base is None or final_model_input_tensor is None:
#         raise ValueError(f"Could not determine input tensors for MobileNet based on dataset '{config.dataset}' and model '{config.model}'.")
    
#     # Đảm bảo processed_input_for_base là 3 kênh trước khi đưa vào MobileNetV2 base
#     if processed_input_for_base.shape[-1] != 3:
#         raise ValueError(f"Internal Error: 'processed_input_for_base' tensor going into MobileNetV2 base must have 3 channels, but got shape {processed_input_for_base.shape}")

#     if config.verbose_mode:
#         print(f"    [MobileNet Create] Effective input tensor to MobileNetV2 base has shape: {processed_input_for_base.shape}")

#     # Sử dụng input_tensor khi khởi tạo MobileNetV2 base
#     base_mobilenet = MobileNetV2(input_tensor=processed_input_for_base, 
#                                  include_top=False,
#                                  weights='imagenet',
#                                  name="MobileNetV2_Base") # Giữ tên này nhất quán
    
#     if config.verbose_mode: 
#         # Sửa lỗi AttributeError bằng cách kiểm tra kiểu của base_mobilenet.input
#         actual_input_to_base = base_mobilenet.input
#         if isinstance(actual_input_to_base, list):
#             if actual_input_to_base: # Nếu list không rỗng
#                 print(f"    [MobileNet Create] base_mobilenet.input is a LIST. Shape of first input tensor: {actual_input_to_base[0].shape}")
#             else:
#                 print(f"    [MobileNet Create] base_mobilenet.input is an EMPTY LIST.")
#         elif hasattr(actual_input_to_base, 'shape'): # Nếu là một tensor đơn lẻ
#             print(f"    [MobileNet Create] base_mobilenet.input is a TENSOR. Shape: {actual_input_to_base.shape}")
#         else:
#             print(f"    [MobileNet Create] base_mobilenet.input is of unexpected type: {type(actual_input_to_base)}")
    
#     x = base_mobilenet.output 
#     if config.verbose_mode: 
#         print(f"    [MobileNet Create] x.shape (output of base_mobilenet): {x.shape}")
    
#     x = GlobalAveragePooling2D(name="MobileNet_GlobalAvgPool")(x)
    
#     random_seed_val = getattr(config, 'RANDOM_SEED', None)
#     x = Dropout(0.2, seed=random_seed_val, name="MobileNet_Dropout_1")(x)
#     x = Dense(512, activation='relu', name="MobileNet_Dense_1")(x)
#     x = Dense(32, activation='relu', name="MobileNet_Dense_2")(x)

#     # Lớp output
#     if num_classes == 2:
#         out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
#     elif num_classes > 2:
#         out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
#     else: 
#         print(f"[WARNING create_mobilenet_model] num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid.")
#         out = Dense(1, activation='sigmoid', name='MobileNet_Output')(x)
            
#     final_model = Model(inputs=final_model_input_tensor, outputs=out, name=f'MobileNetV2_Custom_{config.dataset}')

#     if getattr(config, 'verbose_mode', False):
#         print(f"--- MobileNetV2_Custom ({config.dataset}) Summary ---")
#         final_model.summary(line_length=150)
            
#     return final_model

def create_mobilenet_model(num_classes: int):
    img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
    img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)
    
    # Tensor đầu vào cho toàn bộ Model custom (MobileNetV2_Custom_...)
    # và tensor sẽ được đưa vào MobileNetV2 base.
    final_model_input_layer = None 
    tensor_fed_to_mobilenet_base = None

    dataset_name_upper = getattr(config, 'dataset', '').upper()
    model_name_upper = getattr(config, 'model', '').upper()

    if config.verbose_mode:
        print(f"    [MobileNet Create] Initializing for Dataset: {config.dataset}, Model: {config.model}")

    if model_name_upper == "MOBILENET":
        if dataset_name_upper == "INBREAST":
            # INbreast: Giả định hàm load_inbreast_data_no_pectoral_removal đã cung cấp ảnh 3 kênh.
            # Input của model này sẽ là 3 kênh.
            inp_rgb = Input(shape=(img_height, img_width, 3), name="Input_RGB_INbreast_MobileNet")
            if config.verbose_mode: print(f"    [MobileNet Create INBREAST] Expecting 3-channel input: {inp_rgb.shape}")
            tensor_fed_to_mobilenet_base = inp_rgb # Dùng trực tiếp, không Concatenate
            final_model_input_layer = inp_rgb
        
        elif dataset_name_upper == "CMMD":
            # CMMD: Giả định hàm import_cmmd_dataset cung cấp ảnh 1 kênh.
            # Model sẽ nhận 1 kênh và Concatenate thành 3 kênh bên trong.
            inp_gray = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_CMMD_MobileNet")
            if config.verbose_mode: print(f"    [MobileNet Create CMMD] Expecting 1-channel input: {inp_gray.shape}")
            
            concatenated_rgb = Concatenate(name="CMMD_MobileNet_Grayscale_to_RGB")([inp_gray, inp_gray, inp_gray])
            if config.verbose_mode: print(f"    [MobileNet Create CMMD] Concatenated to 3-channel: {concatenated_rgb.shape}")
            tensor_fed_to_mobilenet_base = concatenated_rgb
            final_model_input_layer = inp_gray # Input của model tổng thể là 1 kênh
        
        else: # Trường hợp dataset khác không được xử lý riêng cho MobileNet
            print(f"    [MobileNet Create DEFAULT] Dataset '{config.dataset}' with MobileNet. Assuming 1-channel input, will concatenate.")
            inp_gray_default = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_Default_MobileNet")
            concatenated_default_rgb = Concatenate(name="Default_MobileNet_Grayscale_to_RGB")([inp_gray_default, inp_gray_default, inp_gray_default])
            tensor_fed_to_mobilenet_base = concatenated_default_rgb
            final_model_input_layer = inp_gray_default
    else:
        # Nếu hàm này được gọi nhưng config.model không phải là MOBILENET
        # (điều này không nên xảy ra nếu CnnModel.__init__ hoạt động đúng)
        # Tạo một input 3 kênh mặc định để tránh lỗi, nhưng cảnh báo.
        print(f"[ERROR create_mobilenet_model] This function was called when config.model is '{config.model}', not 'MOBILENET'. This is unexpected.")
        print("    Defaulting to a 3-channel input for MobileNet base, but data pipeline might be incorrect.")
        inp_rgb_error_fallback = Input(shape=(img_height, img_width, 3), name="Input_RGB_Error_Fallback")
        tensor_fed_to_mobilenet_base = inp_rgb_error_fallback
        final_model_input_layer = inp_rgb_error_fallback

    if tensor_fed_to_mobilenet_base is None or final_model_input_layer is None:
        raise ValueError(f"Critical Error: Input tensors for MobileNet could not be constructed for dataset '{config.dataset}'.")
    
    # Đảm bảo tensor đưa vào MobileNetV2 base PHẢI LÀ 3 kênh
    if tensor_fed_to_mobilenet_base.shape[-1] != 3:
        raise ValueError(f"Internal Error: Tensor 'tensor_fed_to_mobilenet_base' intended for MobileNetV2 base must have 3 channels, but got shape {tensor_fed_to_mobilenet_base.shape}")

    if config.verbose_mode:
        print(f"    [MobileNet Create] Effective tensor being fed to MobileNetV2 base has shape: {tensor_fed_to_mobilenet_base.shape}")

    # Khởi tạo MobileNetV2 base.
    # Cách 1: Sử dụng input_tensor (ổn nếu tensor_fed_to_mobilenet_base là một Keras tensor hợp lệ)
    base_mobilenet_app = MobileNetV2(input_tensor=tensor_fed_to_mobilenet_base, 
                                     include_top=False,
                                     weights='imagenet',
                                     name="MobileNetV2_Base")

    # # Cách 2: Khai báo input_shape và gọi model như một layer (thường ổn định hơn)
    # base_mobilenet_app = MobileNetV2(input_shape=(img_height, img_width, 3), 
    #                                  include_top=False,
    #                                  weights='imagenet',
    #                                  name="MobileNetV2_Base_Called_As_Layer")
    # x_from_base = base_mobilenet_app(tensor_fed_to_mobilenet_base)


    if config.verbose_mode: 
        # Sửa lỗi AttributeError bằng cách kiểm tra kiểu của base_mobilenet_app.input
        actual_input_received_by_base = base_mobilenet_app.input
        if isinstance(actual_input_received_by_base, list):
            if actual_input_received_by_base: 
                print(f"    [MobileNet Create] base_mobilenet_app.input is a LIST. Shape of first input tensor: {actual_input_received_by_base[0].shape}")
            else:
                print(f"    [MobileNet Create] base_mobilenet_app.input is an EMPTY LIST.")
        elif hasattr(actual_input_received_by_base, 'shape'): 
            print(f"    [MobileNet Create] base_mobilenet_app.input is a TENSOR. Shape: {actual_input_received_by_base.shape}")
        else:
            print(f"    [MobileNet Create] base_mobilenet_app.input is of unexpected type: {type(actual_input_received_by_base)}")
    
    # Lấy output từ base_mobilenet_app
    # Nếu dùng Cách 2 ở trên, x sẽ là x_from_base
    x = base_mobilenet_app.output 
                                 
    if config.verbose_mode: 
        print(f"    [MobileNet Create] x.shape (output of base_mobilenet_app): {x.shape}")
    
    x = GlobalAveragePooling2D(name="MobileNet_GlobalAvgPool")(x)
    
    random_seed_val = getattr(config, 'RANDOM_SEED', None) # Lấy seed từ config
    x = Dropout(0.4, seed=random_seed_val, name="MobileNet_Dropout_1")(x)
    # x = Dense(512, activation='relu', name="MobileNet_Dense_1")(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005), name="MobileNet_Dense_1")(x)
    x = Dense(32, activation='relu', name="MobileNet_Dense_2")(x)
    x = Dropout(0.3, name="MobileNet_Dropout_2")(x) # Giảm từ 0.5 xuống 0.3

    # Lớp output
    if num_classes == 2:
        out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
    elif num_classes > 2:
        out = Dense(num_classes, activation='softmax', name='MobileNet_Output')(x)
    else: 
        # Trường hợp num_classes = 1 hoặc lỗi (<=0), dùng sigmoid.
        # Tuy nhiên, logic compile trong CnnModel có thể cần điều chỉnh cho trường hợp này.
        if config.verbose_mode: print(f"[WARNING create_mobilenet_model] num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid.")
        out = Dense(1, activation='sigmoid', name='MobileNet_Output')(x)
            
    # Model cuối cùng sẽ có input là final_model_input_tensor
    final_model = Model(inputs=final_model_input_layer, outputs=out, name=f'MobileNetV2_Custom_{config.dataset}')

    if getattr(config, 'verbose_mode', False):
        print(f"--- MobileNetV2_Custom ({config.dataset}) Summary ---")
        # final_model.summary(line_length=150)
            
    return final_model