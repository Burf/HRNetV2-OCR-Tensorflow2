import tensorflow as tf
import numpy as np

def hr_basic_block(x, n_feature, stride_size = (1, 1), shortcut = False, **kwargs):
    out = tf.keras.layers.Conv2D(n_feature, 3, strides = stride_size, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(x)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
    out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
    out = tf.keras.layers.Conv2D(n_feature, 3, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
    
    if shortcut:
        x = tf.keras.layers.Conv2D(n_feature, 1, strides = stride_size, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(x)
        x = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(x)
        out = tf.keras.layers.Add()([out, x])
    else:
        out = tf.keras.layers.Add()([out, x])
    
    out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
    return out

def hr_bottleneck_block(x, n_feature, stride_size = (1, 1), shortcut = False, expansion = 4, **kwargs):
    n_decode_filter = n_feature // expansion
    out = tf.keras.layers.Conv2D(n_decode_filter, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(x)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
    out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
    out = tf.keras.layers.Conv2D(n_decode_filter, 3, strides = stride_size, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
    out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
    out = tf.keras.layers.Conv2D(n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
   
    if shortcut:
        x = tf.keras.layers.Conv2D(n_feature, 1, strides = stride_size, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(x)
        x = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(x)
        out = tf.keras.layers.Add()([out, x])
    else:
        out = tf.keras.layers.Add()([out, x])
    
    out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
    return out

def hr_transition_block(x, n_feature, **kwargs):
    if not isinstance(x, list):
        x = [x]
    if isinstance(n_feature, int):
        n_feature = [n_feature * (2 ** pow) for pow in range(len(x) + 1)]

    out = []
    for index, _n_feature in enumerate(n_feature):
        if (index + 1) == len(n_feature):
            feature = x[index - 1]
            stride_size = (2, 2)
        else:
            feature = x[index]
            stride_size = (1, 1)
        o = tf.keras.layers.Conv2D(_n_feature, 3, strides = stride_size, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(feature)
        o = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(o)
        o = tf.keras.layers.Activation(tf.keras.activations.relu)(o)
        out.append(o)
    return out

def hr_module(x, n_branch = 4, shortcut = False, **kwargs):
    if not isinstance(x, list):
        x = [x]
    n_feature = [tf.keras.backend.int_shape(_x)[-1] for _x in x]

    out = list(x)
    # branch
    for _ in range(n_branch):
        for index, _n_feature in enumerate(n_feature):
            out[index] = hr_basic_block(out[index], _n_feature, shortcut = shortcut, **kwargs)

    # fuse
    outs = []
    for index, _n_feature in enumerate(n_feature):
        _out = []
        for seq, o in enumerate(out):
            if seq < index:
                for k in range(index - seq):
                    o = tf.keras.layers.Conv2D(_n_feature, 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(o)
                    o = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(o)
                    if k != (index - seq - 1):
                        o = tf.keras.layers.Activation(tf.keras.activations.relu)(o)
            elif seq == index:
                pass
            else: #index < seq
                o = tf.keras.layers.Conv2D(_n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(o)
                o = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(o)
                upsample_size = [2 ** (seq - index)] * 2
                o = tf.keras.layers.UpSampling2D(upsample_size, interpolation = "bilinear")(o)
            _out.append(o)
        _out = tf.keras.layers.Add()(_out)
        _out = tf.keras.layers.Activation(tf.keras.activations.relu)(_out)
        outs.append(_out)
    return outs

def object_attention_context(feature, prob, scale = 1):
    feature = tf.keras.layers.Reshape([-1, tf.keras.backend.int_shape(feature)[-1]])(feature)
    feature = tf.keras.layers.Permute([2, 1])(feature)
    prob = tf.keras.layers.Reshape([-1, tf.keras.backend.int_shape(prob)[-1]])(prob)
    prob = tf.keras.activations.softmax(prob * scale, axis = -2)
    context = tf.keras.layers.Dot([2, 1])([feature, prob]) #batch x featrue ch x prob ch
    context = tf.keras.backend.expand_dims(context, axis = -2) #batch x featrue ch x 1 x prob ch
    return context

def object_attention(feature, prob, n_feature = 256, scale = 1, **kwargs):
    if 1 < scale:
        feature = tf.keras.layers.MaxPooling2D((scale, scale), padding = "same")(feature)
    proxy_context = object_attention_context(feature, prob, scale)

    query = tf.keras.layers.Conv2D(n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(feature)
    query = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(query)
    query = tf.keras.layers.Activation(tf.keras.activations.relu)(query)
    query = tf.keras.layers.Conv2D(n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(query)
    query = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(query)
    query = tf.keras.layers.Activation(tf.keras.activations.relu)(query)
    query = tf.keras.layers.Reshape([-1, tf.keras.backend.int_shape(query)[-1]])(query)

    key = tf.keras.layers.Conv2D(n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(proxy_context)
    key = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(key)
    key = tf.keras.layers.Activation(tf.keras.activations.relu)(key)
    key = tf.keras.layers.Conv2D(n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(key)
    key = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(key)
    key = tf.keras.layers.Activation(tf.keras.activations.relu)(key)
    key = tf.keras.layers.Reshape([-1, tf.keras.backend.int_shape(key)[-1]])(key)
    key = tf.keras.layers.Permute([2, 1])(key)

    value = tf.keras.layers.Conv2D(n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(proxy_context)
    value = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(value)
    value = tf.keras.layers.Activation(tf.keras.activations.relu)(value)
    value = tf.keras.layers.Reshape([-1, tf.keras.backend.int_shape(value)[-1]])(value)

    sim = tf.keras.layers.Dot([2, 1])([query, key])
    sim = sim * (n_feature ** -0.5)
    sim = tf.keras.activations.softmax(sim)

    context = tf.keras.layers.Dot([2, 1])([sim, value])
    context = tf.keras.layers.Reshape(tf.keras.backend.int_shape(feature)[-3:-1] + (n_feature,))(context)
    context = tf.keras.layers.Conv2D(tf.keras.backend.int_shape(feature)[-1], 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(context)
    context = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(context)
    context = tf.keras.layers.Activation(tf.keras.activations.relu)(context)
    
    if 1 < scale:
        context = tf.keras.layers.UpSampling2D((scale, scale))(context)
    return context

def ocr_module(feature, prob, n_feature = 512, n_attention_feature = 256, dropout_rate = 0.05, scale = 1, **kwargs):
    context = object_attention(feature, prob, n_attention_feature, scale, **kwargs)
    out = tf.keras.layers.Concatenate(axis = -1)([feature, context])
    out = tf.keras.layers.Conv2D(n_feature, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
    out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    return out

def hrnet_v2(x, n_class = 35, include_top = True, n_channel = 48, n_module = [1, 4, 3], n_branch = [2, 3, 4], stage1_channel = 64, stage1_module = 1, ocr_feature_channel = 512, ocr_attention_channel = 256, ocr_dropout_rate = 0.05, ocr_scale = 1, mode = "ocr", **kwargs):
    if mode not in ("seg", "clsf", "ocr"):
        raise ValueError("unknown mode '{0}'".format(mode))
    if isinstance(n_channel, int):
        n_channel = [n_channel * (2 ** pow) for pow in range(4)]
    if isinstance(n_module, int):
        n_module = [n_module] * 3
    if isinstance(n_branch, int):
        n_branch = [n_branch] * 3

    if not isinstance(x, list):
        #Stem
        out = tf.keras.layers.Conv2D(64, 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(x)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
        out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
        out = tf.keras.layers.Conv2D(64, 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
        out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
        
        #Stage1
        out = hr_bottleneck_block(out, stage1_channel, shortcut = True, **kwargs)
        for _ in range(1, stage1_module):
            out = hr_bottleneck_block(out, stage1_channel, **kwargs)
    else:
        x = list(x)
        shape = tf.keras.backend.int_shape(x[0])[-3:-1]
        for index in range(len(x)):
            o = tf.keras.layers.Conv2D(n_channel[index], 3, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(x[index])
            o = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(o)
            o = tf.keras.layers.Activation(tf.keras.activations.relu)(o)
            if 0 < index:
                target_size = np.divide(shape, 2 ** index).astype(np.int32)
                o = tf.image.resize(o, target_size, method = "bilinear")
            x[index] = o

    #Stage2: transition -> branch -> fuse x2
    # transition
    if not isinstance(x, list):
        out = hr_transition_block(out, n_channel[:2], **kwargs)
    else:
        out = x[:2]
    # branch & fuse
    for _ in range(n_module[0]):
        out = hr_module(out, n_branch = n_branch[0], shortcut = False, **kwargs)

    #Stage3: transition -> branch -> fuse x3
    # transition
    if not isinstance(x, list):
        out = hr_transition_block(out, n_channel[:3], **kwargs)
    else:
        out = out + x[2:3]
    # branch & fuse
    for _ in range(n_module[1]):
        out = hr_module(out, n_branch = n_branch[1], shortcut = False, **kwargs)

    #Stage4: transition -> branch -> fuse x4
    # transition
    if not isinstance(x, list):
        out = hr_transition_block(out, n_channel[:4], **kwargs)
    else:
        out = out + x[3:4]
    # branch & fuse
    for _ in range(n_module[2]):
        out = hr_module(out, n_branch = n_branch[2], shortcut = False, **kwargs)

    if mode != "clsf":
        #Concatenate
        for index in range(1, len(out)):
            upsample_size = np.divide(tf.keras.backend.int_shape(out[0])[-3:-1], tf.keras.backend.int_shape(out[index])[-3:-1]).astype(np.int32)
            out[index] = tf.keras.layers.UpSampling2D(upsample_size, interpolation = "bilinear")(out[index])
        out = tf.keras.layers.Concatenate(axis = -1)(out)
    else:
        #Add
        o = hr_bottleneck_block(out[0], tf.keras.backend.int_shape(out[0])[-1], **kwargs)
        for _o in out[1:]:
            ch = tf.keras.backend.int_shape(_o)[-1]
            _o = hr_bottleneck_block(_o, ch, **kwargs)
            o = tf.keras.layers.Conv2D(ch, 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(o)
            o = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(o)
            o = tf.keras.layers.Activation(tf.keras.activations.relu)(o)
            o = tf.keras.layers.Add()([o, _o])
        out = o

    if include_top:
        if mode != "clsf":
            out_aux = tf.keras.layers.Conv2D(tf.keras.backend.int_shape(out)[-1], 1, use_bias = True, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
            out_aux = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out_aux)
            out_aux = tf.keras.layers.Activation(tf.keras.activations.relu)(out_aux)
            out_aux = tf.keras.layers.Conv2D(n_class, 1, use_bias = True, kernel_initializer = "he_normal", name = "logits", **kwargs)(out_aux)
            if mode == "ocr":
                out = tf.keras.layers.Conv2D(ocr_feature_channel, 3, padding = "same", use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
                out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
                out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
                out = ocr_module(out, out_aux, ocr_feature_channel, ocr_attention_channel, ocr_dropout_rate, ocr_scale, **kwargs)
                out = tf.keras.layers.Conv2D(n_class, 1, use_bias = True, kernel_initializer = "he_normal", name = "logits_ocr", **kwargs)(out)
            else:
                out = out_aux
        else:
            out = tf.keras.layers.Conv2D(2048, 1, use_bias = False, kernel_initializer = "he_normal", bias_initializer = "zeros")(out)
            out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)(out)
            out = tf.keras.layers.Activation(tf.keras.activations.relu)(out)
            out = tf.keras.layers.GlobalAveragePooling2D()(out)
            out = tf.keras.layers.Dense(n_class, use_bias = True, name = "logits")(out)
    return out

def hrnet18_v2(x, n_class = 35, include_top = True, mode = "ocr"):
    out = hrnet_v2(x, n_class, include_top, n_channel = [18, 36, 72, 144], n_module = [1, 4, 3], n_branch = [2, 3, 4], stage1_channel = 64, stage1_module = 1, ocr_feature_channel = 512, ocr_attention_channel = 256, ocr_dropout_rate = 0.05, ocr_scale = 1, mode = mode)
    return out

def hrnet32_v2(x, n_class = 35, include_top = True, mode = "ocr"):
    out = hrnet_v2(x, n_class, include_top, n_channel = [32, 64, 128, 256], n_module = [1, 4, 3], n_branch = [2, 3, 4], stage1_channel = 64, stage1_module = 1, ocr_feature_channel = 512, ocr_attention_channel = 256, ocr_dropout_rate = 0.05, ocr_scale = 1, mode = mode)
    return out

def hrnet48_v2(x, n_class = 35, include_top = True, mode = "ocr"):
    out = hrnet_v2(img_input, n_class, include_top, n_channel = [48, 94, 192, 384], n_module = [1, 4, 3], n_branch = [2, 3, 4], stage1_channel = 64, stage1_module = 1, ocr_feature_channel = 512, ocr_attention_channel = 256, ocr_dropout_rate = 0.05, ocr_scale = 1, mode = mode)
    return out