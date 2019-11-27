g_conv2linear = {
    'input_shape': (512, 2, 2),
    'layers': [
        # {'layer': 'conv', 'args': (64, 3, 1, 0), 'bn': False, 'act': 'ReLU'},
        # {'layer': 'conv', 'args': (32, 3, 1, 0), 'bn': False, 'act': 'ReLU'},
        {'layer': 'flatten'},
    ],
    'local_task_idx': (None, None)
}

g_linear2scalar = {
    'input_shape': (2048+64, ),
    'layers': [
        {'layer': 'linear', 'args': (512, ), 'bn': False, 'act': 'ReLU'},
        {'layer': 'linear', 'args': (512, ), 'bn': False, 'act': 'ReLU'},
        {'layer': 'linear', 'args': (1, ), 'bn': False, 'act': None}
    ],
    'local_task_idx': (None, None)
}

l_conv2conv = {
    'input_shape': (256+64, 4, 4),
    'layers': [
        {'layer': 'conv', 'args': (512, 1, 1, 0), 'bn': False, 'act': 'ReLU'},
        {'layer': 'conv', 'args': (512, 1, 1, 0), 'bn': False, 'act': 'ReLU'},
        {'layer': 'conv', 'args': (1, 1, 1, 0), 'bn': False, 'act':  None},
    ],
    'local_task_idx': (None, None)
}

p_linear2scalar = {
    'input_shape': (64, ),
    'layers': [
        {'layer': 'linear', 'args': (1000, ), 'bn': False, 'act': 'ReLU'},
        {'layer': 'linear', 'args': (200, ), 'bn': False, 'act': 'ReLU'},
        {'layer': 'linear', 'args': (1, ), 'bn': False, 'act': None}
    ],
    'local_task_idx': (None, None)
}
