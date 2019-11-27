naive_linear2scalar = {
    'input_shape': (64, ),
    'layers': [
        {'layer': 'linear', 'args': (200, ), 'bn': True, 'act': 'ReLU'},
        {'layer': 'linear', 'args': (1, ), 'bn': False, 'act': None}
    ],
    'local_task_idx': (None, None)
}