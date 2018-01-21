def get_lr_schedule(idx):
    if idx == 1:
        return [1e-2]
    elif idx == 2:
        return [5e-3]
    elif idx == 3:
        return [1e-3]
    elif idx == 4:
        return [5e-4]
    elif idx == 5:
        return [1e-4]
    elif idx == 6:
        return [5e-5]
    elif idx == 7:
        return [1e-5]
