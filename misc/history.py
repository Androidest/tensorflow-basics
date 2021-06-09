def display_history(history, curve_type='loss'):
    history = history.history
    if len(history) == 0:
        return
    
    if not isinstance(curve_type, list):
        curve_type = [curve_type]

    for ctype in curve_type:
        plt.title('model ' + ctype)
        plt.xlabel('epoch')
        plt.ylabel(ctype)

        names = []
        for curve_name in history:
            if ctype in curve_name:
                plt.plot(history[curve_name])
                names.append(curve_name)
        plt.legend(names)
        plt.show()
