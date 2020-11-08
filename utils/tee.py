def tee(msg, file=None):
    print(msg)
    if file is not None:
        print(msg, file=file)
