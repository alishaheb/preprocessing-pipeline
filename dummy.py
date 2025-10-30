def start_dec(func):
    print ("dummy")

    def wapper():
        print ("start_wrapper")
        func()
        print("wrapper")
    return wapper
def print_name():
    print ("ali")
print_name=start_dec(print_name)
print_name()