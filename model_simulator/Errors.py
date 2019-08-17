# define Python user-defined exceptions


class ErrorPSD(Exception):
    def __init__(self, *args):
        # *args is used to get a list of the parameters passed in
        self.args = [a for a in args]
        print ("allo")

# try:
#     raise ErrorWithArgs(1, "text", "some more text")
# except ErrorWithArgs as e:
#     print("%d: %s - %s" % (e.args[0], e.args[1], e.args[2]))

