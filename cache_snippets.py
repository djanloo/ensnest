def cachedproperty(func):
    " Used on methods to convert them to methods that replace themselves\
        with their return value once they are called. "

    def cache(*args):
        self = args[0] # Reference to the class who owns the method
        funcname = func.__name__
        ret_value = func(self)
        setattr(self, funcname, ret_value) # Replace the function with its value
        return ret_value # Return the result of the function

    return property(cache)
