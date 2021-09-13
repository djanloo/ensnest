'''profiler setup found on youtube explained
by a holy indian guy

1.a run this code onto the file you want to compile
1.b set the necessary things on the profiler.py file
    then create a dummy function that will inevitably
    call the function of interest (say, dummy())
2. switch to ipython
3. run like:
    %load_ext line_profiler
    import line_profiler
    import module_of_interest
    import profiler

    %lprun -f module_of_interest.f_of_interest profiler.dummy()
'''

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True #enables profiling
directive_defaults['binding'] = True
directive_defaults ['language_level'] = 3 #enable python3 semantics
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

extensions = [Extension("cy_samplers", ["cy_samplers.pyx"],define_macros=[('CYTHON_TRACE', '1')])]

setup(ext_modules =cythonize(extensions, compiler_directives = {'linetrace': True},annotate= True)) #linetrace compiler directive to enable both profiling and line tracing
