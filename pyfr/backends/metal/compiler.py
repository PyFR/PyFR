from pyfr.backends.metal.util import call_


class MetalCompiler:
    def __init__(self, backend):
        self.backend = backend
        self.dev = backend.dev

    def build_program(self, src):
        from Metal import MTLCompileOptions

        opts = MTLCompileOptions.new()
        opts.setFastMathEnabled_(True)

        lib, err = call_(self.dev, 'newLibraryWith', source=src,
                         options=opts, error=None)
        if err is not None:
            raise ValueError(f'Metal compile error: {err}')

        return lib

    def build_pipeline(self, src, name):
        from Metal import MTLComputePipelineDescriptor

        # Compile the library and get function
        lib = self.build_program(src)
        func = call_(lib, 'newFunctionWith', name=name)
        if func is None:
            raise KeyError(f'Unable to load function {name}')

        # Create pipeline descriptor
        pipe_desc = MTLComputePipelineDescriptor.alloc().init()
        pipe_desc.setComputeFunction_(func)
        pipe_desc.setThreadGroupSizeIsMultipleOfThreadExecutionWidth_(True)

        # Create the pipeline state
        pipeline, err = call_(self.dev, 'newComputePipelineStateWith',
                              descriptor=pipe_desc, error=None)
        if err is not None:
            raise ValueError(f'Pipeline creation error: {err}')

        return pipeline, func
