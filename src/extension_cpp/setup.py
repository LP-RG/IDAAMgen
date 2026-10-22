from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mat_mul',
    ext_modules=[
        CUDAExtension(
            name='mat_mul',
            sources=['matmul.cu'],
            extra_compile_args={
                'nvcc': ['-g', '-Xptxas', '-v']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)