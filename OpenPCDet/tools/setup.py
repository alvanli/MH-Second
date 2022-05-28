import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    print("heyo", [os.path.join(*module.split('.'), src) for src in sources])
    return cuda_ext

if __name__ == '__main__':
    version = '0.0.1+%s' % get_git_commit_number()

    setup(
        name='JustSECOND',
        version=version,
        description='JustSECOND is just SECOND, based on implementation from OpenPCDet',
        install_requires=[
            'numpy<=1.20',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'SharedArray',
            # 'spconv',  # spconv has different names depending on the cuda version
        ],

        author='Alvin',
        author_email='ch35li@watonomous.ca',
        license='Apache License 2.0',
        packages=find_packages(os.path.abspath(os.path.dirname(__file__)), include=["JustSECOND", "JustSECOND.*"], exclude=['scripts', 'eval_utils', 'train_utils', 'visual_utils']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='JustSECOND',
                sources=[
                    'cuda/src/iou3d_cpu.cpp',
                    'cuda/src/iou3d_nms_api.cpp',
                    'cuda/src/iou3d_nms.cpp',
                    'cuda/src/iou3d_nms_kernel.cu',
                ]
            )
        ],
    )
