from .basic import PSMNet as basic
from .stackhourglass import PSMNet as stackhourglass


def psmnet(maxdisp, framework):
    expect_framework = ['basic', 'stackhourglass']
    assert framework in expect_framework
    return eval(framework)(maxdisp)
