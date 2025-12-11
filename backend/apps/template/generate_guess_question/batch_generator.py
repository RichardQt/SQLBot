"""批量生成推荐问题的模板生成器"""
from apps.template.template import get_base_template


def get_guess_batch_template():
    """获取批量生成推荐问题的模板"""
    template = get_base_template()
    return template['template']['guess_batch']
