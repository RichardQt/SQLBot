from apps.template.template import get_base_template

def get_context_analysis_template():
    return get_base_template()['template']['context_analysis']

def get_question_complete_template():
    return get_base_template()['template']['question_complete']
